"""
Closed-set EEG subject identification demo.

Given one held-out EEG trial, predict which enrolled subject produced it.

Required files created from preprocressing_pipline.py:
--------------
X_raw.npy          shape: (trials, channels, samples)
y_id_encoded.npy   shape: (trials,)

Optional display-label files
----------------------------
y_identity.npy       shape: (trials,), string subject IDs
subject_classes.npy  shape: (n_subjects,), names corresponding to encoded IDs

Outputs (Saved inside 'subject_identification_results' directory)
-------
subject_identification_summary.csv
subject_identification_predictions.csv
subject_identification_report.txt
subject_identification_confusion.csv
subject_identification_breakdown.png  
fold_performance_distribution.png     
subject_identifier_model.joblib
subject_feature_metadata.npz
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import matplotlib.pyplot as plt
import seaborn as sns


# Configuration


BASE_OUTPUT_DIR = Path("./subject_identification_results")

X_FILE = "X_raw.npy"
ENCODED_ID_FILE = "y_id_encoded.npy"
IDENTITY_FILE = "y_identity.npy"
SUBJECT_CLASSES_FILE = "subject_classes.npy"

SFREQ = 1000.0
N_SPLITS = 5
RANDOM_STATE = 42
N_TREES = 300  # Reduced slightly to balance performance with nested Grid Search
PSD_BATCH_SIZE = 64

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "gamma": (30.0, 45.0),
}

WELCH_NPERSEG = 512
WELCH_OVERLAP_FRACTION = 0.50
EPSILON = np.finfo(np.float64).eps

SUMMARY_CSV = "subject_identification_summary.csv"
PREDICTIONS_CSV = "subject_identification_predictions.csv"
REPORT_TXT = "subject_identification_report.txt"
CONFUSION_CSV = "subject_identification_confusion.csv"
BREAKDOWN_PNG = "subject_identification_breakdown.png"
DISTRIBUTION_PNG = "fold_performance_distribution.png"  # New metric distribution plot
MODEL_FILE = "subject_identifier_model.joblib"
FEATURE_METADATA_FILE = "subject_feature_metadata.npz"



# Loading and validation

def load_required_array(filename: str, allow_pickle: bool = False) -> np.ndarray:
    path = Path(filename)
    if not path.exists():
        sys.exit(f"Required file not found: {filename}\nRun the preprocessing pipeline first.")
    try:
        return np.load(path, allow_pickle=allow_pickle)
    except Exception as exc:
        sys.exit(f"Could not load {filename}: {exc}")


def load_subject_targets(n_trials: int) -> tuple[np.ndarray, np.ndarray]:
    if Path(IDENTITY_FILE).exists():
        identities = np.load(IDENTITY_FILE, allow_pickle=True).astype(str)
        if identities.ndim != 1 or len(identities) != n_trials:
            sys.exit(f"{IDENTITY_FILE} must have shape ({n_trials},), received {identities.shape}.")
        class_names, y_subject = np.unique(identities, return_inverse=True)
        return y_subject.astype(int), class_names.astype(str)

    encoded = load_required_array(ENCODED_ID_FILE, allow_pickle=True)
    if encoded.ndim != 1 or len(encoded) != n_trials:
        sys.exit(f"{ENCODED_ID_FILE} must have shape ({n_trials},), received {encoded.shape}.")

    encoded = encoded.astype(int)
    unique_values = np.unique(encoded)
    value_to_class = {int(value): class_index for class_index, value in enumerate(unique_values)}
    y_subject = np.asarray([value_to_class[int(value)] for value in encoded], dtype=int)

    if Path(SUBJECT_CLASSES_FILE).exists():
        supplied_names = np.load(SUBJECT_CLASSES_FILE, allow_pickle=True).astype(str)
        if len(supplied_names) == len(unique_values):
            class_names = supplied_names
        else:
            print(f"Warning: {SUBJECT_CLASSES_FILE} contains {len(supplied_names)} names for {len(unique_values)} classes. IDs used.")
            class_names = np.asarray([f"subject_{value}" for value in unique_values], dtype=str)
    else:
        class_names = np.asarray([f"subject_{value}" for value in unique_values], dtype=str)

    return y_subject, class_names


def validate_inputs(X: np.ndarray, y_subject: np.ndarray) -> None:
    if X.ndim != 3:
        sys.exit(f"X_raw.npy must have shape (trials, channels, samples). Received {X.shape}.")
    if len(y_subject) != len(X):
        sys.exit(f"X contains {len(X)} trials but y contains {len(y_subject)} labels.")
    if not np.issubdtype(X.dtype, np.number):
        sys.exit("X_raw.npy must contain numeric data.")
    if not np.all(np.isfinite(X)):
        sys.exit("X_raw.npy contains NaN or infinite values.")
    if X.shape[-1] < 8:
        sys.exit("Trials contain too few samples for spectral analysis.")


# Features & Folds


def integrate_psd(psd: np.ndarray, frequencies: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected_frequencies = frequencies[mask]
    if selected_frequencies.size == 0:
        raise ValueError("A requested frequency band contains no FFT bins.")
    if selected_frequencies.size == 1:
        return psd[..., mask][..., 0]
    if hasattr(np, "trapezoid"):
        return np.trapezoid(psd[..., mask], selected_frequencies, axis=-1)
    return np.trapz(psd[..., mask], selected_frequencies, axis=-1)


def extract_subject_features(X: np.ndarray, sfreq: float, batch_size: int = 64) -> tuple[np.ndarray, list[str]]:
    n_trials, n_channels, n_samples = X.shape
    nperseg = min(WELCH_NPERSEG, n_samples)
    noverlap = min(int(nperseg * WELCH_OVERLAP_FRACTION), nperseg - 1)

    feature_batches: list[np.ndarray] = []
    band_names = list(BANDS.keys())

    for start in range(0, n_trials, batch_size):
        stop = min(start + batch_size, n_trials)
        batch = np.asarray(X[start:stop], dtype=np.float64)

        frequencies, psd = welch(batch, fs=sfreq, window="hann", nperseg=nperseg, noverlap=noverlap, detrend="constant", return_onesided=True, scaling="density", axis=-1)
        total_mask = (frequencies >= 1.0) & (frequencies <= 45.0)
        total_power = integrate_psd(psd, frequencies, total_mask)

        absolute_bands = []
        for low, high in BANDS.values():
            band_mask = (frequencies >= low) & (frequencies <= high)
            absolute_bands.append(integrate_psd(psd, frequencies, band_mask))

        absolute_bands = np.stack(absolute_bands, axis=-1)
        log_absolute = np.log10(np.maximum(absolute_bands, EPSILON))
        relative = absolute_bands / np.maximum(total_power[..., np.newaxis], EPSILON)

        channel_mean = np.mean(batch, axis=-1)
        channel_std = np.std(batch, axis=-1)
        channel_rms = np.sqrt(np.mean(batch**2, axis=-1))
        channel_ptp = np.ptp(batch, axis=-1)

        channel_statistics = np.stack([channel_mean, channel_std, channel_rms, channel_ptp], axis=-1)

        global_features = np.column_stack([
            np.mean(batch, axis=(1, 2)),
            np.std(batch, axis=(1, 2)),
            np.sqrt(np.mean(batch**2, axis=(1, 2))),
            np.min(batch, axis=(1, 2)),
            np.max(batch, axis=(1, 2)),
            np.mean(np.abs(batch), axis=(1, 2)),
        ])

        batch_features = np.concatenate([
            log_absolute.reshape(stop - start, -1),
            relative.reshape(stop - start, -1),
            channel_statistics.reshape(stop - start, -1),
            global_features,
        ], axis=1)

        feature_batches.append(batch_features)
        print(f"\rExtracted subject features for trials {stop}/{n_trials}", end="", flush=True)

    print()
    features = np.concatenate(feature_batches, axis=0).astype(np.float32)
    
    feature_names: list[str] = []
    for channel in range(n_channels):
        for band in band_names: feature_names.append(f"ch{channel}_log_abs_{band}")
    for channel in range(n_channels):
        for band in band_names: feature_names.append(f"ch{channel}_relative_{band}")
    for channel in range(n_channels):
        for statistic in ("mean", "std", "rms", "ptp"): feature_names.append(f"ch{channel}_{statistic}")

    feature_names.extend(["global_mean", "global_std", "global_rms", "global_min", "global_max", "global_mean_abs"])
    return features, feature_names


def hash_trial(trial: np.ndarray) -> str:
    canonical = np.asarray(trial, dtype="<f4", order="C")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def make_duplicate_safe_block_folds(X: np.ndarray, y_subject: np.ndarray, n_splits: int) -> tuple[np.ndarray, dict[int, int]]:
    fold_id = np.full(len(X), -1, dtype=np.int16)
    unique_groups_per_subject: dict[int, int] = {}

    for subject in np.unique(y_subject):
        subject_indices = np.flatnonzero(y_subject == subject)
        hash_to_indices: dict[str, list[int]] = {}

        for global_index in subject_indices:
            digest = hash_trial(X[global_index])
            hash_to_indices.setdefault(digest, []).append(int(global_index))

        ordered_duplicate_groups = sorted(hash_to_indices.values(), key=lambda indices: min(indices))
        unique_groups_per_subject[int(subject)] = len(ordered_duplicate_groups)

        if len(ordered_duplicate_groups) < n_splits:
            raise RuntimeError(f"Subject {subject} has only {len(ordered_duplicate_groups)} groups; {n_splits} folds impossible.")

        group_positions = np.arange(len(ordered_duplicate_groups))
        contiguous_blocks = np.array_split(group_positions, n_splits)

        for current_fold, block_positions in enumerate(contiguous_blocks):
            for group_position in block_positions:
                indices = ordered_duplicate_groups[int(group_position)]
                fold_id[indices] = current_fold

    return fold_id, unique_groups_per_subject


# Visualizations


def save_subject_plots(matrix: np.ndarray, class_names: np.ndarray, save_path: Path) -> None:
    matrix_perc = (matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]) * 100
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, max(6, len(class_names) * 0.5)))
    
    left_accum = np.zeros(len(class_names))
    colors = sns.color_palette("Spectral", n_colors=len(class_names))
    
    for idx, pred_name in enumerate(class_names):
        ax.barh(class_names, matrix_perc[:, idx], left=left_accum, label=f"Guessed as {pred_name}", color=colors[idx])
        left_accum += matrix_perc[:, idx]
        
    ax.set_xlabel('Percentage of Guesses (%)')
    ax.set_ylabel('True Enrolled Subject')
    ax.set_title('Model Prediction Breakdown Per Subject (Across All Folds)')
    ax.set_xlim(0, 100)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", title="Predictions")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_distribution_plot(fold_rows: list[dict], save_path: Path) -> None:
    """Generates a combined summary Box plot and jittered Swarm plot tracking cross-fold metrics."""
    df = pd.DataFrame(fold_rows)
    df_melted = df.melt(id_vars=["fold"], value_vars=["accuracy_pct", "balanced_accuracy_pct"], 
                        var_name="Metric", value_name="Percentage")
    
    df_melted["Metric"] = df_melted["Metric"].replace({
        "accuracy_pct": "Raw Accuracy", 
        "balanced_accuracy_pct": "Balanced Accuracy"
    })
    
    plt.figure(figsize=(7, 6))
    sns.set_theme(style="whitegrid")
    
    # Draw summary distribution box plots (The "Umbrella")
    sns.boxplot(x="Metric", y="Percentage", data=df_melted, width=0.4, 
                palette="Pastel1", fliersize=0, boxprops=dict(alpha=0.6))
    
    # Overlay individual raw fold performance data points (The "Rain")
    sns.swarmplot(x="Metric", y="Percentage", data=df_melted, color="0.25", size=7, linewidth=1)
    
    plt.ylabel('Performance (%)')
    plt.xlabel('')
    plt.title('Distribution of Performance Metrics Across Folds')
    plt.ylim(max(0, df_melted["Percentage"].min() - 10), 105)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def top_k_accuracy(y_true: np.ndarray, probabilities: np.ndarray, k: int) -> float:
    top_k = np.argpartition(probabilities, kth=-k, axis=1)[:, -k:]
    return float(np.mean([int(true_label) in row for true_label, row in zip(y_true, top_k)]))



# Main Routine

def main() -> None:
    print("Initializing EEG Subject Identifier...")
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X = load_required_array(X_FILE)
    y_subject, class_names = load_subject_targets(len(X))
    validate_inputs(X, y_subject)

    n_subjects = len(class_names)
    
    print("\nExtracting spectral and amplitude features...")
    X_features, feature_names = extract_subject_features(X, sfreq=SFREQ, batch_size=PSD_BATCH_SIZE)
    fold_id, unique_groups_per_subject = make_duplicate_safe_block_folds(X, y_subject, N_SPLITS)

    oof_predictions = np.full(len(X), -1, dtype=int)
    oof_probabilities = np.zeros((len(X), n_subjects), dtype=np.float64)
    fold_rows: list[dict] = []

    # Hyperparameter tuning setup (Grid Search parameters isolated inside folds)
    param_grid = {
        'max_features': ['sqrt', 'log2'],
        'min_samples_leaf': [1, 2]
    }

    print("\nRunning block-based out-of-fold evaluation with internal Hyperparameter Tuning...")
    for current_fold in range(N_SPLITS):
        train_indices = np.flatnonzero(fold_id != current_fold)
        test_indices = np.flatnonzero(fold_id == current_fold)

        # Baseline estimator
        rf = RandomForestClassifier(n_estimators=N_TREES, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced_subsample")
        
        # Grid Search strictly over training folds to prevent data leakage
        clf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='balanced_accuracy')
        clf.fit(X_features[train_indices], y_subject[train_indices])
        
        best_model = clf.best_estimator_

        predictions = best_model.predict(X_features[test_indices])
        local_probabilities = best_model.predict_proba(X_features[test_indices])

        oof_predictions[test_indices] = predictions
        for local_column, class_value in enumerate(best_model.classes_):
            oof_probabilities[test_indices, int(class_value)] = local_probabilities[:, local_column]

        fold_accuracy = accuracy_score(y_subject[test_indices], predictions)
        fold_balanced = balanced_accuracy_score(y_subject[test_indices], predictions)

        fold_rows.append({
            "fold": current_fold, "train_trials": len(train_indices), "test_trials": len(test_indices),
            "accuracy_pct": fold_accuracy * 100.0, "balanced_accuracy_pct": fold_balanced * 100.0
        })
        print(f"  Fold {current_fold + 1}/{N_SPLITS} complete. Best Params: {clf.best_params_}")

    overall_accuracy = accuracy_score(y_subject, oof_predictions)
    balanced_accuracy = balanced_accuracy_score(y_subject, oof_predictions)
    macro_f1 = f1_score(y_subject, oof_predictions, average="macro", zero_division=0)
    top3_accuracy = top_k_accuracy(y_subject, oof_probabilities, k=min(3, n_subjects))

    report = classification_report(y_subject, oof_predictions, labels=np.arange(n_subjects), target_names=class_names, digits=3, zero_division=0)
    matrix = confusion_matrix(y_subject, oof_predictions, labels=np.arange(n_subjects))

    # - Generate Visual Charts -
    print("\nGenerating per-subject and cross-fold metric distribution graphs...")
    save_subject_plots(matrix, class_names, BASE_OUTPUT_DIR / BREAKDOWN_PNG)
    save_distribution_plot(fold_rows, BASE_OUTPUT_DIR / DISTRIBUTION_PNG)

    # - Save Reports and Metadata -
    with open(BASE_OUTPUT_DIR / SUMMARY_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold", "train_trials", "test_trials", "accuracy_pct", "balanced_accuracy_pct"])
        writer.writeheader()
        writer.writerows(fold_rows)
        writer.writerow({"fold": "overall", "train_trials": "", "test_trials": len(X), "accuracy_pct": overall_accuracy * 100.0, "balanced_accuracy_pct": balanced_accuracy * 100.0})

    with open(BASE_OUTPUT_DIR / PREDICTIONS_CSV, "w", newline="", encoding="utf-8") as handle:
        fieldnames = ["trial_index", "fold", "true_subject", "predicted_subject", "correct", "predicted_probability", "top_1_subject", "top_1_probability", "top_2_subject", "top_2_probability", "top_3_subject", "top_3_probability"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trial_index in range(len(X)):
            probability_order = np.argsort(oof_probabilities[trial_index])[::-1]
            top_indices = list(probability_order[: min(3, n_subjects)]) + [probability_order[-1]] * (3 - min(3, n_subjects))
            predicted = int(oof_predictions[trial_index])
            true_value = int(y_subject[trial_index])
            writer.writerow({
                "trial_index": trial_index, "fold": int(fold_id[trial_index]), "true_subject": class_names[true_value], "predicted_subject": class_names[predicted],
                "correct": int(true_value == predicted), "predicted_probability": float(oof_probabilities[trial_index, predicted]),
                "top_1_subject": class_names[int(top_indices[0])], "top_1_probability": float(oof_probabilities[trial_index, int(top_indices[0])]),
                "top_2_subject": class_names[int(top_indices[1])], "top_2_probability": float(oof_probabilities[trial_index, int(top_indices[1])]),
                "top_3_subject": class_names[int(top_indices[2])], "top_3_probability": float(oof_probabilities[trial_index, int(top_indices[2])])
            })

    with open(BASE_OUTPUT_DIR / REPORT_TXT, "w", encoding="utf-8") as handle:
        handle.write(f"EEG SUBJECT IDENTIFICATION\n==========================\nOOF accuracy: {overall_accuracy * 100:.2f}%\nBalanced accuracy: {balanced_accuracy * 100:.2f}%\n\n" + report)

    with open(BASE_OUTPUT_DIR / CONFUSION_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted", *class_names.tolist()])
        for class_name, row in zip(class_names, matrix): writer.writerow([class_name, *row.tolist()])

    # Train final model payload across everything using grid search to find final configurations
    final_rf = RandomForestClassifier(n_estimators=N_TREES, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced_subsample")
    final_clf = GridSearchCV(estimator=final_rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='balanced_accuracy')
    final_clf.fit(X_features, y_subject)
    
    joblib.dump({"classifier": final_clf.best_estimator_, "class_names": class_names, "sfreq": SFREQ, "bands": BANDS, "feature_names": feature_names}, BASE_OUTPUT_DIR / MODEL_FILE)
    np.savez_compressed(BASE_OUTPUT_DIR / FEATURE_METADATA_FILE, class_names=class_names, feature_names=np.asarray(feature_names, dtype=str), fold_id=fold_id, oof_predictions=oof_predictions, oof_probabilities=oof_probabilities)

    print("\nSaved:")
    print(f"  {BASE_OUTPUT_DIR / BREAKDOWN_PNG} (Per-Subject Performance Graph)")
    print(f"  {BASE_OUTPUT_DIR / DISTRIBUTION_PNG} (Metric Cross-Fold Distribution Graph)")
    print(f"  {BASE_OUTPUT_DIR / SUMMARY_CSV}")


if __name__ == "__main__":
    main()
