"""
Preprocesses raw EEGLAB data into structured inputs for within-subject imagined-speech decoding.

This script outputs the four primary arrays required by the decoder:
    X_raw.npy          - Trial data matrix of shape (trials, channels, samples)
    y_id_encoded.npy   - Encoded subject identifiers of shape (trials,)
    y_word.npy         - Categorical word labels of shape (trials,)
    trial_group.npy    - Group labels of shape (trials,) used to enforce strict block/session CV split boundaries

Note: `trial_group.npy` defines independent acquisition blocks or sessions that must not cross-contaminate 
training and testing sets. You must explicitly set `--group-mode` based on your data structure.


Strict Validation & Integrity Checks
------------------------------------
To prevent data leakage,  or bad data from silently corrupting down-stream model training, 
the execution will deliberately abort if it encounters any of the following anomalies:
- Missing or ambiguous word-label keys or subject IDs.
- Inconsistent sampling rates or channel selections across recordings.
- Overlapping trial windows (violating window independence) or exact trial/file duplicates.
- Mismatched array sizes breaking the one-to-one mapping between trial data and labels.
- Insufficient independent blocks per subject (fewer than two groups), making cross-validation impossible.

"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
import scipy.io
from scipy.io.matlab import mat_struct
from sklearn.preprocessing import LabelEncoder



# Verified Defaults

DEFAULT_ROOT_DIR = Path("/path/to/kara_one_dataset")

DEFAULT_OUTPUT_DIR = DEFAULT_ROOT_DIR / "word_decoder_inputs"

TARGET_TIME_SAMPLES = 4000
EXPECTED_EEG_CHANNELS = 62
MATLAB_INDICES_ARE_ONE_BASED = True



CANONICAL_EEG_CHANNELS: list[str] | None = None

# Strict safety defaults.
FAIL_ON_OVERLAPPING_TRIALS = True
FAIL_ON_EXACT_TRIAL_DUPLICATES = True
REQUIRE_AT_LEAST_TWO_GROUPS_PER_SUBJECT = True
REQUIRE_ALL_WORDS_IN_EVERY_GROUP = False

SUBJECT_PATTERN = re.compile(r"^(?:MM\d+|P\d+)$", re.IGNORECASE)

mne.set_log_level("ERROR")



# Command Line Interface



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create leakage-resistant imagined-speech decoder inputs with "
            "verified run/block/session groups."
        )
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=DEFAULT_ROOT_DIR,
        help="Dataset root containing the subject folders and .set files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to <root-dir>/word_decoder_inputs. "
            "Run the decoder from this directory."
        ),
    )
    parser.add_argument(
        "--word-label-key",
        default=None,
        help=(
            "Verified MATLAB variable containing exactly one word/stimulus "
            "label per thinking trial. Required outside --inspect mode."
        ),
    )
    parser.add_argument(
        "--group-mode",
        choices=("set_file", "parent_folder"),
        default=None,
        help=(
            "Verified independent acquisition unit. 'set_file' makes each "
            ".set file one group; 'parent_folder' merges all .set files in "
            "the same directory into one group. Required outside --inspect."
        ),
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help=(
            "List recordings, candidate groups, MATLAB keys, and annotations "
            "without extracting or saving trials."
        ),
    )
    parser.add_argument(
        "--inspect-annotations",
        type=int,
        default=40,
        help="Maximum annotations printed per recording in --inspect mode.",
    )
    parser.add_argument(
        "--allow-single-group-subjects",
        action="store_true",
        help=(
            "Save subjects with only one acquisition group. Such subjects "
            "cannot support held-out-group validation and will be skipped or "
            "must be interpreted as preliminary."
        ),
    )
    parser.add_argument(
        "--require-all-words-per-group",
        action="store_true",
        help=(
            "Fail unless every acquisition group contains every word class. "
            "Useful when the decoder uses leave-one-group-out evaluation."
        ),
    )
    return parser.parse_args()


# General Helpers

def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "unnamed"


def calculate_file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def hash_eeg_array(
    eeg_matrix: np.ndarray,
    channel_names: list[str],
    sfreq: float,
    time_chunk: int = 100_000,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(tuple(eeg_matrix.shape)).encode("utf-8"))
    hasher.update("\n".join(channel_names).encode("utf-8"))
    hasher.update(np.asarray([sfreq], dtype="<f8").tobytes())

    for start in range(0, eeg_matrix.shape[1], time_chunk):
        stop = min(start + time_chunk, eeg_matrix.shape[1])
        chunk = np.asarray(
            eeg_matrix[:, start:stop],
            dtype="<f4",
            order="C",
        )
        hasher.update(chunk.tobytes())

    return hasher.hexdigest()


def hash_trial(trial: np.ndarray) -> str:
    canonical = np.asarray(trial, dtype="<f4", order="C")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def extract_subject_id(path: Path) -> str:
    matches = [part for part in path.parts if SUBJECT_PATTERN.fullmatch(part)]
    unique_matches = list(dict.fromkeys(matches))

    if len(unique_matches) != 1:
        raise ValueError(
            f"Expected exactly one subject ID in path {path}, but found "
            f"{unique_matches}. Adjust SUBJECT_PATTERN."
        )

    return unique_matches[0]


def extract_trial_group(
    set_path: Path,
    subject_id: str,
    root_dir: Path,
    group_mode: str,
) -> str:
    try:
        relative_path = set_path.resolve().relative_to(root_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"Recording is outside ROOT_DIR: {set_path}") from exc

    if group_mode == "set_file":
        group_component = relative_path.with_suffix("").as_posix()
    elif group_mode == "parent_folder":
        group_component = relative_path.parent.as_posix()
    else:
        raise ValueError(f"Unsupported group mode: {group_mode!r}")

    return safe_name(f"{subject_id}__{group_mode}__{group_component}")


def find_epoch_inds_file(set_path: Path) -> Path:
    search_levels = [set_path.parent, set_path.parent.parent]

    for directory in search_levels:
        matches = sorted(directory.glob("epoch_inds.mat"))
        if len(matches) == 1:
            return matches[0].resolve()
        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple epoch_inds.mat files found in {directory}:\n"
                + "\n".join(f"  {path}" for path in matches)
            )

    raise FileNotFoundError(
        f"No epoch_inds.mat was found beside or one level above {set_path}."
    )


def describe_mat_keys(mat_data: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in mat_data.items():
        if key.startswith("__"):
            continue
        lines.append(
            f"  {key}: shape={getattr(value, 'shape', None)}, "
            f"dtype={getattr(value, 'dtype', None)}"
        )
    return "\n".join(lines) or "  <no user variables found>"



# MATLAB Parsing

def collect_numeric_values(value: Any) -> list[float]:
    values: list[float] = []

    if isinstance(value, mat_struct):
        for field_name in value._fieldnames or []:
            values.extend(collect_numeric_values(getattr(value, field_name)))
        return values

    if isinstance(value, np.ndarray):
        for item in value.reshape(-1):
            values.extend(collect_numeric_values(item))
        return values

    if isinstance(value, np.generic):
        return collect_numeric_values(value.item())

    if isinstance(value, (int, float)) and np.isfinite(value):
        values.append(float(value))

    return values


def normalize_trial_bounds(raw_thinking_inds: Any) -> list[Any]:
    array = np.asarray(raw_thinking_inds)
    squeezed = array.squeeze()

    if squeezed.ndim == 0:
        return [squeezed.item()]

    if squeezed.dtype == object:
        all_scalar = all(
            np.asarray(item).ndim == 0 for item in squeezed.reshape(-1)
        )
        if squeezed.ndim == 2 and all_scalar:
            if squeezed.shape[1] <= 4:
                return [squeezed[row, :] for row in range(squeezed.shape[0])]
            if squeezed.shape[0] <= 4:
                return [squeezed[:, col] for col in range(squeezed.shape[1])]
        return list(squeezed.reshape(-1))

    if squeezed.ndim == 1:
        return [squeezed]

    if squeezed.ndim == 2:
        if squeezed.shape[1] <= 4:
            return [squeezed[row, :] for row in range(squeezed.shape[0])]
        if squeezed.shape[0] <= 4:
            return [squeezed[:, col] for col in range(squeezed.shape[1])]

    raise ValueError(
        "Unsupported thinking_inds layout: "
        f"shape={array.shape}, dtype={array.dtype}"
    )


def extract_start_sample(trial_bounds: Any) -> int:
    numeric_values = collect_numeric_values(trial_bounds)
    if not numeric_values:
        raise ValueError("Trial bounds contained no numeric sample index.")

    raw_start = numeric_values[0]
    rounded_start = int(round(raw_start))
    if not np.isclose(raw_start, rounded_start):
        raise ValueError(f"Non-integer sample index encountered: {raw_start}")

    return rounded_start - 1 if MATLAB_INDICES_ARE_ONE_BASED else rounded_start


def matlab_value_to_string(value: Any) -> str:
    current = value

    while isinstance(current, np.ndarray) and current.size == 1:
        current = current.reshape(-1)[0]

    if isinstance(current, bytes):
        return current.decode("utf-8").strip()
    if isinstance(current, str):
        return current.strip()
    if isinstance(current, mat_struct):
        raise ValueError(
            "A MATLAB struct was found where a scalar word label was expected."
        )

    if isinstance(current, np.ndarray):
        squeezed = current.squeeze()
        if squeezed.dtype.kind in {"U", "S"}:
            return "".join(
                str(item) for item in squeezed.reshape(-1).tolist()
            ).strip()
        if squeezed.size == 1:
            return matlab_value_to_string(squeezed.item())
        raise ValueError(
            "A non-scalar MATLAB array was found inside one label cell: "
            f"shape={current.shape}, dtype={current.dtype}"
        )

    if isinstance(current, np.generic):
        current = current.item()

    return str(current).strip()


def labels_from_matlab_array(value: Any, num_trials: int) -> np.ndarray:
    array = np.asarray(value)
    squeezed = array.squeeze()
    candidates: list[list[str]] = []

    if squeezed.dtype.kind in {"U", "S"}:
        if squeezed.ndim == 0:
            candidates.append([str(squeezed.item()).strip()])
        elif squeezed.ndim == 1:
            items = [str(item).strip() for item in squeezed.tolist()]
            candidates.append(items)
            if num_trials == 1 and all(len(item) <= 1 for item in items):
                candidates.append(["".join(items).strip()])
        elif squeezed.ndim == 2:
            row_labels = [
                "".join(str(item) for item in row).strip()
                for row in squeezed.tolist()
            ]
            col_labels = [
                "".join(str(item) for item in squeezed[:, col].tolist()).strip()
                for col in range(squeezed.shape[1])
            ]
            candidates.extend([row_labels, col_labels])
    else:
        object_array = np.asarray(value, dtype=object).squeeze()
        if object_array.ndim == 0:
            candidates.append([matlab_value_to_string(object_array.item())])
        else:
            candidates.append(
                [
                    matlab_value_to_string(item)
                    for item in object_array.reshape(-1)
                ]
            )

    valid = [candidate for candidate in candidates if len(candidate) == num_trials]
    unique_valid: list[list[str]] = []
    for candidate in valid:
        if candidate not in unique_valid:
            unique_valid.append(candidate)

    if len(unique_valid) != 1:
        lengths = [len(candidate) for candidate in candidates]
        raise ValueError(
            "Could not unambiguously parse exactly one label per trial. "
            f"Expected {num_trials}; candidate lengths were {lengths}."
        )

    labels = np.asarray(unique_valid[0], dtype=str)
    empty_positions = np.flatnonzero(np.char.strip(labels) == "")
    if len(empty_positions):
        raise ValueError(
            "Empty labels were found at trial positions "
            f"{empty_positions.tolist()}. Labels are not deleted because "
            "that would shift trial-label alignment."
        )

    return np.char.strip(labels)


def extract_word_labels(
    mat_data: dict[str, Any],
    num_trials: int,
    source_path: Path,
    word_label_key: str,
) -> np.ndarray:
    if word_label_key not in mat_data:
        raise KeyError(
            f"Verified word-label key {word_label_key!r} was not found in "
            f"{source_path}. Available variables:\n{describe_mat_keys(mat_data)}"
        )

    labels = labels_from_matlab_array(mat_data[word_label_key], num_trials)
    if len(labels) != num_trials:
        raise AssertionError("Internal label-length validation failed.")
    return labels



# EEG Helpers 

def select_and_validate_eeg_channels(
    raw: mne.io.BaseRaw,
    canonical_names: list[str] | None,
) -> tuple[np.ndarray, list[str]]:
    if len(set(raw.ch_names)) != len(raw.ch_names):
        raise ValueError("Duplicate channel names exist in the recording.")

    if canonical_names is None:
        eeg_picks = mne.pick_types(
            raw.info,
            meg=False,
            eeg=True,
            eog=False,
            ecg=False,
            emg=False,
            stim=False,
            misc=False,
            exclude=[],
        )
        detected_names = [raw.ch_names[index] for index in eeg_picks]
        if len(detected_names) != EXPECTED_EEG_CHANNELS:
            raise ValueError(
                f"MNE detected {len(detected_names)} EEG channels, but "
                f"EXPECTED_EEG_CHANNELS={EXPECTED_EEG_CHANNELS}. Set "
                "CANONICAL_EEG_CHANNELS explicitly rather than selecting the "
                "first N channels. Detected channels:\n  "
                + "\n  ".join(detected_names)
            )
        canonical_names = detected_names
    else:
        if len(canonical_names) != EXPECTED_EEG_CHANNELS:
            raise ValueError(
                f"CANONICAL_EEG_CHANNELS must contain exactly "
                f"{EXPECTED_EEG_CHANNELS} names."
            )
        if len(set(canonical_names)) != len(canonical_names):
            raise ValueError("CANONICAL_EEG_CHANNELS contains duplicates.")

    missing = [name for name in canonical_names if name not in raw.ch_names]
    if missing:
        raise ValueError(f"Recording is missing canonical channels: {missing}")

    channel_indices = [raw.ch_names.index(name) for name in canonical_names]
    channel_types = raw.get_channel_types(picks=channel_indices)
    non_eeg = [
        (name, channel_type)
        for name, channel_type in zip(canonical_names, channel_types)
        if channel_type != "eeg"
    ]
    if non_eeg:
        raise ValueError(
            "Canonical channels not typed as EEG by MNE: " + repr(non_eeg)
        )

    eeg_matrix = raw.get_data(picks=channel_indices)
    return eeg_matrix, list(canonical_names)



# Inspection Mode

def inspect_dataset(
    set_files: list[Path],
    root_dir: Path,
    annotation_limit: int,
) -> None:
    print(f"Discovered {len(set_files)} .set files beneath {root_dir}\n")

    rows: list[dict[str, Any]] = []
    seen_epoch_paths: set[Path] = set()

    for index, set_path in enumerate(set_files, start=1):
        subject_id = extract_subject_id(set_path)
        inds_path = find_epoch_inds_file(set_path)

        rows.append(
            {
                "number": index,
                "subject": subject_id,
                "set_file": str(set_path.relative_to(root_dir)),
                "set_file_group": extract_trial_group(
                    set_path, subject_id, root_dir, "set_file"
                ),
                "parent_folder_group": extract_trial_group(
                    set_path, subject_id, root_dir, "parent_folder"
                ),
                "epoch_inds": str(inds_path.relative_to(root_dir)),
            }
        )

        print("=" * 100)
        print(f"[{index}/{len(set_files)}] {set_path}")
        print(f"Subject: {subject_id}")
        print(
            "Candidate set-file group: "
            + extract_trial_group(set_path, subject_id, root_dir, "set_file")
        )
        print(
            "Candidate parent-folder group: "
            + extract_trial_group(
                set_path, subject_id, root_dir, "parent_folder"
            )
        )
        print(f"Epoch/label file: {inds_path}")

        if inds_path not in seen_epoch_paths:
            mat_data = scipy.io.loadmat(
                inds_path,
                squeeze_me=False,
                struct_as_record=False,
            )
            print("MATLAB variables:")
            print(describe_mat_keys(mat_data))
            seen_epoch_paths.add(inds_path)
        else:
            print("MATLAB variables: already printed for this shared file")

        raw = mne.io.read_raw_eeglab(
            str(set_path),
            preload=False,
            verbose=False,
        )
        descriptions, counts = np.unique(
            raw.annotations.description,
            return_counts=True,
        )
        print(f"Sampling frequency: {float(raw.info['sfreq']):g} Hz")
        print(f"Channels: {len(raw.ch_names)} total")
        print(f"Annotations: {len(raw.annotations)}")
        print("Annotation descriptions:")
        for description, count in zip(descriptions, counts):
            print(f"  {description!r}: {count}")

        if annotation_limit > 0:
            print(f"First {min(annotation_limit, len(raw.annotations))} annotations:")
            for ann_index, annotation in enumerate(
                raw.annotations[:annotation_limit]
            ):
                print(
                    f"  {ann_index:03d} | onset={annotation['onset']:.4f} | "
                    f"duration={annotation['duration']:.4f} | "
                    f"description={annotation['description']!r}"
                )

        del raw

    inspection_path = root_dir / "word_decoder_group_inspection.csv"
    pd.DataFrame(rows).to_csv(inspection_path, index=False)
    print("\nInspection complete.")
    print(f"Candidate grouping table saved to: {inspection_path}")
    print(
        "Choose --group-mode set_file only if each .set file is an "
        "independent run/session. Otherwise use parent_folder or adapt "
        "extract_trial_group() to a verified run/session field."
    )



# Dataset Audits

def audit_group_design(
    y_identity: np.ndarray,
    y_word: np.ndarray,
    trial_groups: np.ndarray,
    output_dir: Path,
    require_two_groups: bool,
    require_all_words_per_group: bool,
) -> None:
    group_subject_table = pd.crosstab(trial_groups, y_identity)
    mixed_groups = group_subject_table.gt(0).sum(axis=1) > 1
    if mixed_groups.any():
        raise RuntimeError(
            "A trial group contains more than one participant. Problem groups: "
            f"{group_subject_table.index[mixed_groups].tolist()}"
        )

    all_words = sorted(np.unique(y_word).tolist())
    group_word_table = pd.crosstab(trial_groups, y_word).reindex(
        columns=all_words,
        fill_value=0,
    )
    group_word_table.to_csv(output_dir / "trial_group_word_counts.csv")

    subject_group_counts = (
        pd.DataFrame({"subject": y_identity, "trial_group": trial_groups})
        .drop_duplicates()
        .groupby("subject")["trial_group"]
        .nunique()
        .sort_index()
    )
    subject_group_counts.to_csv(
        output_dir / "independent_groups_per_subject.csv",
        header=["independent_group_count"],
    )

    if require_two_groups:
        insufficient = subject_group_counts[subject_group_counts < 2]
        if not insufficient.empty:
            raise RuntimeError(
                "Held-out-group validation requires at least two independent "
                "groups per subject. These subjects do not qualify:\n"
                + "\n".join(
                    f"  {subject}: {count} group(s)"
                    for subject, count in insufficient.items()
                )
                + "\nVerify --group-mode or provide a custom run/session "
                "identifier. Do not split one acquisition into artificial "
                "groups merely to satisfy cross-validation."
            )

    missing_word_counts = (group_word_table == 0).sum(axis=1)
    incomplete_groups = missing_word_counts[missing_word_counts > 0]
    if not incomplete_groups.empty:
        detail = (
            output_dir / "trial_group_word_counts.csv"
        )
        message = (
            f"{len(incomplete_groups)} trial group(s) do not contain every "
            f"word class. See {detail}."
        )
        if require_all_words_per_group:
            raise RuntimeError(message)
        print(f"{message}")
        print(
            "   Leave-one-group-out folds may have missing test classes. "
            "Stratified grouped folds may be more appropriate when several "
            "blocks together form one balanced test partition."
        )

    print("\nIndependent group counts per subject:")
    for subject, count in subject_group_counts.items():
        print(f"  {subject}: {int(count)}")


def audit_label_schedule(
    y_identity: np.ndarray,
    y_word: np.ndarray,
    y_trial_index: np.ndarray,
    output_dir: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    global_classes = sorted(np.unique(y_word).tolist())
    n_classes = len(global_classes)

    for subject in np.unique(y_identity):
        mask = y_identity == subject
        labels = y_word[mask]
        trial_indices = y_trial_index[mask]

        transitions = int(np.sum(labels[1:] != labels[:-1]))
        transition_rate = transitions / max(len(labels) - 1, 1)

        phase_map: dict[int, str] = {}
        phase_correct = 0
        for phase in range(n_classes):
            phase_labels = labels[trial_indices % n_classes == phase]
            if len(phase_labels) == 0:
                continue
            values, counts = np.unique(phase_labels, return_counts=True)
            majority = str(values[int(np.argmax(counts))])
            phase_map[phase] = majority
            phase_correct += int(np.max(counts))

        phase_accuracy = phase_correct / len(labels) if len(labels) else 0.0
        rows.append(
            {
                "subject_id": subject,
                "trials": len(labels),
                "label_transitions": transitions,
                "label_transition_rate": transition_rate,
                "trial_index_mod_class_count_accuracy": phase_accuracy,
                "phase_majority_mapping": json.dumps(phase_map, sort_keys=True),
            }
        )

    schedule_df = pd.DataFrame(rows)
    schedule_df.to_csv(output_dir / "label_schedule_audit.csv", index=False)

    if not schedule_df.empty:
        mean_phase_accuracy = float(
            schedule_df["trial_index_mod_class_count_accuracy"].mean()
        )
        print(
            "\nLabel-order audit: mean accuracy from trial_index % "
            f"{n_classes} = {mean_phase_accuracy * 100:.2f}%"
        )
        if mean_phase_accuracy >= 0.90:
            print(
                "Word identity is highly predictable from within-recording "
                "trial order. This is an experimental-schedule confound. Keep "
                "entire runs/sessions out of training and do not use trial "
                "index as a neural feature."
            )


# Preprocressing and Validation Pipeline  

def run_preprocessing(
    set_files: list[Path],
    root_dir: Path,
    output_dir: Path,
    word_label_key: str,
    group_mode: str,
    require_two_groups: bool,
    require_all_words_per_group: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Discovered {len(set_files)} .set files.")
    print(f"Verified word-label key: {word_label_key!r}")
    print(f"Verified group mode: {group_mode!r}")
    print(f"Output directory: {output_dir}")

    # Detect byte-identical .set files before loading signals.
    set_hash_to_paths: dict[str, list[Path]] = {}
    set_file_hashes: dict[Path, str] = {}
    for set_path in set_files:
        file_hash = calculate_file_hash(set_path)
        set_file_hashes[set_path] = file_hash
        set_hash_to_paths.setdefault(file_hash, []).append(set_path)

    byte_duplicate_groups = [
        paths for paths in set_hash_to_paths.values() if len(paths) > 1
    ]
    if byte_duplicate_groups:
        details = "\n\n".join(
            "Duplicate group:\n" + "\n".join(f"  {path}" for path in paths)
            for paths in byte_duplicate_groups
        )
        raise RuntimeError(
            "Byte-identical .set files were found. Remove copies before "
            f"continuing.\n{details}"
        )

    X_list: list[np.ndarray] = []
    subject_ids: list[str] = []
    word_labels_all: list[str] = []
    recording_ids: list[str] = []
    trial_groups_all: list[str] = []
    trial_indices: list[int] = []
    trial_starts: list[int] = []
    raw_trial_hashes: list[str] = []
    set_paths_metadata: list[str] = []
    epoch_paths_metadata: list[str] = []
    set_hashes_metadata: list[str] = []
    signal_hashes_metadata: list[str] = []
    sfreq_metadata: list[float] = []

    canonical_channels = (
        None if CANONICAL_EEG_CHANNELS is None else list(CANONICAL_EEG_CHANNELS)
    )
    canonical_sfreq: float | None = None
    signal_hash_to_path: dict[str, Path] = {}
    trial_hash_to_info: dict[str, tuple[Path, int, str, str]] = {}
    overlap_rows: list[dict[str, Any]] = []

    for recording_number, set_path in enumerate(set_files, start=1):
        print(f"\n[{recording_number}/{len(set_files)}] {set_path}")

        subject_id = extract_subject_id(set_path)
        trial_group = extract_trial_group(
            set_path,
            subject_id,
            root_dir,
            group_mode,
        )
        inds_path = find_epoch_inds_file(set_path)

        inds_data = scipy.io.loadmat(
            inds_path,
            squeeze_me=False,
            struct_as_record=False,
        )
        if "thinking_inds" not in inds_data:
            raise KeyError(
                f"'thinking_inds' was not present in {inds_path}. Available "
                f"variables:\n{describe_mat_keys(inds_data)}"
            )

        raw_trial_bounds = normalize_trial_bounds(inds_data["thinking_inds"])
        num_trials = len(raw_trial_bounds)
        if num_trials == 0:
            raise ValueError(f"No trials were found in {inds_path}.")

        word_labels = extract_word_labels(
            inds_data,
            num_trials=num_trials,
            source_path=inds_path,
            word_label_key=word_label_key,
        )
        starts = np.asarray(
            [extract_start_sample(bounds) for bounds in raw_trial_bounds],
            dtype=np.int64,
        )
        if np.any(starts < 0):
            bad = np.flatnonzero(starts < 0).tolist()
            raise ValueError(
                f"Negative zero-based trial starts in {inds_path}: {bad}. "
                "Check MATLAB_INDICES_ARE_ONE_BASED."
            )

        # Fixed windows must not reuse EEG samples across nominal trials.
        order = np.argsort(starts)
        sorted_starts = starts[order]
        gaps = np.diff(sorted_starts)
        overlap_positions = np.flatnonzero(gaps < TARGET_TIME_SAMPLES)

        for position in overlap_positions:
            first_trial = int(order[position])
            second_trial = int(order[position + 1])
            overlap_rows.append(
                {
                    "set_path": str(set_path),
                    "trial_group": trial_group,
                    "trial_a": first_trial,
                    "trial_b": second_trial,
                    "start_a": int(starts[first_trial]),
                    "start_b": int(starts[second_trial]),
                    "overlap_samples": int(
                        TARGET_TIME_SAMPLES - gaps[position]
                    ),
                }
            )

        if overlap_positions.size and FAIL_ON_OVERLAPPING_TRIALS:
            overlap_path = output_dir / "overlapping_trial_windows.csv"
            pd.DataFrame(overlap_rows).to_csv(overlap_path, index=False)
            raise RuntimeError(
                f"Overlapping fixed-length windows were found in {set_path}. "
                f"See {overlap_path}. Overlapping windows are not independent."
            )

        raw = mne.io.read_raw_eeglab(
            str(set_path),
            preload=True,
            verbose=False,
        )
        sfreq = float(raw.info["sfreq"])
        eeg_matrix, selected_channels = select_and_validate_eeg_channels(
            raw,
            canonical_channels,
        )

        if canonical_channels is None:
            canonical_channels = selected_channels
        if selected_channels != canonical_channels:
            raise AssertionError("Canonical channel ordering changed.")

        if canonical_sfreq is None:
            canonical_sfreq = sfreq
        elif not np.isclose(sfreq, canonical_sfreq, rtol=0, atol=1e-9):
            raise ValueError(
                f"Sampling-rate mismatch: expected {canonical_sfreq}, found "
                f"{sfreq} in {set_path}."
            )

        signal_hash = hash_eeg_array(
            eeg_matrix,
            canonical_channels,
            sfreq,
        )
        if signal_hash in signal_hash_to_path:
            raise RuntimeError(
                "Two .set files contain equivalent canonical EEG signals:\n"
                f"  {signal_hash_to_path[signal_hash]}\n  {set_path}\n"
                "Remove duplicate exports before extracting trials."
            )
        signal_hash_to_path[signal_hash] = set_path

        recording_id = safe_name(
            f"{subject_id}__{set_path.stem}__{signal_hash[:12]}"
        )

        out_of_bounds = starts + TARGET_TIME_SAMPLES > eeg_matrix.shape[1]
        if np.any(out_of_bounds):
            bad = np.flatnonzero(out_of_bounds).tolist()
            raise ValueError(
                f"Out-of-bounds trials in {set_path}: {bad}. The script does "
                "not silently drop trials because that would shift labels."
            )

        for trial_index, start_sample in enumerate(starts.tolist()):
            stop_sample = start_sample + TARGET_TIME_SAMPLES
            trial = np.asarray(
                eeg_matrix[:, start_sample:stop_sample],
                dtype=np.float32,
                order="C",
            )
            expected_shape = (EXPECTED_EEG_CHANNELS, TARGET_TIME_SAMPLES)
            if trial.shape != expected_shape:
                raise ValueError(
                    f"Trial {trial_index} in {set_path} has shape "
                    f"{trial.shape}; expected {expected_shape}."
                )
            if not np.all(np.isfinite(trial)):
                raise ValueError(
                    f"Trial {trial_index} in {set_path} contains NaN or inf."
                )

            trial_digest = hash_trial(trial)
            if trial_digest in trial_hash_to_info:
                previous_path, previous_trial, previous_group, previous_word = (
                    trial_hash_to_info[trial_digest]
                )
                duplicate_report = pd.DataFrame(
                    [
                        {
                            "raw_trial_sha256": trial_digest,
                            "first_set_path": str(previous_path),
                            "first_trial_index": previous_trial,
                            "first_trial_group": previous_group,
                            "first_word": previous_word,
                            "duplicate_set_path": str(set_path),
                            "duplicate_trial_index": trial_index,
                            "duplicate_trial_group": trial_group,
                            "duplicate_word": str(word_labels[trial_index]),
                        }
                    ]
                )
                duplicate_path = output_dir / "exact_trial_duplicates.csv"
                duplicate_report.to_csv(duplicate_path, index=False)

                if FAIL_ON_EXACT_TRIAL_DUPLICATES:
                    raise RuntimeError(
                        "An exact raw EEG trial duplicate was found:\n"
                        f"  {previous_path}, trial {previous_trial}, "
                        f"word={previous_word}, group={previous_group}\n"
                        f"  {set_path}, trial {trial_index}, "
                        f"word={word_labels[trial_index]}, group={trial_group}\n"
                        f"See {duplicate_path}. Exact duplicates must be "
                        "removed rather than counted as separate observations."
                    )
            else:
                trial_hash_to_info[trial_digest] = (
                    set_path,
                    trial_index,
                    trial_group,
                    str(word_labels[trial_index]),
                )

            X_list.append(trial)
            subject_ids.append(subject_id)
            word_labels_all.append(str(word_labels[trial_index]))
            recording_ids.append(recording_id)
            trial_groups_all.append(trial_group)
            trial_indices.append(trial_index)
            trial_starts.append(start_sample)
            raw_trial_hashes.append(trial_digest)
            set_paths_metadata.append(str(set_path))
            epoch_paths_metadata.append(str(inds_path))
            set_hashes_metadata.append(set_file_hashes[set_path])
            signal_hashes_metadata.append(signal_hash)
            sfreq_metadata.append(sfreq)

        print(
            f"  subject={subject_id}, trial_group={trial_group}, "
            f"trials={num_trials}, sfreq={sfreq:g} Hz"
        )

        del raw
        del eeg_matrix

    if not X_list:
        raise RuntimeError("No trials were extracted.")

    X = np.stack(X_list).astype(np.float32)
    y_identity = np.asarray(subject_ids, dtype=str)
    y_word = np.asarray(word_labels_all, dtype=str)
    y_recording = np.asarray(recording_ids, dtype=str)
    trial_group_array = np.asarray(trial_groups_all, dtype=str)
    y_trial_index = np.asarray(trial_indices, dtype=np.int64)
    y_trial_start = np.asarray(trial_starts, dtype=np.int64)

    if not (
        len(X)
        == len(y_identity)
        == len(y_word)
        == len(trial_group_array)
        == len(y_recording)
    ):
        raise AssertionError("Output arrays lost one-to-one trial alignment.")

    identity_encoder = LabelEncoder()
    y_id_encoded = identity_encoder.fit_transform(y_identity)

    print("\nCompiled dataset")
    print(f"  X shape: {X.shape}")
    print(f"  participants: {len(np.unique(y_identity))}")
    print(f"  recording files: {len(np.unique(y_recording))}")
    print(f"  independent trial groups: {len(np.unique(trial_group_array))}")
    print(f"  word labels: {len(np.unique(y_word))}")

    participant_word_table = pd.crosstab(y_identity, y_word)
    participant_word_table.to_csv(output_dir / "participant_word_counts.csv")

    audit_group_design(
        y_identity=y_identity,
        y_word=y_word,
        trial_groups=trial_group_array,
        output_dir=output_dir,
        require_two_groups=require_two_groups,
        require_all_words_per_group=require_all_words_per_group,
    )
    audit_label_schedule(
        y_identity=y_identity,
        y_word=y_word,
        y_trial_index=y_trial_index,
        output_dir=output_dir,
    )

    metadata_df = pd.DataFrame(
        {
            "global_trial_index": np.arange(len(X), dtype=np.int64),
            "subject_id": y_identity,
            "identity_encoded": y_id_encoded,
            "word_label": y_word,
            "recording_id": y_recording,
            "trial_group": trial_group_array,
            "trial_index_within_recording": y_trial_index,
            "trial_start_sample_zero_based": y_trial_start,
            "trial_stop_sample_exclusive": y_trial_start + TARGET_TIME_SAMPLES,
            "raw_trial_sha256": raw_trial_hashes,
            "set_file_sha256": set_hashes_metadata,
            "loaded_signal_sha256": signal_hashes_metadata,
            "sampling_frequency_hz": sfreq_metadata,
            "word_label_key": word_label_key,
            "group_mode": group_mode,
            "set_path": set_paths_metadata,
            "epoch_inds_path": epoch_paths_metadata,
        }
    )
    metadata_df.to_csv(output_dir / "preprocessing_metadata.csv", index=False)

    arrays_to_save: dict[str, np.ndarray] = {
        # Required strict-decoder inputs.
        "X_raw.npy": X,
        "y_id_encoded.npy": y_id_encoded,
        "y_word.npy": y_word,
        "trial_group.npy": trial_group_array,
        # Additional audit/reproducibility arrays.
        "y_identity.npy": y_identity,
        "y_recording.npy": y_recording,
        "y_session_group.npy": trial_group_array,
        "y_trial_index.npy": y_trial_index,
        "y_trial_start.npy": y_trial_start,
        "subject_classes.npy": np.asarray(identity_encoder.classes_, dtype=str),
        "canonical_eeg_channels.npy": np.asarray(canonical_channels, dtype=str),
    }

    for filename, array in arrays_to_save.items():
        np.save(output_dir / filename, array)

    manifest = {
        "root_dir": str(root_dir),
        "output_dir": str(output_dir),
        "target_time_samples": TARGET_TIME_SAMPLES,
        "expected_eeg_channels": EXPECTED_EEG_CHANNELS,
        "canonical_eeg_channels": canonical_channels,
        "sampling_frequency_hz": canonical_sfreq,
        "matlab_indices_are_one_based": MATLAB_INDICES_ARE_ONE_BASED,
        "word_label_key": word_label_key,
        "group_mode": group_mode,
        "fail_on_overlapping_trials": FAIL_ON_OVERLAPPING_TRIALS,
        "fail_on_exact_trial_duplicates": FAIL_ON_EXACT_TRIAL_DUPLICATES,
        "require_at_least_two_groups_per_subject": require_two_groups,
        "require_all_words_in_every_group": require_all_words_per_group,
        "n_trials": int(len(X)),
        "n_subjects": int(len(np.unique(y_identity))),
        "n_trial_groups": int(len(np.unique(trial_group_array))),
        "n_words": int(len(np.unique(y_word))),
    }
    with (output_dir / "preprocessing_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)

    print("\nSaved grouped word-decoder inputs:")
    for filename in arrays_to_save:
        print(f"  {output_dir / filename}")
    print(f"  {output_dir / 'preprocessing_metadata.csv'}")
    print(f"  {output_dir / 'preprocessing_manifest.json'}")

    print("\nNext command:")
    print(f"  cd {json.dumps(str(output_dir))}")
    print(
        "  python3 /path/to/word_decoder_all_subjects.py\n\n"
        "The decoder must load trial_group.npy and keep each complete group "
        "inside either training or testing."
    )


# Script Orchestration and CLI Execution Flow

def main() -> None:
    args = parse_args()
    root_dir = args.root_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (root_dir / "word_decoder_inputs").resolve()
    )

    print("RUNNING FILE:", Path(__file__).resolve())
    print("DATASET ROOT:", root_dir)

    if not root_dir.exists():
        sys.exit(f"Dataset directory does not exist:\n{root_dir}")

    set_files = sorted(path.resolve() for path in root_dir.rglob("*.set"))
    if not set_files:
        sys.exit(f"No .set files were found beneath:\n{root_dir}")

    if args.inspect:
        inspect_dataset(
            set_files=set_files,
            root_dir=root_dir,
            annotation_limit=max(args.inspect_annotations, 0),
        )
        return

    if not args.word_label_key:
        sys.exit(
            "--word-label-key is required in production mode. Run with "
            "--inspect first, verify the MATLAB variable containing exactly "
            "one word label per thinking trial, then pass that key explicitly."
        )

    if not args.group_mode:
        sys.exit(
            "--group-mode is required in production mode. Run with --inspect "
            "and verify whether each .set file is an independent acquisition "
            "('set_file') or all .set files in a directory belong to one "
            "session ('parent_folder')."
        )

    run_preprocessing(
        set_files=set_files,
        root_dir=root_dir,
        output_dir=output_dir,
        word_label_key=args.word_label_key,
        group_mode=args.group_mode,
        require_two_groups=(
            REQUIRE_AT_LEAST_TWO_GROUPS_PER_SUBJECT
            and not args.allow_single_group_subjects
        ),
        require_all_words_per_group=(
            REQUIRE_ALL_WORDS_IN_EVERY_GROUP
            or args.require_all_words_per_group
        ),
    )


if __name__ == "__main__":
    main()
