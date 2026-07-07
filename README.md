# EEG Subject Identification Using Imagined Speech

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Dataset](https://img.shields.io/badge/Dataset-KARA%20ONE-green)
![Signal](https://img.shields.io/badge/Signal-EEG-purple)
![Framework](https://img.shields.io/badge/Framework-scikit--learn-orange)
![Use](https://img.shields.io/badge/Use-Research%20Only-red)

A Python project for identifying enrolled subjects from imagined-speech EEG recordings.

The project includes:

- A Jupyter notebook for inspecting and validating the downloaded dataset
- A preprocessing pipeline for converting EEGLAB recordings into validated NumPy arrays
- Spectral and amplitude feature extraction
- Random Forest subject classification
- Cross-validation reports, prediction summaries, confusion matrices, and plots

## Project Structure

```text
.
├── inspect_dataset.ipynb
├── preprocessing_pipeline.py
├── subject_identifier.py
├── requirements.txt
├── subject_identification_results/
└── README.md
```

## Requirements

- Python 3.10 or newer
- Jupyter Notebook or JupyterLab for running `inspect_dataset.ipynb` (optional)
- The **Kara One (KARA ONE) dataset**
- EEG recordings in EEGLAB `.set` format
- A corresponding `epoch_inds.mat` file containing trial boundaries and word labels

Using a virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/j-thie/Neurotechnology_EEG_Subject_Identification_Using_Imagined_Speech.git
cd Neurotechnology_EEG_Subject_Identification_Using_Imagined_Speech
```

### 2. Download the Kara One dataset

This project requires the **Kara One imagined-speech EEG dataset**. Download the participant archives from the official University of Toronto dataset page:

[Download the Kara One dataset](https://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html)

Extract the downloaded participant archives into a dataset directory, for example:

```text
kara_one_dataset/
├── MM05/
├── MM08/
├── MM09/
└── ...
```

The complete dataset is approximately 24 GB, so make sure you have enough storage space. Review the dataset's academic-use and citation requirements before using it.

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

## Usage

### 1. Inspect the dataset with the notebook (optional)

The `inspect_dataset.ipynb` notebook can be used to explore the KARA ONE files before preprocessing. It:

- Scans the dataset for MATLAB files
- Reports EEG shapes, channel counts, sampling rates, and recording durations
- Inspects `epoch_data.mat` structures and imagined-speech trials
- Reads EEGLAB `.set` metadata
- Displays available event annotations and trigger mappings

Before running the notebook, replace the hardcoded example paths with the location of the KARA ONE dataset on your computer:

```python
root = Path("/path/to/kara_one_dataset")
```

Also update the example EEGLAB file path used in the trigger-inspection cell:

```python
file_path = Path(
    "/path/to/kara_one_dataset/MM05/Acquisition 232 Data.set"
)
```

Open the notebook in VS Code, Jupyter Notebook, or JupyterLab. For example:

```bash
jupyter notebook inspect_dataset.ipynb
```

The notebook is intended for dataset exploration and validation. It does not replace the preprocessing pipeline.

### 2. Inspect the dataset with the preprocessing script

Run the preprocessing script in inspection mode before creating the model inputs:

```bash
python preprocessing_pipeline.py \
  --root-dir /path/to/dataset \
  --inspect
```

This displays the discovered recordings, MATLAB variables, candidate grouping methods, channel information, and annotations.

### 3. Preprocess the EEG data

After identifying the correct word-label variable and acquisition grouping, run:

```bash
python preprocessing_pipeline.py \
  --root-dir /path/to/dataset \
  --output-dir ./word_decoder_inputs \
  --word-label-key VERIFIED_KEY \
  --group-mode set_file
```

Available grouping modes:

- `set_file`: treats every `.set` file as an independent recording
- `parent_folder`: treats recordings in the same parent folder as one acquisition group

The preprocessing pipeline produces files such as:

```text
X_raw.npy
y_id_encoded.npy
y_identity.npy
y_word.npy
trial_group.npy
subject_classes.npy
preprocessing_metadata.csv
preprocessing_manifest.json
```

### 4. Configure the subject identifier

Before running `subject_identifier.py`, review these constants near the top of the file:

```python
BASE_OUTPUT_DIR = Path("/path/to/subject_identification_results")
SFREQ = 1000.0
N_SPLITS = 5
```

Set `BASE_OUTPUT_DIR` to a valid location and confirm that `SFREQ` matches the EEG sampling frequency recorded in `preprocessing_manifest.json`.

### 5. Run subject identification

The classifier loads its NumPy input files from the current working directory. Run it from the preprocessing output directory:

```bash
cd word_decoder_inputs
python ../subject_identifier.py
```

## Model

The subject-identification script:

1. Extracts EEG band-power features for delta, theta, alpha, beta, and gamma bands
2. Calculates channel-level and global amplitude statistics
3. Trains a Random Forest classifier
4. Performs five-fold evaluation with hyperparameter tuning
5. Saves the trained model, metrics, predictions, and visualizations

## Outputs

The classification stage can generate:

```text
subject_identification_summary.csv
subject_identification_predictions.csv
subject_identification_report.txt
subject_identification_confusion.csv
subject_identification_breakdown.png
fold_performance_distribution.png
subject_identifier_model.joblib
subject_feature_metadata.npz
```

## Important Notes

- The preprocessing pipeline expects 62 EEG channels and extracts 4,000 samples per trial by default.
- Replace the hardcoded local paths in `inspect_dataset.ipynb` before running it.
- Dataset paths and subject-folder naming must match the assumptions in `preprocessing_pipeline.py`.
- Always inspect and verify the word-label key before preprocessing.
- Use genuine recording, run, or session groups to reduce data leakage.
- The classifier performs closed-set identification, meaning test subjects must already be represented in the enrolled subject set.

## License

This repository is licensed under the MIT License.

## Citation
Zhao, S., & Rudzicz, F. (2015). *Classifying phonological categories in imagined and articulated speech*. In **Proceedings of ICASSP 2015** (pp. 992–996). https://doi.org/10.1109/ICASSP.2015.7178118
