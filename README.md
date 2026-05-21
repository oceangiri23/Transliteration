# Nepali Transliteration & Normalization Toolkit

This directory contains tools, datasets, notebooks and pipelines for transliterating, normalizing and preparing Nepali text (Romanized and Devanagari). The components here were developed to support dataset cleaning, phonetic encoding, clustering, and producing augmented parallel corpora for model training and evaluation.

**Goals**
- Provide a reproducible phonetic/transliteration encoder for Nepali.
- Normalize noisy Romanized text and align it with Devanagari forms.
- Produce cleaned, augmented parallel datasets for downstream NLP tasks.

**Where to look**
- Core encoder: [Transliteration/nepPhoneticEncoder.py](Transliteration/nepPhoneticEncoder.py)
- App helpers / small demo utilities: [Transliteration/app_hf.py](Transliteration/app_hf.py)
- Pipeline components: [Transliteration/pipeline_bundle/two_sentence_pipeline.py](Transliteration/pipeline_bundle/two_sentence_pipeline.py)
- Phonetic dictionaries & merged corpora: [Transliteration/ADS/Complete_phonetic_Dictionary.json](Transliteration/ADS/Complete_phonetic_Dictionary.json)
- Final cleaned datasets: [Transliteration/Final%20Datasets/augmented_parallel.csv](Transliteration/Final%20Datasets/augmented_parallel.csv)

Directory layout (high level)
- `nepPhoneticEncoder.py` — phonetic encoding and normalization logic.
- `app_hf.py` — helper functions for small apps or demos.
- `ADS/` — auxiliary datasets, merged corpora and supporting files.
- `Dataset Preparation/` — notebooks used for data collection and initial cleaning.
- `Final Datasets/` — produced parallel corpora and augmented CSVs ready for training.
- `Normalization/` — clustering experiments and normalization artifacts.
- `pipeline_bundle/` — reusable pipeline modules used across preprocessing.

Prerequisites
- Python 3.8 or newer.
- Typical packages used across scripts and notebooks: `pandas`, `numpy`, `regex`. Some notebooks may require `fasttext`, `sentencepiece`, or `transformers` depending on the experiment.

Setup (recommended)
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
# install packages as needed, for example:
pip install pandas numpy regex
```

Quick start
- Run the phonetic encoder on an input file (example):
```powershell
python Transliteration/nepPhoneticEncoder.py --input path/to/input.txt --output path/to/output.json
```
- Run a streaming/two-sentence pipeline component (example):
```powershell
python Transliteration/pipeline_bundle/two_sentence_pipeline.py --source src.txt --out processed.txt
```
Refer to the header comments or `--help` in each script for accurate argument names and examples.

Datasets & reproducibility
- The `Final Datasets/` directory contains the cleaned and augmented datasets used for training models. Treat these as generated artifacts — regenerate them using the notebooks and scripts in `Dataset Preparation/` when changes are needed.
- Intermediate merges, phonetic dictionaries and clustering outputs live in `ADS/` and `Normalization/`.

Notebooks
- Several Jupyter notebooks are included for experimentation (e.g., `fasttext.ipynb`, `pipeline.ipynb`). Use these to reproduce clustering, dictionary creation, and augmentation steps.

Best practices
- Work in a virtual environment to keep dependencies isolated.
- Re-run notebooks step-by-step when reproducing or updating datasets; avoid modifying produced files in `Final Datasets/` without also saving the regeneration steps.

Contributing
- Add data sources with a short description and licensing information.
- Include a notebook or script that documents preprocessing steps used to generate any dataset you add or modify.

License & attribution
- Check repository-level license for redistribution and reuse rules. Confirm third-party dataset licenses before external sharing.

Contact
- This project is maintained inside the workspace. For questions about a specific script or dataset, open the corresponding notebook/script and raise an issue or reach out to the maintainers listed in project notes.

Model checkpoints & metrics
- Model training artifacts and evaluation metrics (mT5, byT5) are referenced in the workspace; see relevant experiment notebooks for full training logs and links to model checkpoints.
