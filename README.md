# Network Intrusion Detection on UNSW-NB15

Integrated project, Machine Learning course (Prof. Youness Moukafih), UIR 2026/2027, Subject 13.

Team: Walid El Haimer, Ayman Ezzougari, Ahmed Ilyas Elmoussi, Ahmed Graoui.

The system reads the statistics of a network flow (duration, bytes, packets, TTL, protocol, …) and
answers two questions:

1. **Binary:** is the flow normal traffic or an attack? (column `label`)
2. **Multiclass:** which of the 10 categories is it? (column `attack_cat`: Normal and 9 attack types,
   including the rare Analysis, Backdoor, Shellcode and Worms)

As the project guidelines ask, the focus is on understanding the data and justifying the preprocessing,
using simple models from the course (logistic regression, k-nearest neighbours, decision tree, random forest).
The written report is in [`report/report.pdf`](report/report.pdf) and the slides in
[`report/slides.pdf`](report/slides.pdf).

## Installation

Python 3.14 (any recent Python 3 with the versions below should work).

```
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

The dataset (two CSV files, 47 MB) is downloaded automatically into `data/raw/` the first time a script
runs. Source: the official UNSW-NB15 partitions (https://research.unsw.edu.au/projects/unsw-nb15-dataset),
mirrored at https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15.
In this copy the two file names are swapped (the "testing-set" file has the 175,341 training rows);
`configs/config.yaml` takes care of it.

## Usage

Run every command from this folder.

| Command | What it does | Time |
|---|---|---|
| `python explore.py` | Data exploration: class balance, duplicates, skew, TTL. Saves plots to `figures/` | ~10 s |
| `python preprocessing_study.py` | Compares preprocessing choices (scaling, duplicates, TTL features) | ~3-4 min |
| `python main.py` | **Training entry point**: loads and preprocesses the data, searches the hyperparameters of every model for both tasks, chooses the alarm threshold, saves `models/best_model.pkl` | ~3 min |
| `python test.py` | **Independent test**: loads the saved model (no retraining) and evaluates it on the official test file, analyses false alarms and rare classes, and shows predictions for a few flows | ~15 s |
| `python test.py --input flows.csv` | **Inference** on your own flows (same columns as the dataset); writes `predictions.csv` | seconds |

### Web interface

A one-page Django website lets anyone try the saved model without writing code: fill in the 42
features of a flow (the three text features are drop-down menus) and the page shows the attack
probability, whether an alarm is raised, and the predicted attack category. A menu loads real test
flows of each category as starting points.

```
cd webapp
python manage.py runserver      # then open http://127.0.0.1:8000
python manage.py test           # 6 tests of the page
```

It needs `models/best_model.pkl` (included) and uses the same `src/` code as `test.py`.

`main.py` and `test.py` are independent: `test.py` only needs `models/best_model.pkl`, which is included
in the repository. Every script accepts `--config path/to/config.yaml`. All settings (paths, seed,
validation size, preprocessing options, hyperparameter search spaces, alarm threshold rule) are in
[`configs/config.yaml`](configs/config.yaml); the fixed seed (42) makes every run give the same numbers.
Result tables are written as CSV files to `results/`.

## Method in short

* **Data:** official train file (175,341 flows) and test file (82,332 flows). 38.6% of the train file are
  exact duplicates and are removed, then 20% is kept aside for validation (stratified by category).
  The test file is only used by `test.py`.
* **Preprocessing:** numeric features → `log(1+x)` then standardization; `proto`, `service`, `state` →
  one-hot encoding, with rare values grouped and unseen values (state ACC/CLO in test) handled.
  All statistics are learned on the training part only (inside a scikit-learn `Pipeline`).
* **Two-stage system:** stage 1 decides normal vs attack; flows that raise an alarm go to stage 2, which
  names the attack category (it is trained on attack flows only, so the two answers never contradict).
* **Models and search space:** logistic regression, decision tree and random forest, see `search` in the
  config. The binary model is selected by its detection rate when at most 5% of normal validation flows
  raise an alarm; the category model by macro F1, so that rare classes count as much as common ones.
* **Results (test file):** random forests win both stages. Detection rate 0.964 with a false alarm rate of
  0.168 (the threshold allowed 5% on validation: the report explains this distribution shift);
  macro F1 over the 9 attack categories 0.526.

## Repository structure

| File or path | Purpose |
|---|---|
| `README.md` | Installation and usage instructions |
| `requirements.txt` | Python dependencies |
| `configs/config.yaml` | Paths, preprocessing options, hyperparameter search spaces |
| `explore.py` | Data exploration script |
| `preprocessing_study.py` | Experiments comparing preprocessing choices |
| `main.py` | Training entry point |
| `test.py` | Independent test and inference with the trained model |
| `src/data.py` | Data loading, duplicate removal, train/validation split |
| `src/preprocessing.py` | Preprocessing pipeline (scaling, one-hot encoding) |
| `src/model.py` | Model definitions and prediction on new flows |
| `src/train.py` | Hyperparameter search and alarm threshold choice |
| `src/evaluate.py` | Metrics, false alarm and per-category analysis, plots |
| `src/explore.py` | Exploration tables and plots |
| `src/utils.py` | Configuration, saving/loading, small helpers |
| `src/theme.py` | Dark green colour theme shared by every plot |
| `webapp/` | Django web interface: `detector/predictor.py` (link to the model), `forms.py`, `views.py`, template, stylesheet, tests |
| `models/best_model.pkl` | Saved models (binary + multiclass) and alarm threshold |
| `results/` | Result tables (CSV) produced by the scripts |
| `figures/` | Plots produced by the scripts |
| `report/` | Written report and slides (LaTeX sources and PDFs) |

## Reference

N. Moustafa and J. Slay, "UNSW-NB15: a comprehensive data set for network intrusion detection systems
(UNSW-NB15 network data set)", Military Communications and Information Systems Conference (MilCIS), 2015.
