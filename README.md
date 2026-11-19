# Social Network Mining: Spam-Account Detection

An end-to-end analysis of a labelled social-network dataset combining
user-profile classification, exploratory graph analysis, and network
visualisation.

## Pipeline

`fakhraei_minipipeline.py`:

1. Loads the compressed user table and the first 100,000 edges of the
   relation graph.
2. Builds numeric features from the user-profile table. Graph structure is
   used for exploratory analysis only and is not fed into the classifiers.
3. Uses stratified 60/20/20 train/validation/test splits.
4. Compares class-balanced logistic regression and random forest models.
5. Tunes the random-forest decision threshold on the validation split.
6. Produces ROC/PR curves, confusion matrices, feature importances, degree
   distributions, a graph summary, and a t-SNE embedding of user features.

## Results

The archived run uses 50,000 sampled users, four numeric profile columns, and
a spam prevalence of about 6% (598 positives in the 10,000-user test set).

| Model | ROC-AUC | PR-AUC | Spam precision | Spam recall | Spam F1 |
|---|---:|---:|---:|---:|---:|
| Logistic regression (balanced) | 0.761 | 0.258 | 0.106 | 0.756 | 0.186 |
| Random forest (balanced) | 0.708 | 0.187 | 0.315 | 0.194 | 0.240 |
| Random forest, tuned threshold (0.60) | 0.708 | 0.187 | 0.361 | 0.177 | 0.238 |

Logistic regression ranks accounts better (higher ROC-AUC and PR-AUC) and
catches most spammers, at the cost of many false positives. The random forest
is more precise but misses most spammers, and threshold tuning trades a little
recall for precision without improving F1. With only four profile columns,
neither model separates the classes well; adding graph-derived features is the
obvious next step.

In the graph sample, the 1,000 highest-degree nodes share no edges with each
other, so the subgraph figure shows isolated hubs rather than community
structure.

## Structure

```text
.
├── fakhraei_minipipeline.py
├── data/fakhraei/           # expected local dataset; not committed
├── output/fakhraei/         # archived figures and JSON metrics
└── docs/                    # final report source and PDF
```

## Setup and run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python fakhraei_minipipeline.py
```

Place the course dataset under `data/fakhraei/`. Generated artifacts are
written to `output/fakhraei/`.

## Report

The complete methodology and discussion are available in
[`docs/final_report.pdf`](docs/final_report.pdf).
