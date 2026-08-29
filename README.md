# Social Network Mining: Spam-Account Detection

An end-to-end analysis of a labelled social-network dataset combining graph
structure, user-profile features, supervised classification, anomaly detection,
and network visualisation.

## Pipeline

`fakhraei_minipipeline.py`:

1. Loads the compressed user table and graph data.
2. Builds numeric profile and graph features.
3. Uses stratified train/validation/test splits.
4. Compares class-balanced logistic regression and random forest models.
5. Tunes the random-forest decision threshold on the validation split.
6. Produces ROC/PR curves, confusion matrices, feature importances, degree
   distributions, graph summaries, and user embeddings.

The archived run contains 50,000 sampled users and four numeric features. The
random forest reached 0.708 ROC-AUC; class imbalance makes PR-AUC and minority
recall more informative than raw accuracy.

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
