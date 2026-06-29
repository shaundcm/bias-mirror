# BiasMirror — Algorithmic Fairness Auditing Tool

A Python tool that trains multiple classifiers on the UCI Adult Income dataset and audits each one for demographic bias across racial groups, producing a ranked fairness report with statistical significance testing.

---

## What It Does

BiasMirror runs a full fairness audit pipeline:

1. Loads and preprocesses the Adult Income dataset (auto-detects target column, handles missing values, encodes categoricals)
2. Trains 6 classifiers on the income prediction task
3. Computes per-group fairness metrics for each model using one-vs-rest decomposition
4. Runs bootstrap confidence intervals and permutation significance tests on bias scores
5. Runs a multiclass meta-probe to measure demographic signal leakage across model outputs
6. Exports a master CSV and ranked fairness summary

---

## Models

| Model | Config |
|---|---|
| Logistic Regression | `max_iter=1000` |
| Decision Tree | `max_depth=6` |
| Random Forest | `n_estimators=200` |
| K-Nearest Neighbors | `k=7` |
| Naive Bayes | Gaussian |
| MLP | `hidden_layers=(50,), max_iter=500` |

---

## Fairness Metrics

For each model × demographic group (one-vs-rest):

| Metric | Description |
|---|---|
| **SPD** (Statistical Parity Difference) | P(ŷ=1 \| group) − P(ŷ=1 \| rest) |
| **DI** (Disparate Impact) | P(ŷ=1 \| group) / P(ŷ=1 \| rest) |
| **TPR gap** | True positive rate per group |
| **FPR gap** | False positive rate per group |
| **PPV gap** | Precision per group |
| **MI leakage** | Mutual information between model scores and group membership |
| **Bootstrap CI** | 95% confidence interval on SPD (500 resamples) |
| **Permutation p-value** | Significance of observed SPD (500 permutations) |

Models are ranked by mean |SPD| across groups — lower = fairer.

### Meta-Probe

A multiclass logistic regression is trained on the stacked probability outputs of all 6 models to predict group membership, evaluated with 5-fold cross-validated balanced accuracy. This measures how much demographic signal leaks through model outputs even when the sensitive attribute is excluded from training.

---

## Getting Started

```bash
pip install scikit-learn statsmodels seaborn
```

**In Colab (recommended):**

1. Paste the notebook cell and run
2. Upload `adult_income.csv` when prompted
3. Outputs are written to `/content/biasmirror_outputs/`

**Locally:**

Place `adult_income.csv` in the working directory and run:

```bash
python biasmirror.py
```

To audit by sex instead of race, change one line in the config:

```python
SENSITIVE_ATTR = 'sex'   # default is 'race'
```

---

## Output Files

```
biasmirror_outputs/
├── biasmirror_master_table.csv    # per-model per-group fairness metrics
├── model_probs_test.csv           # stacked model probability outputs
└── meta_probe_summary.txt         # meta-probe CV scores and interpretation
```

---

## Dataset

UCI Adult Income (`adult_income.csv`) — predicts whether income exceeds $50K/year based on census features. Available at [archive.ics.uci.edu](https://archive.ics.uci.edu/ml/datasets/adult).

---

## Authors

Deishaun Colins Martin (23PT05) · Devanand K (23PT06)  
PSG College of Technology
