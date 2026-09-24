# Default of Credit Card Clients — Probability of Default Model

This project aims to predict the probability that a credit card customer will default on their payment
next month.

## Problem 

This is treated as a **binary classification problem evaluated through its predicted
probabilities**. The target — `default payment next month` — is a binary outcome (0/1), but
the actual deliverable is a **calibrated probability of default (PD)**: a risk score a
lender could use to support decisions such as flagging an existing account for credit-
limit review, rather than a single approve/decline label.

## Dataset

**Source:** Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine
Learning Repository. https://doi.org/10.24432/C55S3H

**Original paper:** Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining
techniques for the predictive accuracy of probability of default of credit card
clients. *Expert Systems with Applications*, 36(2), 2473–2480.

30,000 customers from Taiwan (2005), 23 features + target. Feature groups:

- **Demographics:** credit limit, sex, education, marriage, age
- **Repayment status:** `PAY_1`–`PAY_6`, monthly repayment status for the 6 months
  preceding the prediction (renamed from the original `PAY_0`–`PAY_6` for consistent,
  sequential naming — most recent month first)
- **Bill statements:** `BILL_AMT1`–`BILL_AMT6`, monthly statement balance
- **Payment amounts:** `PAY_AMT1`–`PAY_AMT6`, monthly amount actually paid

# Notes:

The dataset's documentation is incomplete in a few places, identified during EDA:

- `EDUCATION` and `MARRIAGE` contain undocumented category codes (`0`, `5`, `6` and `0`
  respectively) beyond what the original paper defines. These are merged into each
  column's existing "others" category as a documented simplification, given the small
  affected share (~1.6% and ~0.2% of rows).
- `PAY_1`–`PAY_6` contain the documented delay codes (`-1`, `1`–`9`) plus two
  undocumented codes, `-2` and `0`. These are **not** treated as missing, following the community-established interpretation used consistently across independent analyses of this dataset (`-2` = no consumption that month, `0` = use of revolving credit), since they represent genuine, distinct payment states rather than unrecorded data.
- 35 exact duplicate rows (after dropping the `ID` column) were removed.
- The dataset does not document the precise operational definition of "default" or
  "delay" beyond the ordinal scale provided (e.g. no stated days-past-due threshold).

## Project structure

```
.
├── default of credit card clients.xls   # raw source data 
├── default credit.csv                   # cleaned dataset, exported from EDA
├── eda.ipynb                            # exploratory data analysis
├── logistic_regression.ipynb            # modeling, tuning, evaluation
└── README.md
```

## Methodology

### Feature engineering

`BILL_AMT1`–`BILL_AMT6` are severely collinear with each other (r = 0.80–0.95) despite
each having weak individual correlation with the target. These six columns are
replaced with three engineered features for modeling:

- `BILL_AMT_avg` : mean balance across the 6 months
- `BILL_AMT_trend` = `BILL_AMT1 − BILL_AMT6` : direction of change over the window
- `credit utilization ratio` = `BILL_AMT1 / LIMIT_BAL` : a snapshot measure (not an
  average), consistent with how utilization is defined in credit scores.

Right-skewed continuous features (`AGE`, `LIMIT_BAL`, `PAY_AMT1`–`6`, and the three
engineered features above) are log-transformed (`log1p`, or a signed variant for
features that can be negative) prior to scaling.

### Model

**Logistic regression**, chosen for interpretability and because its output is a
probability by construction, a reasonable fit for a project whose deliverable is a
risk score, not just a label.

- **Preprocessing:** `StandardScaler`, fit on the training split only.
- **Class imbalance (22.1% default rate):** handled *without* class weighting or
  resampling (e.g. `class_weight='balanced'`, SMOTE). Both were evaluated and rejected,
  since they distort predicted probabilities away from the dataset's true base rate —
  acceptable for a pure classifier, not acceptable when the deliverable is the
  probability itself.
- **Hyperparameter tuning:** exhaustive `GridSearchCV` over every solver/penalty
  combination `LogisticRegression` supports (L1, L2, and ElasticNet penalties;
  `lbfgs`, `liblinear`, `newton-cg`, `newton-cholesky`, `sag`, `saga` solvers),
  evaluated with 5-fold stratified cross-validation.
- **Scoring metric:** `neg_log_loss`, not accuracy or AUC. Log-loss is a proper scoring
  rule. It is only optimized by predictions that are both well-ranked and
  well-calibrated.

**Best model:** `C=0.01`, `penalty='l2'`, `solver='newton-cg'` (CV log-loss ≈ 0.452).

### Threshold selection

The classification threshold is treated explicitly as a **business decision**, not a
modeling one. Rather than defaulting to 0.5, the threshold is chosen to minimize a
cost function `(false negatives × cost) + (false positives × cost)`, using an
illustrative cost ratio reflecting that a missed default (unrecovered exposure) is
more costly than an unnecessary account flag (lost opportunity cost). This is a
documented assumption, not a figure sourced from real institutional data.

## Results

- **Test AUC:** ≈ 0.75
- **Calibration:** well-calibrated at low predicted risk, where most of the test set
  sits (thousands of customers per bin). Predictions in the mid-risk range are
  underconfident (true default rate exceeds predicted). High
  predicted risk is based on very few customers per bin (24–53).

## Limitations

- ...

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib
jupyter notebook eda.ipynb
```

Run `eda.ipynb` first to produce `default credit.csv`, then `logistic_regression.ipynb`.