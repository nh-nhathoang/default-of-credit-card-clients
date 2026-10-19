# Probability of Default for Credit Card Accounts

This project estimates the probability that a credit card client defaults on next month's payment. It then turns that probability into a cost-based decision about which accounts to flag for review.

## Key results

- **Main model.** Logistic regression reaches a **test AUC of 0.769**, and its probabilities are well calibrated: the mean predicted PD is 22.0% against an observed default rate of 22.1%.
- **Benchmark.** Gradient boosting reaches an **AUC of 0.774**. 
- **Decision rule.** Assume a missed default costs 8 times the interest earned on a good account. Flagging every account with PD ≥ 0.11 then turns a loss of −5,941 (flagging nobody) into a profit of **+570** on the test set. Profits are in interest units: the interest earned on one repaid account.
- **Cost assumption.** The decision depends far more on the assumed cost ratio than on the model. Between ratios of 2:1 and 12:1, the share of accounts left unflagged falls from about 79% to about 20%.
- **Choice.** Logistic regression is preferred. It performs as well as gradient boosting, and its coefficients can be explained to a client.

## Problem

The target, `default payment next month`, is binary. The deliverable, however, is a **calibrated probability of default (PD)**, not a yes/no label.

Every client in the data already holds a card and has six months of repayment history. The model is therefore a behavioural score for existing accounts. Its PD supports decisions such as flagging an account for a credit-limit review, not approving new applicants.

## Data

- **Source:** Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H
- **Original paper:** Yeh, I.-C., & Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473–2480.

The data cover 30,000 clients of a Taiwanese bank, observed from April to September 2005. The target is default on the payment due in October 2005. After cleaning, the default rate is 22.1%.

| Group | Columns | Description |
|---|---|---|
| Credit limit | `LIMIT_BAL` | Credit limit, including supplementary credit for family members (NT$) |
| Demographics | `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` | Sex: 1 = male, 2 = female. Education: 1 = graduate school, 2 = university, 3 = high school, 4 = others. Marital status: 1 = married, 2 = single, 3 = others. Age in years. |
| Repayment status | `PAY_1`–`PAY_6` | Monthly status from September (`PAY_1`) back to April. Documented codes: −1 = paid duly; 1–9 = months of delay (9 = nine months or more). |
| Bill amounts | `BILL_AMT1`–`BILL_AMT6` | Monthly bill statement (NT$), September back to April |
| Payments | `PAY_AMT1`–`PAY_AMT6` | Amount paid each month (NT$), September back to April |

The September status column is called `PAY_0` in the original file. It is renamed to `PAY_1` so that its index matches `BILL_AMT1` and `PAY_AMT1`.

### Cleaning and data-quality findings
- **Missing values** There is no missing value, but some undocumented data.
- **Undocumented category codes.**
  - `EDUCATION` codes 0, 5 and 6 (345 clients, 1.2%) are merged into 4 = others, giving 468 clients.
  - `MARRIAGE` code 0 (54 clients, 0.2%) is merged into 3 = others.

- **Undocumented repayment codes −2 and 0.** Neither the documentation nor the original paper defines them. The bill and payment patterns suggest:
  - −2 means "no consumption": small or zero bills, almost always paid in full.
  - 0 means "revolving credit": large balances that are rarely paid in full, but without delay.

  Code 0 is the most common status, covering 49% of clients in September. Both codes are kept as valid categories, not treated as missing.
- **Inconsistent recording over time.**
  - Code 1 (one month late) appears for 3,667 clients in September, then 28 in August, 4 in July, 2 in June and none in May or April.
  - Two-month delays, by contrast, are common in every month.
  - Code 9 never occurs.

- **Negative bills.** 1,930 clients have a negative bill in at least one month, presumably a credit balance from overpayment.
- **Undefined target.** Neither the documentation nor the paper says what counts as a default, for example how many days past due. The paper also reports 25,000 clients, while the UCI file contains 30,000.
- **Duplicates.** After dropping `ID`, 35 rows are exact duplicates and are removed.


## Method

### Validation

- **Split.** The data are split once into stratified training (80%) and test (20%) sets with `random_state=42`. The test set is used only once, to evaluate the final models.
- **Tuning.** Hyperparameters are tuned on the training set by grid search with stratified 5-fold cross-validation. The selection criterion is **log-loss**, a proper scoring rule that rewards calibrated probabilities, which the decision rule below depends on.
- **Leakage.** All fitted preprocessing (scaling, one-hot encoding) sits inside a scikit-learn `Pipeline`, so it is refit within each fold. The log transforms are fixed functions with no fitted parameters, so applying them before the split leaks nothing.
- **Threshold.** The decision threshold is chosen on **out-of-fold training predictions** (`cross_val_predict`), never on the test set.

### Features

The six monthly bill amounts are highly correlated with each other (r = 0.80–0.95), because each bill carries over the unpaid balance from the month before. They are replaced by three features that are much less correlated (r = 0.32–0.51):

- `BILL_AMT_avg`: the mean bill over the six months.
- `BILL_AMT_trend` = `BILL_AMT1 − BILL_AMT6`: the change in the bill over the window.
- `credit utilization ratio` = `BILL_AMT1 / LIMIT_BAL`.

**Logistic regression only:**

- **Log transforms.** `log1p` is applied to `AGE`, `LIMIT_BAL` and `PAY_AMT1`–`PAY_AMT6`. A signed log, sign(x)·log(1 + |x|), is applied to the three engineered features, which can be negative.
- **Repayment status.** `PAY_1`–`PAY_6` are **one-hot encoded**, because the codes are not ordered by risk: −2, −1 and 0 are non-delay states, while 1–8 count months of delay.
  - The reference category is −2.
  - Codes seen fewer than 30 times in the training data are grouped into one category.
- **Other categoricals and scaling.** `SEX`, `EDUCATION` and `MARRIAGE` are one-hot encoded, and the continuous features are standardized.

**Gradient boosting** uses the same features, untransformed: no log, no scaling and no one-hot encoding, with category codes used as integers. Trees split on rank order, and one-hot versus integer repayment codes gave practically the same CV log-loss.

### Models

| Model | Search space | Selected | CV log-loss |
|---|---|---|---|
| Logistic regression (elastic net, `saga`) | `C` ∈ {0.001, 0.01, 0.1, 1, 10, 100}; `l1_ratio` ∈ {0, 0.25, 0.5, 0.75, 1} | `C` = 0.1, `l1_ratio` = 0 (pure L2) | 0.431 |
| Gradient boosting (`GradientBoostingClassifier`) | `learning_rate` ∈ {0.01, 0.05, 0.1}; `max_depth` ∈ {2, 3, 5}; `n_estimators` ∈ {100, 200, 400} | 0.1, depth 3, 100 trees | 0.426 |

**Class imbalance.** The data are 22.1% defaults. Class weights and resampling are not used, because they shift predicted probabilities away from the true base rate, and the cost-based threshold below relies on that base rate.

### Decision rule

The decision uses the expected-value framework of Provost & Fawcett (2013):

- An unflagged account that repays earns the interest (+1).
- An unflagged account that defaults loses L.
- A flagged account is assumed to earn and lose nothing.

Leaving an account unflagged is profitable when (1 − p) − pL > 0, that is, when **p < 1 / (1 + L)**

- **Main case.** The main analysis uses L = 8, which gives a theoretical threshold of 1/9 ≈ 0.111.
- **Empirical check.** The threshold chosen empirically on out-of-fold predictions is 0.11 for both models. This match is further evidence that the probabilities are calibrated.
- **Sensitivity.** Because L is an assumption, the analysis is repeated for L ∈ {2, 4, 8, 12}.

## Results

### Predictive performance

| Model | Train AUC | Train log-loss | CV log-loss | Test AUC | Test log-loss |
|---|---|---|---|---|---|
| Logistic regression | 0.783 | 0.428 | 0.431 | 0.769 | 0.436 |
| Gradient boosting | 0.809 | 0.410 | 0.426 | 0.774 | 0.431 |

- **Logistic regression** has almost the same log-loss on training, CV and test data.
- **Gradient boosting** fits the training data more closely, but its CV log-loss predicts its test log-loss well, so neither model overfits in a way that matters.
- **The two models rank clients almost identically** and are both well calibrated (ROC and calibration curves in GB notebook).

### Decision and profit (test set, L = 8)

| Policy | Threshold | Accounts not flagged | Profit |
|---|---|---|---|
| Logistic regression | 0.11 | 32.6% | +570 |
| Gradient boosting | 0.11 | 35.0% | +596 |
| Flag nobody | — | 100% | −5,941 |

- **Model difference.** A paired bootstrap of the profit difference (gradient boosting − logistic regression) gives +25 (95% CI −133 to 185).
- **Accuracy.** At the cost-based threshold, logistic regression's accuracy drops from 0.82 (at a 0.5 threshold) to 0.50. This is expected, because the two kinds of error no longer cost the same.

**Sensitivity to the cost ratio.** Thresholds are chosen on out-of-fold training predictions. Acceptance and profit are measured on the test set.

| L | 1/(1+L) | LR threshold | LR not flagged | LR profit | GB threshold | GB not flagged | GB profit | Flag nobody |
|---|---|---|---|---|---|---|---|---|
| 2 | 0.333 | 0.29 | 78.5% | 2,793 | 0.32 | 79.5% | 2,804 | 2,015 |
| 4 | 0.200 | 0.22 | 70.9% | 1,719 | 0.21 | 68.3% | 1,701 | −637 |
| 8 | 0.111 | 0.11 | 32.6% | 570 | 0.11 | 35.0% | 596 | −5,941 |
| 12 | 0.077 | 0.09 | 20.1% | 100 | 0.08 | 17.3% | 285 | −11,245 |

The model with the higher profit also changes with the ratio: logistic regression wins at 4:1, gradient boosting at the other three.

![Profit curves on the test set](figures/profit_curve.png)

The profit curves rank accounts from lowest to highest predicted PD. Both peak when about 38–39% of accounts are left unflagged: at 609 for logistic regression and 615 for gradient boosting. These peaks are found in hindsight on the test set and were not used to choose anything. The thresholds chosen on CV data reach 570 and 596, close to the peaks.

### What the logistic regression says

- **Recent repayment status dominates.**
  - Being two or three months late in September has the largest coefficients: +1.39 and +1.13, relative to "no consumption" (−2).
  - Revolving clients (0) score −0.85.
  - Compared with a revolving client, being two months late multiplies the odds of default by about e^(1.39 + 0.85) ≈ 9.4.
- **Utilization and payments.** Higher credit utilization raises the predicted risk (+0.28). Higher payments (−0.04 to −0.15) and a higher average bill (−0.18) lower it.
- **Credit limit.** The credit limit has only a small coefficient (−0.09), even though the default rate falls from 36% in the lowest to 12% in the highest limit decile. Most of that effect is already captured by utilization and payments.
- **Age.** Age has almost no effect (+0.02), because a single linear term cannot represent the U-shaped relation between age and default.
- **Education.** `EDUCATION = 4` has a large coefficient (−0.76), but this group has only 468 clients and includes the undocumented codes. It should not be read as an effect of education.
- **Earlier months.** The repayment variables for earlier months are strongly correlated, so the model splits their joint effect between months in an unstable way. Only their combined effect should be interpreted: recent delays strongly increase default risk.

## Limitations

- **Assumed costs.** The 8:1 ratio is an assumption. The model treats the full credit as lost on default and assumes flagging has no cost. In practice:
  - Loss given default (LGD) is below one, because part of the debt is recovered.
  - Exposure at default (EAD) varies between clients.
  - Flagging a good client loses revenue.
- **One snapshot.** The data come from one bank in 2005, during a credit card debt crisis, with a 22% default rate. That is far above typical portfolios. With only one target month, out-of-time validation is not possible.
- **Data quality.** The undocumented and inconsistently recorded codes, the negative bills and the undefined target are described above.
- **Fairness.** The models use sex, marital status and age, which are protected or sensitive attributes in credit decisions in many jurisdictions. Their effect on the decisions has not been tested.

## Possible extensions

- **Expected loss in NT$.** Express it as EL = PD × LGD × EAD, using each client's bill as the exposure. This gives client-specific thresholds instead of a single cut-off.
- **Scorecard.** Build a weight-of-evidence binned scorecard (points and reason codes). It keeps the model interpretable while capturing non-linear effects such as the U-shape in age.
- **Fairness check.** Retrain without `SEX`, `MARRIAGE` and `AGE`. Compare the performance cost, and the flag rates and default rates by group.
- **Interpretable boosting.** Try gradient boosting with monotonic constraints, explained with SHAP.

## Project structure

```
.
├── default of credit card clients.xls           # raw data (UCI)
├── cleaned default of credit card clients.csv   # cleaned data + engineered features (written by eda.ipynb)
├── eda.ipynb                                    # data cleaning, quality checks, exploratory analysis
├── logistic_regression.ipynb                    # main model; writes test_proba_lr.csv
├── gradient_boosting.ipynb                      # benchmark and model comparison; writes test_proba_gb.csv
├── figures/                                     # roc_calibration.png, profit_curve.png
├── report.pdf                                   # course report
└── README.md
```

## How to run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xlrd jupyter
```

Run the notebooks in this order:

1. `eda.ipynb` writes the cleaned CSV.
2. `logistic_regression.ipynb` writes `test_proba_lr.csv`.
3. `gradient_boosting.ipynb` writes `test_proba_gb.csv` and compares the two models.

## References

1. Yeh, I.-C., & Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473–2480.
2. Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H
3. Provost, F., & Fawcett, T. (2013). *Data Science for Business*. O'Reilly Media.
