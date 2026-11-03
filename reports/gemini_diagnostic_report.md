# DataDrift-Shield Diagnostic Report

## 1. Statistically Significant Feature Drift ($\alpha = 0.05$)
At the defined significance level of $\alpha = 0.05$:
* **`MonthlyCharges`** shows statistically significant drift ($p\text{-value} = 1.16 \times 10^{-57} < 0.05$, `drift_detected`: `true`).
* **`tenure`** does **not** show statistically significant drift ($p\text{-value} = 0.772003 > 0.05$, `drift_detected`: `false`).
* **`TotalCharges`** does **not** show statistically significant drift ($p\text{-value} = 0.104172 > 0.05$, `drift_detected`: `false`).

---

## 2. Explanation of KS Statistics and p-Values
* **KS Statistic (Kolmogorov-Smirnov Statistic):** Measures the maximum distance between the cumulative probability distributions of the reference data and the incoming batch data.
  * `MonthlyCharges` has a KS statistic of `0.21026`, indicating a measurable shift between distributions.
  * `tenure` (`0.01711`) and `TotalCharges` (`0.03149`) show much smaller distributional differences.
* **p-Value:** Represents the probability of observing a KS statistic at least as extreme as the measured value under the assumption that both datasets come from the identical distribution.
  * A $p\text{-value}$ below $0.05$ indicates that the difference is statistically significant. The $p\text{-value}$ for `MonthlyCharges` ($1.16 \times 10^{-57}$) is far below $0.05$, confirming distribution drift.
  * The $p\text{-values}$ for `tenure` ($0.772$) and `TotalCharges` ($0.104172$) remain above $0.05$, indicating no statistically significant drift for these features.

---

## 3. Insights from Top SHAP Features
SHAP values measure feature contributions to model predictions (i.e., how strongly each feature influences the output), rather than causal effects.

The top feature contributions to model predictions are:
1. `cat__Contract_Month-to-month`: $0.70382$
2. `num__tenure`: $0.43976$
3. `num__MonthlyCharges`: $0.28620$
4. `cat__OnlineSecurity_No`: $0.23146$
5. `cat__InternetService_Fiber optic`: $0.19818$
6. `cat__TechSupport_No`: $0.18616$
7. `num__TotalCharges`: $0.17984$
8. `cat__PaymentMethod_Electronic check`: $0.16523$
9. `cat__Contract_Two year`: $0.15259$
10. `cat__PaperlessBilling_No`: $0.11243$

`num__MonthlyCharges` ranks as the third most influential feature on predictions among the evaluated features, with a contribution weight of $0.28620$.

---

## 4. Findings & Experimental Limitations

### What Can Be Concluded:
* A 20% increase in `MonthlyCharges` in the incoming synthetic batch resulted in detectable, statistically significant distribution drift for `MonthlyCharges`.
* `tenure` and `TotalCharges` did not exhibit statistically significant distribution drift in this experiment.
* `MonthlyCharges` is an active contributor to the model's prediction output (SHAP value: $0.28620$).

### What Cannot Be Concluded:
* **Model Performance Degradation:** Distribution drift is distinct from model performance degradation. Because no model performance measurements were provided in this experiment, **it cannot be claimed that model performance has declined.**
* **Causal Relationships:** SHAP feature importance reflects model prediction contributions only; it does not prove real-world cause-and-effect relationships.
* **Production Behavior:** This run was conducted on a static dataset with artificially simulated drift (a 20% increase to `MonthlyCharges`), not on live production data.

---

## 5. Pre-Retraining Assessment Protocol
Because distribution drift alone does not prove performance degradation, and this script does not make retraining decisions, the following must be verified before considering model retraining:

1. **Measure Actual Model Performance:** Acquire and evaluate model performance metrics (e.g., accuracy, error rates) on the new batch, as no performance measurements were included in this dataset.
2. **Evaluate Real Production Data:** Verify whether drift occurs in actual live production environments rather than in a static, synthetically altered dataset.
3. **Assess Impact on Output:** Analyze how the drift in `MonthlyCharges` impacts actual downstream predictions, given its SHAP importance contribution ($0.28620$).

*Note: DataDrift-Shield does not autonomously approve retraining decisions.*