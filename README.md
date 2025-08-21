# COMPAS-case-study
COMPAS is an algorithmic risk-assessment tool used in U.S. jurisdictions (including New York, Wisconsin, California, and Broward County, FL) to predict whether a defendant is likely to reoffend (recidivate). The hope was that a machine learning model could be less biased than human judges, producing more equitable outcomes.

However, evidence suggests that COMPAS predictions may not be truly unbiased, raising critical questions about the fairness of algorithmic decision-making in the criminal justice system. This repository investigates those issues using the Broward County dataset, made public by ProPublica.

<p align="center">
  <img src="images/compas1.png" alt="Image 1" width="25%">
  <img src="images/compas2.png" alt="Image 2" width="25%">
</p>

## Fairness in Machine Learning

There are many ways to define fairness in machine learning.  
Here, we focus on **Group Fairness**.

### Group Fairness (Statistical Parity)

A classifier satisfies group fairness if all demographic groups are equally likely to be predicted as positive in a binary classification problem.

$$
P(\hat{y} = 1 \mid A = a) = P(\hat{y} = 1 \mid A = b)
$$
- $\hat{y}$ = model’s prediction  
- $A$ = sensitive attribute (e.g., race)  
- $a, b$ = different groups  

### Beyond Statistical Parity
We also examine **False Positive Rates (FPR)** and **False Negative Rates (FNR)** across groups, following the principle of *equal harm/benefit*.

## COMPAS Dataset
The dataset comes from Broward County, FL public records, covering criminal defendants who underwent COMPAS assessments.

Key Variables
- 'prior_convictions': Number of prior convictions  
- 'current_charge': felony (**F**), misdemeanor (**M**), or other (**O**)
- 'charge_description': Description of the arrest charge  
- 'recidivated_last_two_years': Whether the defendant reoffended within two years (*prediction target*)  

Demographics
- Predominantly Caucasian and African-American defendants  
- *Gender imbalance*: more than **4:1 ratio** of men to women

## Results
### Model 1: Baseline
- Accuracy: ~63–64% across groups (similar)
- FPR (False Positive Rate): Higher for African-American defendants (37.2% vs. 26.8%)
- FNR (False Negative Rate): Higher for Caucasian defendants (53.7% vs. 34.3%)

<p align="center">
  <img src="images/InitialAnalysis.png" width="500px" />
</p>

### Model 2: Alternative Model (sans race)

Accuracy: 67.9% (slightly improved overall)
FPR Gap: Still large (18.2% vs. 28.4%)
FNR Gap: Caucasians still more likely to be misclassified as low-risk (64.6% vs. 35.2%)

<p align="center">
  <img src="images/RemovingRace.png" width="500px" />
</p>

**Takeaway: Even with improved accuracy, the disparity in error rates persists.**

## Conclusions
* Accuracy alone is misleading. While both groups had similar overall accuracy, error types (false positives/negatives) were unevenly distributed.
* Impact of imbalance: African-Americans were more likely to be labeled as high risk when they did not reoffend, while Caucasians were more likely to be labeled as low risk when they did reoffend.
* Fairness trade-offs: Improving one metric (e.g., lowering FPR for Caucasians) may worsen another (e.g., increasing FNR).
