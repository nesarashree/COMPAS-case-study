"""
This script explores bias and fairness in machine learning using the COMPAS dataset.
1. Data cleaning & exploratory analysis
2. Baseline predictive models
3. Fairness analysis across race
4. Reflections on fairness interventions

Dataset: Broward County, FL COMPAS records (ProPublica)
"""

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

warnings.filterwarnings("ignore")

# Load Data
!wget -q '[YOUR URL HERE].csv'
data = pd.read_csv("compas-scores-two-years.csv", header=0)

# Drop redundant/irrelevant columns
cols_to_drop = [
    'id','name','first','last','compas_screening_date','dob','days_b_screening_arrest',
    'c_jail_in','c_jail_out','c_case_number','c_offense_date','c_arrest_date','c_days_from_compas',
    'r_case_number','r_charge_degree','r_days_from_arrest','r_offense_date','r_charge_desc',
    'r_jail_in','r_jail_out','vr_case_number','vr_charge_degree','vr_offense_date','decile_score.1',
    'violent_recid','vr_charge_desc','in_custody','out_custody','priors_count.1','start','end',
    'v_screening_date','event','type_of_assessment','v_type_of_assessment','screening_date',
    'score_text','v_score_text','v_decile_score','decile_score','is_recid','is_violent_recid'
]

df = data.drop(cols_to_drop, axis=1)
df.columns = [
    'sex','age','age_category','race','juvenile_felony_count','juvenile_misdemeanor_count',
    'juvenile_other_count','prior_convictions','current_charge','charge_description',
    'recidivated_last_two_years']

# Exploratory Analysis
print("Dataset size:", len(df))
print("Most common charges:", df['charge_description'].value_counts().index[:3].tolist())
print("<25 yrs w/ ≥2 priors:", np.sum((df['age'] < 25) & (df['prior_convictions'] > 1)))

'''
General observations
 - Dataset skews young (20s–30s), mostly male, with African-American and Caucasian overrepresentation.
 - Features like priors and age are likely predictive; race is sensitive.
'''

# Filter rare charges and one-hot encode categoricals
value_counts = df['charge_description'].value_counts()
df = df[df['charge_description'].isin(value_counts[value_counts >= 70].index)].reset_index(drop=True)

for col in df.select_dtypes(include='object').columns:
    df = df.join(pd.get_dummies(df[col])).drop(col, axis=1)

# Baseline Model: Logistic Regression
y_column = 'recidivated_last_two_years'
X_all, y_all = df.drop(y_column, axis=1), df[y_column]
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)

print("Logistic Regression – Train Acc:", log_model.score(X_train, y_train))
print("Logistic Regression – Test Acc:", log_model.score(X_test, y_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Alternate Model: Random Forest
rf_model = RandomForestClassifier(max_depth=5)
rf_model.fit(X_train, y_train)
print("Random Forest – Test Acc:", rf_model.score(X_test, y_test))

# Fairness Analysis
caucasian = X_test[X_test['Caucasian'] == 1]
african_american = X_test[X_test['African-American'] == 1]

print("Group sizes – Caucasian:", len(caucasian), "AA:", len(african_american))

print("Caucasian Acc:", log_model.score(caucasian, y_test[caucasian.index]))
print("AA Acc:", log_model.score(african_american, y_test[african_american.index]))

plot_confusion_matrix(log_model, caucasian, y_test[caucasian.index], cmap=plt.cm.Blues, values_format='d')
plot_confusion_matrix(log_model, african_american, y_test[african_american.index], cmap=plt.cm.Blues, values_format='d')
plt.show()

''' 
Observed: Higher false positive rate (FPR) for African-American group, violating EEOC fairness threshold.
'''

# Removing Race as a Feature
def remove_race(df):
    return df.drop(["African-American","Asian","Caucasian","Hispanic","Native American","Other"], axis=1)

X_train_nr, X_test_nr = remove_race(X_train), remove_race(X_test)
model_nr = LogisticRegression(max_iter=1000)
model_nr.fit(X_train_nr, y_train)

print("No-Race Model – Test Acc:", model_nr.score(X_test_nr, y_test))

plot_confusion_matrix(model_nr, remove_race(caucasian), y_test[caucasian.index], cmap=plt.cm.Blues, values_format='d')
plot_confusion_matrix(model_nr, remove_race(african_american), y_test[african_american.index], cmap=plt.cm.Blues, values_format='d')
plt.show()

''' 
Removing race did not resolve disparities; fairness issues persist, likely via proxy variables (priors, zip codes, etc.)
'''

'''
REFLECTIONS:
 - Logistic regression: baseline ~0.65–0.70 accuracy, biased FPR against AA defendants.
 - Random forest: slightly better accuracy, but fairness concerns remain.
 - Removing race worsens fairness (proxy bias remains).

Key takeaway: Fairness cannot be achieved by simply omitting sensitive variables. Structural biases in data leak into features, 
requiring deeper interventions (fairness-aware training, societal policy changes, etc.).
'''
