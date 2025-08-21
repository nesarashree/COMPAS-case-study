"""
Here, we use COMPAS to explore how different models make predictions and whether they behave fairly across racial groups.
1. Training multiple predictive models (Logistic Regression, SVM, Random Forest, Neural Network)
2. Analyzing bias and fairness across groups
3. Exploring model explainability with different algorithms

This project is based on the “AI and Ethics: Criminal Justice” course by Inspirit AI (2023).
"""

# Imports
import pandas as pd
import numpy as np
import warnings
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

# Load Dataset
url = "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Projects%20-%20AI%20and%20Ethics%20-%20Criminal%20Justice/compas-scores-two-years.csv"
df_raw = pd.read_csv(url)

# The raw dataset contains many identifiers and redundant fields that are not useful for prediction. We drop them here.
drop_cols = [
    'id', 'name', 'first', 'last', 'compas_screening_date', 'dob',
    'days_b_screening_arrest', 'c_jail_in', 'c_jail_out', 'c_case_number',
    'c_offense_date', 'c_arrest_date', 'c_days_from_compas', 'r_case_number',
    'r_charge_degree', 'r_days_from_arrest', 'r_offense_date', 'r_charge_desc',
    'r_jail_in', 'r_jail_out', 'vr_case_number', 'vr_charge_degree',
    'vr_offense_date', 'decile_score.1', 'violent_recid', 'vr_charge_desc',
    'in_custody', 'out_custody', 'priors_count.1', 'start', 'end',
    'v_screening_date', 'event', 'type_of_assessment', 'v_type_of_assessment',
    'screening_date', 'score_text', 'v_score_text', 'v_decile_score',
    'decile_score', 'is_recid', 'is_violent_recid'
]
df = df_raw.drop(labels=drop_cols, axis=1)

# Rename columns for readability
df.columns = [
    'sex', 'age', 'age_category', 'race',
    'juvenile_felony_count', 'juvenile_misdemeanor_count', 'juvenile_other_count',
    'prior_convictions', 'current_charge', 'charge_description',
    'recidivated_last_two_years'
]

# Preprocessing
# Drop rare charges
value_counts = df['charge_description'].value_counts()
df = df[df['charge_description'].isin(value_counts[value_counts >= 70].index)].reset_index(drop=True)

# Convert categorical variables into one-hot encodings (e.g., "race" -> "Caucasian", "African-American", etc.)
for col in df.select_dtypes(include='object').columns:
    one_hot = pd.get_dummies(df[col])
    df = df.drop(col, axis=1).join(one_hot)

# Train-Test Split
y_column = 'recidivated_last_two_years'
X_all, y_all = df.drop(y_column, axis=1), df[y_column]
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3)

# Group splits for fairness analysis
X_caucasian = X_test[X_test['Caucasian'] == 1]
y_caucasian = y_test[X_test['Caucasian'] == 1]
X_afam = X_test[X_test['African-American'] == 1]
y_afam = y_test[X_test['African-American'] == 1]

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Logistic Regression Training accuracy:", model.score(X_train, y_train))
print("Logistic Regression Testing accuracy:", model.score(X_test, y_test))

# Fairness Metrics
def group_fairness(model, X_cauc, X_afam):
    # Statistical parity: compares positive prediction rates between Caucasian and African-American groups.
    y_cauc_pred = model.predict(X_cauc)
    y_afam_pred = model.predict(X_afam)

    p_cauc = np.count_nonzero(y_cauc_pred) / len(y_cauc_pred)
    p_afam = np.count_nonzero(y_afam_pred) / len(y_afam_pred)
    ratio = max(p_cauc, p_afam) / min(p_cauc, p_afam)

    return p_cauc, p_afam, ratio


print("\n=== Group Fairness ===")
p_cauc, p_afam, ratio = group_fairness(model, X_caucasian, X_afam)
print(f"Caucasian positive rate: {p_cauc:.3f}")
print(f"African-American positive rate: {p_afam:.3f}")
print(f"Ratio (larger/smaller): {ratio:.2f}")

# Explainability: SVM
model_svm = svm.SVC(kernel='linear')
model_svm.fit(X_train, y_train)
print("\nSVM Training accuracy:", model_svm.score(X_train, y_train))
print("SVM Testing accuracy:", model_svm.score(X_test, y_test))

# SVM coefficients give us a sense of feature importance
importance_svm = model_svm.coef_[0]
features = X_all.columns

plt.figure(figsize=(15, 8))
plt.bar(features, importance_svm)
plt.xticks(rotation="vertical")
plt.xlabel("Feature")
plt.ylabel("Coefficient Value")
plt.title("SVM Feature Importances")
plt.show()

# Explainability: Random Forest
model_rf = RandomForestClassifier(max_depth=5)
model_rf.fit(X_train, y_train)
print("\nRandom Forest Training accuracy:", model_rf.score(X_train, y_train))
print("Random Forest Testing accuracy:", model_rf.score(X_test, y_test))

# Random forest uses "impurity reduction" to rank features. More important features contribute more to reducing uncertainty.
rf_importances = model_rf.feature_importances_
plt.figure(figsize=(15, 8))
plt.bar(features, rf_importances)
plt.xticks(rotation="vertical")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importances")
plt.show()

# Focus on race features specifically
race_features = features[10:16]
race_importances = rf_importances[10:16]
plt.figure(figsize=(10, 6))
plt.bar(race_features, race_importances)
plt.xlabel("Race Feature")
plt.ylabel("Importance")
plt.title("Race Feature Importances (Random Forest)")
plt.show()

# Neural Network
model_nn = MLPClassifier(hidden_layer_sizes=(10, 10, 10),
                         random_state=1, max_iter=500)
model_nn.fit(X_train, y_train)

print("\nNeural Network Training accuracy:", model_nn.score(X_train, y_train))
print("Neural Network Testing accuracy:", model_nn.score(X_test, y_test))
# "Interpretability-accuracy trade-off" of NNs

print("\nAnalysis complete. Models trained and fairness metrics evaluated.")
