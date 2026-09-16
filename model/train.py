import joblib
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# STEP 1: Load Kaggle Data
# TODO: Use pd.read_csv() to load 'creditcard.csv'.
df = pd.read_csv("../creditcard.csv")
# TODO: Separate the target variable. The Kaggle dataset uses 'Class' as the target (1 = fraud, 0 = legit).
X = df.drop(columns=['Class', 'Time']).reset_index(drop=True)
y = df['Class'].reset_index(drop=True)

# Hint: X should be all columns except 'Class'. y should be just the 'Class' column.

# (Optional Data Prep): The Kaggle dataset features (V1-V28, Amount, Time) are mostly pre-scaled.

# You can leave them as-is for this baseline, or drop 'Time' if you prefer.

# STEP 2: Train/Test Split
# TODO: Split X and y into train and test sets (80/20 ratio).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Hint: Use stratify=y to maintain the extreme class imbalance in both sets.

# STEP 3: Handle Class Imbalance & Train Model
# TODO: Calculate 'scale_pos_weight' for XGBoost (number of negative class / number of positive class).

scale_pos_weight = (df['Class']==0).sum()/(df['Class']==1).sum()
# TODO: Instantiate and fit XGBClassifier using X_train and y_train.
model = XGBClassifier(
    n_estimators=100, 
    max_depth=3,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    objective='binary:logistic'
)

model.fit(X_train, y_train)
# STEP 4: Evaluation
# TODO: Predict probabilities on X_test and calculate the PR-AUC score.
predictions = model.predict_proba(X_test)[:, 1]
predictions = pd.DataFrame(predictions)
print(predictions.head(10))
pr_auc  = average_precision_score(y_test, predictions)

print(f"pr_auc score: {pr_auc}")
# STEP 5: Export Model
# TODO: Save the model to "model.pkl" using joblib.
joblib.dump(model, 'model.pkl')
