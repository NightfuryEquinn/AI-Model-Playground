import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/wine_customer_segmentation.csv")

# Assume the last column is the target variable
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Adjust class labels to start from 0
y_train_adjusted = y_train - y_train.min()
y_test_adjusted = y_test - y_test.min()

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators = 100, random_state = 42)
rf_model.fit(X_train, y_train)

# Train XGBoost
xgb_model = XGBClassifier(n_estimators = 100, objective = "multi:softmax", num_class = len(y.unique()), eval_metric = "logloss")
xgb_model.fit(X_train, y_train_adjusted)

# Predictions
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test) + y_train.min()

# Evaluate performance
rf_acc = accuracy_score(y_test, rf_pred)
xgb_acc = accuracy_score(y_test, xgb_pred)

print(f"Random Forest Accuracy: {rf_acc:.4f}")
print(f"XGBoost Accuracy: {xgb_acc:.4f}")

# Stacking
# Define base models
base_models = [
  ('rf', RandomForestClassifier(n_estimators = 100, random_state = 42)),
  ('xgb', XGBClassifier(n_estimators = 100, objective = "multi:softmax", num_class = len(y.unique()), eval_metric = "logloss"))
]

# Define meta-model (Logistic Regression)
meta_model = LogisticRegression()

# Define stacking classifier
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)

# Train the stacking model
stacking_model.fit(X_train, y_train)

# Predictions
stack_pred = stacking_model.predict(X_test)

# Evaluate performance
stack_acc = accuracy_score(y_test, stack_pred)
print(f"Stacking Ensemble Accuracy: {stack_acc:.4f}")