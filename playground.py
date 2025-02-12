import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Sample dataset for playground
data = {
  "Wine Variety": ["Cabernet Sauvignon", "Chardonnay", "Merlot", "Riesling", "Syrah"],
  "Sweetness": [1, 3, 2, 5, 2],
  "Acidity": [3, 4, 3, 5, 3],
  "Body": [4, 3, 4, 2, 5],
  "Tannins": [5, 1, 3, 1, 4],
  "Alcohol %": [14.5, 13.5, 13.0, 11.0, 14.0]
}

# Convert dataset to DataFrame
df = pd.DataFrame(data)

# Encode wine variety labels
label_encoder = LabelEncoder()
df['Wine Variety'] = label_encoder.fit_transform(df['Wine Variety'])

# Features and labels
X = df.drop('Wine Variety', axis = 1)
y = df["Wine Variety"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators = 100, random_state = 42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100}%")

# Function to recommend wine
def recommend_wine(sweetness, acidity, body, tannins, alcohol):
  user_input = np.array([[sweetness, acidity, body, tannins, alcohol]])
  predicted_class = model.predict(user_input)[0]
  wine_name = label_encoder.inverse_transform([predicted_class])[0]

  return wine_name

recommend_wine = recommend_wine(sweetness = 5, acidity = 7, body = 3, tannins = 2, alcohol = 12.5)
print(f'Recommended Wine: {recommend_wine}')