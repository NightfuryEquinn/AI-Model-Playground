import pandas as pd
from sklearn import svm
from sklearn.preprocessing import LabelEncoder

# Load dataset
wine_df = pd.read_csv('wine_data.csv')
wine_df = wine_df.dropna()

# Using support vector machines (SVM)
def support_vector_machines():
  # Convert country to unique integer
  le = LabelEncoder()
  wine_df['variety'] = le.fit_transform(wine_df['variety'])

  # Features
  x = wine_df.iloc[:, 4:6]
  # Targets
  s = wine_df['variety']

  # Get unique sorted values
  unique_s = sorted(set(s))

  # Create a mapping dictionary
  # Example: { 'Pinot Noir': 0, 'Syrah': 1, ... }
  mapping_d = {}
  for index, value in enumerate(unique_s):
    mapping_d[value] = index
  
  # Convert original values in target to their numeric representations
  y = []
  for variety in s:
    numeric_value = mapping_d[variety]
    y.append(numeric_value)

  clf = svm.SVC()
  clf.fit(x, y)

  # Prediction of wine grapes variety based on points and price
  p = clf.predict([[10.0, 55.0]])
  p_variety = le.inverse_transform([p[0]])[0]

  print(f"Number of Unique Variety: { len(wine_df['variety'].unique()) }")
  print(f"Prediction Variety: { p[0] } - { p_variety }")

support_vector_machines()