import h2o
from h2o.automl import H2OAutoML
from h2o.frame import H2OFrame
import pandas as pd

# Load dataset
df = pd.read_csv("data/wine_customer_segmentation.csv")

# Assume the last column is the target variable
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Initialize H2O
h2o.init()

# Convert to H2O frame
hf = H2OFrame(df)

# Identify the target column dynamically
target_column = df.columns[-1]  # Last column as target
hf[target_column] = hf[target_column].asfactor()  # Convert to categorical (for classification)

# Set feature columns
feature_columns = hf.columns[:-1]  # All except target

# Run AutoML
aml = H2OAutoML(max_models = 10, seed = 42)  # Limit to 10 models for faster execution
aml.train(x = feature_columns, y = target_column, training_frame = hf)

# View the leaderboard
print(aml.leaderboard)

# Make predictions
preds = aml.leader.predict(hf)
print(preds.head(20))

# Evaluate model performance
perf = aml.leader.model_performance(hf)
print(perf)

# Check class distribution
print(df[target_column].value_counts(normalize=True))

# Shutdown H2O after use
h2o.shutdown(prompt = False)