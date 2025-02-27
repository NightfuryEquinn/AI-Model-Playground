import numpy as np
import gym
from gym import spaces
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Q-Learning RL
# Load datasets
data = pd.read_csv('data/wine_customer_segmentation.csv')

features = data.drop('Customer_Segment', axis = 1)
labels = data['Customer_Segment']

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split data
x_train, x_test, y_train, y_test = train_test_split(features_scaled, labels, test_size = 0.5, random_state = 50)

# Environment
class WineEnv(gym.Env):
  def __init__(self, x, y):
    super(WineEnv, self).__init__()
    self.x = x
    self.y = y
    self.n_samples = x.shape[0]
    self.n_features = x.shape[1]

    # Determine the number of classes dynamically
    self.n_classes = len(np.unique(y))
    self.current_index = 0

    # Define observation and action spaces
    self.observation_space = spaces.Box(low = np.min(x), high = np.max(x), shape = (self.n_features,), dtype = np.float32)
    self.action_space = spaces.Discrete(self.n_classes)

  def reset(self):
    self.current_index = 0
    return self.x[self.current_index]
  
  def step(self, action):
    correct_label = self.y.iloc[self.current_index] if hasattr(self.y, 'iloc') else self.y[self.current_index]

    # Reward is 1 for correct, else -1
    reward = 1 if action == correct_label else -1

    self.current_index += 1
    done = self.current_index >= self.n_samples

    if not done:
      next_state = self.x[self.current_index]
    else:
      next_state = None

    return next_state, reward, done, {}

# Instantiate environment with training data
env = WineEnv(x_train, y_train)

def training_model():
  # Initialize Q-table as dictionary
  q_table = defaultdict(lambda: np.zeros(env.action_space.n))

  alpha = 0.1 # Learning rate
  gamma = 0.9 # Discount factor
  epsilon = 0.1 # Exploration rate
  n_episodes = 1000 # Number of episodes

  for episode in range(n_episodes):
    state = env.reset()
    done = False
    
    while not done:
      state_tuple = tuple(state) # Using tuple of features for key

      # Epsilon-greedy policy
      if np.random.rand() < epsilon:
        action = np.random.choice(env.action_space.n)
      else:
        action = np.argmax(q_table[state_tuple])
      
      next_state, reward, done, _ = env.step(action)
      next_state_tuple = tuple(next_state) if next_state is not None else None

      # Q-Learning update
      if not done:
        best_next_action = np.argmax(q_table[next_state_tuple])
        q_table[state_tuple][action] += alpha * (reward + gamma * q_table[next_state_tuple][best_next_action] - q_table[state_tuple][action])
      else:
        q_table[state_tuple][action] += alpha * (reward - q_table[state_tuple][action])

    if (episode + 1) % 50 == 0:
      print(f"Episode {episode + 1} completed.")

  print("Training Completed.")

  # Save Q-table to a file
  with open('models/q_table.pkl', 'wb') as f:
    pickle.dump(dict(q_table), f)

  # Save scaler to a file
  with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# training_model()

# Load the Q-table from file
with open('models/q_table.pkl', 'rb') as f:
  loaded_q_table = pickle.load(f)

# Load the scaler
with open('models/scaler.pkl', 'rb') as f:
  loaded_scaler = pickle.load(f)

def predict_user_input(q_table, scaler):
  columns = ['Alcohol', 'Malic_Acid', 'Ash', 'Ash_Alcanity', 'Magnesium',
           'Total_Phenols', 'Flavanoids', 'Nonflavanoid_Phenols', 'Proanthocyanins',
           'Color_Intensity', 'Hue', 'OD280', 'Proline']
  
  # Prompt user
  print("Enter 13 feature values")
  user_input = input()

  try:
    user_features = [float(x) for x in user_input.split(',')]
    user_df = pd.DataFrame([user_features], columns = columns)

    if len(user_features) != 13:
      print("Please enter exactly 13 numeric values.")
      return
  except ValueError:
    print("Invalid input! Please ensure you enter numeric values.")
    return

  # Scale the user input
  user_features_scaled = scaler.transform(user_df)

  # Convert into tuple for Q-table lookup
  state_tuple = tuple(user_features_scaled[0])

  # Predict using Q-table
  if state_tuple in q_table:
    action = np.argmax(q_table[state_tuple])
  else:
    action = np.random.choice(env.action_space.n)
  
  print(f"Predicted Customer Segment: {action}")

predict_user_input(loaded_q_table, loaded_scaler)