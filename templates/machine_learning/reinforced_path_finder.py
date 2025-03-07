import numpy as np
import pylab as plt
from reinforced_utils import *

# Setting params
# Creating routing list and set goal
points_list = [
    (0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 7), (3, 8), 
    (4, 9), (5, 10), (6, 11), (7, 12), (8, 13), (9, 14), (10, 14), 
    (11, 14), (12, 14), (13, 14)
]
goal = 14

# Show routing graph
showgraph(points_list)

# Number of points of the R matrix
MATRIX_SIZE = 15

# Create matrix R
R = createRmat(MATRIX_SIZE, points_list, goal)

# Create matrix Q
Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))

# Learning param
gamma = 0.9

# Training
scores = []

for i in range(2000):
  # Select a random current_state (Starting point)
  current_state = np.random.randint(0, int(Q.shape[0]))

  # Work out all the available next step actions
  available_act = available_actions(R, current_state)

  # Choose a random next step action
  action = sample_next_action(available_act)

  #Update the Q matrix
  score = update(R, Q, current_state, action, gamma)
  scores.append(score)

  print('Score: ', str(score))

print("Trained Q Matrix: ")
print(Q / np.max(Q) * 100)

# Testing
current_state = 0
steps = [current_state]

while current_state != goal:
  next_step_index = np.where(Q[current_state, ] == np.max(Q[current_state, ]))[1]

  if next_step_index.shape[0] > 1:
    next_step_index = int(np.random.choice(next_step_index, size = 1))
  else:
    next_step_index = int(next_step_index)
  
  steps.append(next_step_index)
  current_state = next_step_index

# Display results
print("Most efficient path: ")
print(steps)

plt.plot(scores)
plt.xlabel("Iterations")
plt.ylabel("Score")
plt.title("Q-learning Training Progress")
plt.show()