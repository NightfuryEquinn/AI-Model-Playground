import pandas as pd
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot
import seaborn as sea
from pandas.plotting import scatter_matrix

# Load datasets
iris = pd.read_csv('iris_data.csv')

# Features
x = iris.iloc[:, :4]
# Targets
s = iris['species']

print(iris.describe())

#######################################
# Using support vector machines (SVM) #
#######################################
def support_vector_machines():
  # Convert to unique integer
  le = LabelEncoder()
  iris['species'] = le.fit_transform(iris['species'])

  # Get unique sorted values
  unique_s = sorted(set(s))

  # Create a mapping dictionary
  mapping_d = {}
  for index, value in enumerate(unique_s):
    mapping_d[value] = index
  
  # Convert original values in target to their numeric representations
  y = []
  for variety in s:
    numeric_value = mapping_d[variety]
    y.append(numeric_value)

  clf = svm.SVC()
  clf.fit(x.values, y)

  # Prediction
  p = clf.predict([[5.5, 1.2, 4.8, 1.4]])
  p_variety = le.inverse_transform([p[0]])[0]

  print(f"Number of Unique Species: {len(iris['species'].unique())}")
  print(f"Prediction Species: {p[0]} - {p_variety}")

  # Display scatterplot
  def scatterplot():
    scatter_matrix(iris)
    pyplot.show()

    iris.hist()
    pyplot.show()

  scatterplot()

  # Determine accuracy score
  def determine_score():
    # Split data to train and test
    x_train, x_test, y_train, y_test = train_test_split(x, s, test_size = 0.2, random_state = 30)

    # Linear Kernel
    clf = svm.SVC(kernel = 'linear')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    percentage_score = score * 100

    print(f"Linear Accuracy Score: {percentage_score}%")

    # RBF Kernel
    clf = svm.SVC(kernel = 'rbf')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    percentage_score = score * 100

    print(f"RBF Accuracy Score: {percentage_score}%")

    # Poly Kernel
    clf = svm.SVC(kernel = 'poly')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    percentage_score = score * 100

    print(f"Poly Accuracy Score: {percentage_score}%")

    # Sigmoid Kernel
    clf = svm.SVC(kernel = 'sigmoid')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    percentage_score = score * 100

    print(f"Sigmoid Accuracy Score: {percentage_score}%")

  determine_score()

support_vector_machines()

#####################
# Using Naive Bayes #
#####################
