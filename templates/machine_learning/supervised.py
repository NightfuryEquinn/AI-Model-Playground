import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn import svm, datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
import joblib

# Load datasets
iris = pd.read_csv('data/iris_data.csv')
wine = pd.read_csv('data/wine_customer_segmentation.csv')

# Features
x = iris.iloc[:, :4]
wine_x = wine.iloc[:, 0:13].values
# Targets
s = iris['species']
wine_y = wine.iloc[:, 13].values

# print(iris.describe())

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
    scatter_matrix(iris.drop('species', axis = 1))
    pyplot.show()

    iris.drop('species', axis = 1).hist()
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

  # Confusion Matrix and Classification Report
  def confusion():
    cancer = datasets.load_breast_cancer()
    x = cancer.data
    y = cancer.target

    # Split to train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 100)

    # SVM
    clf = svm.SVC()
    clf.fit(x_train, y_train)

    # Prediction
    y_predict = clf.predict(x_test)

    # Precision is accuracy of positive predictions
    # Precision = TP / (TP + FP)
    # Recall is fraction of positives that were correctly identified
    # Recall = TP / (TP + FN)
    # F1 Score is weighted harmonic mean of precision and recall, between 0.0 - 1.0
    # F1 = 2 * (Recall * Precision) / (Recall + Precision)
    cm = np.array(confusion_matrix(y_test, y_predict, labels = [0, 1]))
    confusion = pd.DataFrame(cm, index = ['Healthy', 'Cancer'], columns = ['Predicted Healthy', 'Predicted Cancer'])
    
    print(confusion)
    print(classification_report(y_test, y_predict, target_names=['Healthy', 'Cancer']))

  confusion()

# support_vector_machines()

#####################
# Using Naive Bayes #
#####################
def naive_bayes():
  scikit_iris = datasets.load_iris()
  x, y = scikit_iris.data, scikit_iris.target

  save_clf = GaussianNB()
  save_clf.fit(x, y)

  # Save model
  joblib.dump(save_clf, 'models/model.pkl')
  clf = joblib.load('models/model.pkl')

  # Prediction
  p = clf.predict([[5.5, 1.2, 4.8, 1.4]])
  print(f"Prediction Species: {scikit_iris.target_names[p[0]]}")

# naive_bayes()

############################################
# Using Linear Discriminant Analysis (LDA) #
############################################
def lda():
  # Split to train and test
  x_train, x_test, y_train, y_test = train_test_split(wine_x, wine_y, test_size = 0.2, random_state = 100)
  
  # Feature scaling
  sc = StandardScaler()
  x_train = sc.fit_transform(x_train)
  x_test = sc.transform(x_test)

  # Apply LDA
  lda = LDA(n_components = 2)
  x_train = lda.fit_transform(x_train, y_train)
  x_test = lda.transform(x_test)

  # Fitting Logistic Regression to Training set
  classifier = LogisticRegression(random_state = 0)
  classifier.fit(x_train, y_train)

  # Predict Test set results
  y_pred = classifier.predict(x_test)

  # Confusion matrix
  cm = np.array(confusion_matrix(y_test, y_pred))
  confusion = pd.DataFrame(cm, index=[1, 2, 3], columns=[1, 2, 3])

  # Plotting
  pyplot.figure(figsize = (10, 6))
  colors = ['red', 'green', 'blue']
  markers = ['o', 's', 'x']
  for i, label in enumerate(np.unique(y_train)):
    pyplot.scatter(x_train[y_train == label, 0], x_train[y_train == label, 1], c = colors[i], label = f'Class {label}', marker = markers[i])
  pyplot.title('LDA: Training Set Projection')
  pyplot.xlabel('LDA Component 1')
  pyplot.ylabel('LDA Component 2')
  pyplot.legend(title = 'Customer Segment')
  pyplot.show()

  print(confusion)
  print(f"Accuracy Score:", accuracy_score(y_test, y_pred))

# lda()

##########################
# Classification Results #
##########################
def classification_results():
  names = ['SVM', 'Naive Bayes', 'LDA']
  classifiers = [svm.SVC(), GaussianNB(), LDA()]

  scikit_iris = datasets.load_iris()
  x, y = scikit_iris.data, scikit_iris.target
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 100)

  for name, clf in zip(names, classifiers):
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    score_percentage = score * 100
    print(f"{name}: {score_percentage}%")

    # Prediction
    p = clf.predict([[5.5, 1.2, 4.8, 1.4]])
    print(f"{name} - Prediction Species: {scikit_iris.target_names[p[0]]}")
    print("")

classification_results()