import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn import svm, datasets
from sklearn import linear_model
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from statsmodels import api as sm
import scipy.optimize as optimization
from matplotlib import pyplot
import joblib
from sklearn.tree import DecisionTreeClassifier, plot_tree

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

###############################################
# Using Quadratic Discriminant Analysis (QDA) #
###############################################
def qda():
  # Split to train and test
  x_train, x_test, y_train, y_test = train_test_split(x, s, test_size = 0.2, random_state = 100)
  
  # Apply QDA
  qda = QDA()
  qda.fit(x_train, y_train)

  # Predict
  y_pred = qda.predict(x_test)

  # Confusion matrix
  cm = np.array(confusion_matrix(y_test, y_pred))
  confusion = pd.DataFrame(cm, index=[1, 2, 3], columns=[1, 2, 3])

  print(confusion)
  print(f"Accuracy Score:", accuracy_score(y_test, y_pred))

# qda()

#################################################
# Using Regularized Discriminant Analysis (RDA) #
#################################################
def rda():
  # Split to train and test
  x_train, x_test, y_train, y_test = train_test_split(x, s, test_size = 0.2, random_state = 100)
  
  # Apply QDA with regularization param
  qda = QDA(reg_param = 0.5)
  qda.fit(x_train, y_train)

  # Predict
  y_pred = qda.predict(x_test)

  # Confusion matrix
  cm = np.array(confusion_matrix(y_test, y_pred))
  confusion = pd.DataFrame(cm, index=[1, 2, 3], columns=[1, 2, 3])

  print(confusion)
  print(f"Accuracy Score:", accuracy_score(y_test, y_pred))
  
# rda()

#################
# Decision Tree #
#################
def tree():
  scikit_iris = datasets.load_iris()
  x, y = scikit_iris.data, scikit_iris.target

  # Split to train and test
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 100)
  
  # Initialize and train the classifier
  clf = DecisionTreeClassifier(criterion = 'gini', max_depth = 3, random_state = 100)
  clf.fit(x_train, y_train)

  pyplot.figure(figsize = (12, 8))
  plot_tree(clf, filled = True, feature_names = scikit_iris.feature_names, class_names = scikit_iris.target_names)
  pyplot.show()

  # Predict
  y_pred = clf.predict(x_test)

  # Compute accuracy
  print(f"Accuracy Score: ", accuracy_score(y_test, y_pred))
  print(classification_report(y_test, y_pred, target_names = scikit_iris.target_names))

# tree()

#################
# Random Forest #
#################
def forest():
  scikit_iris = datasets.load_iris()
  x, y = scikit_iris.data, scikit_iris.target

  # Split to train and test
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 100)

  # Initialize Random Forest
  clf = RandomForestClassifier(n_estimators = 100, random_state = 42)
  clf.fit(x_train, y_train)

  # Predict
  y_pred = clf.predict(x_test)

  # Visualizing one tree from the forest
  pyplot.figure(figsize = (20, 10))
  plot_tree(clf.estimators_[0], feature_names = scikit_iris.feature_names, class_names = scikit_iris.target_names, filled = True)
  pyplot.show()

  # Compute accuracy
  print(f"Accuracy Score: ", accuracy_score(y_test, y_pred))

# forest()

##############################
# K-Nearest Neighbours (KNN) #
##############################
def knn():
  scikit_iris = datasets.load_iris()
  x, y = scikit_iris.data, scikit_iris.target
  x_vis = x[:, :2]

  # Split to train and test
  x_train, x_test, y_train, y_test = train_test_split(x_vis, y, test_size = 0.2, random_state = 100)

  # Create and train KNN classifier
  knn = KNeighborsClassifier(n_neighbors = 5)
  knn.fit(x_train, y_train)

  # Predict
  y_pred = knn.predict(x_test)
  print("Total points: ", y_test.shape[0])
  print("Correctly labeled points: ", (y_test == y_pred).sum())

  # Create a meshgrid based on feature ranges
  x_min, x_max = x_vis[:, 0].min() - 1, x_vis[:, 0].max() + 1
  y_min, y_max = x_vis[:, 1].min() - 1, x_vis[:, 1].max() + 1
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                      np.linspace(y_min, y_max, 100))

  # Predict classes for each point in the meshgrid
  Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  # Plot decision boundaries and training points
  pyplot.figure(figsize = (10, 6))
  pyplot.contourf(xx, yy, Z, alpha = 0.3, cmap = "coolwarm")
  pyplot.scatter(x_train[:, 0], x_train[:, 1], c = y_train, cmap = "coolwarm", edgecolors = "k", label = "Train")
  pyplot.scatter(x_test[:, 0], x_test[:, 1], c = y_test, cmap = "coolwarm", marker = "s", edgecolors = "k", label = "Test")
  pyplot.xlabel(scikit_iris.feature_names[0])
  pyplot.ylabel(scikit_iris.feature_names[1])
  pyplot.title("KNN Decision Boundary (k = 5)")
  pyplot.legend()
  pyplot.show()

  # Compute accuracy
  print(f"Accuracy Score: ", accuracy_score(y_test, y_pred))

# knn()

##########################
# Classification Results #
##########################
def classification_results():
  names = ['SVM', 'Naive Bayes', 'LDA', 'QDA', 'RDA', 'Decision Tree with Gini', 'Decision Tree with Entropy', 'Random Forest', 'KNN', 'Neural Network']
  classifiers = [
    svm.SVC(), 
    GaussianNB(), 
    LDA(), 
    QDA(), 
    QDA(reg_param = 0.5), 
    DecisionTreeClassifier(criterion = 'gini', max_depth = 3, random_state = 0), 
    DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, random_state = 0),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    MLPClassifier(alpha = 1, max_iter = 1000)
  ]

  scikit_iris = datasets.load_iris()
  x, y = scikit_iris.data, scikit_iris.target
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5, random_state = 0)

  for name, clf in zip(names, classifiers):
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    score_percentage = score * 100
    print(f"{name}: {score_percentage}%")

    # Prediction
    p = clf.predict([[5.5, 1.2, 4.8, 1.4]])
    print(f"{name} - Prediction Species: {scikit_iris.target_names[p[0]]}")
    print("")

# classification_results()

##############
# Regression #
##############
def regression():
  x = [0, 1, 2, 3, 4, 5, 6, 7]
  y = [3, 5, 5, 6, 8, 9, 12, 16]

  # Linear
  x1 = sm.add_constant(x)

  model = sm.OLS(y, x1)
  results = model.fit()
  print(results.params)
  print(results.summary())

  y_pred = results.predict(x1)

  pyplot.scatter(x, y)
  pyplot.xlabel('X')
  pyplot.ylabel('Y')
  pyplot.plot(x, y_pred, "r")
  pyplot.show()

  # Polynomial
  y1 = [3, 5, 5, 6, 8, 7, 4, 2]

  mymodel = np.poly1d(np.polyfit(x, y1, 3))
  print(mymodel)

  myline = np.linspace(0, 7, 100)
  pyplot.scatter(x, y1)
  pyplot.plot(myline, mymodel(myline))
  pyplot.show()

# regression()

#########################
# Least Squares Fitting #
#########################
def lsf():
  x = np.array([0, 1, 2, 3, 4, 5])
  y = np.array([100, 90, 60, 30, 20, 1])

  def myFunc(x, a, b, c):
    return a * np.exp(-b * x) + c
  
  popt, pcov = optimization.curve_fit(myFunc, x, y)
  print('Best fit a b c: ', popt)
  print('Best fit covariance: ', pcov)

# lsf()

##############################
# Multiple Linear Regression #
##############################
def mlr():
  x = np.array([[0, 3, 5], [1, 4, 6], [2, 5, 6], [3, 6, 8], [4, 7, 9]])
  y = np.array([3, 5, 5, 6, 8])

  reg = linear_model.LinearRegression()
  reg.fit(x, y)
  print('Coefficient: \n', reg.coef_)
  print('Intercept: \n', reg.intercept_)

  pred = reg.predict([[5, 8, 10]])
  print('Prediction: \n', pred)

# mlr()

#######################
# Logistic Regression #
#######################
def log_reg():
  x = np.array([[0], [1], [2], [3], [4], [5]])
  y = np.array([1, 2, 3, 30, 32, 31])

  clf = LogisticRegression(random_state = 0).fit(x, y)
  print(clf.predict([[6]]))

  print(clf.predict_proba([[6]]))
  print(clf.score(x, y))

# log_reg()