from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor, VotingRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC

# Iris classification
# Load datasets
def iris_classification():
  iris = datasets.load_iris()
  x, y = iris.data[:, 1:3], iris.target

  clf1 = LogisticRegression(random_state = 1)
  clf2 = RandomForestClassifier(n_estimators = 50, random_state = 1)
  clf3 = GaussianNB()
  clf4 = SVC()

  eclf = VotingClassifier(
    estimators = [('lr', clf1), ('rf', clf2), ('gnb', clf3), ('svc', clf4)],
    voting = 'hard'
  )

  classifiers = [clf1, clf2, clf3, clf4, eclf]
  labels = ['Logistic Regression', 'Random Forest', 'Naïve Bayes', 'SVM', 'Ensemble']

  for clf, label in zip(classifiers, labels):
    scores = cross_val_score(clf, x, y, scoring = 'accuracy', cv = 5)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

iris_classification()

# Diabetes regression
# Load datasets
def diabetes_regression():
  x, y = datasets.load_diabetes(return_X_y = True)

  # Train classifiers
  reg1 = GradientBoostingRegressor(random_state = 1)
  reg2 = RandomForestRegressor(random_state = 1)
  reg3 = LinearRegression()
  reg4 = MLPRegressor()

  reg1.fit(x, y)
  reg2.fit(x, y)
  reg3.fit(x, y)
  reg4.fit(x, y)

  ereg = VotingRegressor(estimators = [('gb', reg1), ('rf', reg2), ('lr', reg3), ('NN', reg4)])
  print(ereg.fit(x, y))

  # Making predictions
  xt = x[:20]

  pred1 = reg1.predict(xt)
  pred2 = reg2.predict(xt)
  pred3 = reg3.predict(xt)
  pred4 = reg4.predict(xt)
  pred5 = ereg.predict(xt)

  plt.figure()
  plt.plot(pred1, 'gd', label = 'Gradient Boosting Regressor')
  plt.plot(pred2, 'b^', label = 'Random Forest Regressor')
  plt.plot(pred3, 'ys', label = 'Linear Regressor')
  plt.plot(pred4, 'mo', label = 'Neural Network')
  plt.plot(pred5, 'r*', ms = 10, label = 'Voting Regressor')

  plt.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False)
  plt.ylabel('Predicted')
  plt.xlabel('Training Samples')
  plt.legend()
  plt.title("Regressor predictions and Averages")

  plt.show()

diabetes_regression()