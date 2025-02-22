from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot

######################################
# Principal Component Analysis (PCA) #
######################################
def pca():
  iris_data = datasets.load_iris()
  iris_x = iris_data.data
  iris_y = iris_data.target

  pyplot.scatter(iris_x[:, 0], iris_x[:, 1], c = iris_y)
  pyplot.xlabel('Sepal Length')
  pyplot.ylabel('Sepal Width')
  pyplot.title('Original Dataset')
  pyplot.colorbar()
  pyplot.show()

  pyplot.scatter(iris_x[:, 2], iris_x[:, 3], c = iris_y)
  pyplot.xlabel('Petal Length')
  pyplot.ylabel('Petal Width')
  pyplot.title('Original Dataset')
  pyplot.colorbar()
  pyplot.show()

  # Reduce to 2 components
  pca = PCA(n_components = 4)
  pca.fit(iris_x)
  iris_x_pca = pca.transform(iris_x)
  iris_y = np.choose(iris_y, [1, 2, 3, 0]).astype(float)

  # Plot PCA
  pyplot.scatter(iris_x_pca[:, 0], iris_x_pca[:, 1], c = iris_y)
  pyplot.xlabel('Principal Component 1')
  pyplot.ylabel('Principal Component 2')
  pyplot.title('PCA on Iris Dataset')
  pyplot.colorbar()
  pyplot.show()

  # Higher variance contribute more to visualization and dimensionality reduction
  print("Explained Variance Ratio:", pca.explained_variance_ratio_)
  print("Cumulative Explained Variance:", np.cumsum(pca.explained_variance_ratio_))

# pca()

###########################
# Hierarchical Clustering #
###########################


######################
# K-means Clustering #
######################


########
# K-NN #
########


################################
# Singular-value Decomposition #
################################


##################################
# Independent Component Analysis #
##################################

