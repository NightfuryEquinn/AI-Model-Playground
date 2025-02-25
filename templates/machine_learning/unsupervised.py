from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
import numpy as np
from matplotlib import pyplot
import seaborn as sns
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors import KNeighborsClassifier

# Compare using silhouette score
# Close to 1 indicate clusters well-separated
# Around 0 suggest overlapping clusters
# Close to -1 indicate potential misclassification

# Function to plot clusters
def plot_clusters(x, labels, title):
  pyplot.figure(figsize = (6, 5))
  sns.scatterplot(x = x[:, 0], y = x[:, 1], hue = labels, palette = 'viridis', legend = 'full')
  pyplot.title(title)
  pyplot.xlabel('Feature 1')
  pyplot.ylabel('Feature 2')
  pyplot.show()

# Load dataset
iris_data = datasets.load_iris()
iris_x = iris_data.data
iris_y = iris_data.target

######################################
# Principal Component Analysis (PCA) #
######################################
def pca():
  global iris_x, iris_y
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
  plot_clusters(iris_x_pca, iris_y, 'PCA on Iris Dataset')

  # Higher variance contribute more to visualization and dimensionality reduction
  print("Explained Variance Ratio:", pca.explained_variance_ratio_)
  print("Cumulative Explained Variance:", np.cumsum(pca.explained_variance_ratio_))

pca()

###########################
# Hierarchical Clustering #
###########################
def hier():
  hierarchical = AgglomerativeClustering(n_clusters = 3)
  y_hier = hierarchical.fit_predict(iris_x)
  score_hier = silhouette_score(iris_x, y_hier)

  print(f"Hierarchical Clustering Silhouette Score: {score_hier:.3f}")

  # Dendrogram
  pyplot.figure(figsize = (10, 5))
  z = linkage(iris_x, method = 'ward')
  dendrogram(z, truncate_mode = 'level', p = 3)
  pyplot.title('Hierarchical Clustering Dendrogram')
  pyplot.xlabel('Sample index (Cluster Size)')
  pyplot.ylabel('Distance')
  pyplot.show()

hier()

######################
# K-means Clustering #
######################
def kmeans_cluster():
  kmeans = KMeans(n_clusters = 3, random_state = 42, n_init = 10)
  y_kmeans = kmeans.fit_predict(iris_x)
  score_kmeans = silhouette_score(iris_x, y_kmeans)

  print(f"K-Means Clustering Silhouette Score: {score_kmeans:.3f}")

kmeans_cluster()

########
# K-NN #
########
def knn():
  knn = KNeighborsClassifier(n_neighbors = 5)
  knn.fit(iris_x, iris_y)
  y_knn = knn.predict(iris_x)
  score_knn = silhouette_score(iris_x, y_knn)

  print(f'KNN Silhouette Score: {score_knn:.3f}')

knn()

################################
# Singular-value Decomposition #
################################
def sv_decomposition():
  svd = TruncatedSVD(n_components = 2, random_state = 42)
  x_svd = svd.fit_transform(iris_x)

  plot_clusters(x_svd, iris_y, 'SVD (2 Components)')

sv_decomposition()

##################################
# Independent Component Analysis #
##################################
def ica():
  ic_analysis = FastICA(n_components = 2, random_state = 42)
  x_ica = ic_analysis.fit_transform(iris_x)

  plot_clusters(x_ica, iris_y, 'ICA (2 Components)')

ica()