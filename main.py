import numpy as np
import hnswlib
from sklearn.mixture import GaussianMixture
from sklearn.cluster import kmeans_plusplus

# define the number of datapoints
n_cluster = 200

# define the number of cluster and the cluster centers
k = 6
c_means = np.array([[0, 0], [1.5, 2], [-1, -2.5], [4.2, 5], [6.1, -3], [-5, 8.5]])

# define covariance for each cluster. Assume isotropic distr. i.e. identity covariance(scaled)
c_sigma = np.array([1, 1, 1, 1, 1, 1])

# Create synthetic data points
X = []
for mean, sigma in zip(c_means, c_sigma):
    cov = sigma * np.identity(2)
    x = np.random.multivariate_normal(mean, cov, n_cluster)
    n_cluster *= 2
    X += list(x)

X = np.array(X)
np.random.shuffle(X)
print('Dataset shape', X.shape)

# Learning Gaussian Mixture Model
gm = GaussianMixture(n_components=k, random_state=0).fit(X)
print('GMM params:')
print(gm.means_)
print(gm.weights_)
print('GMM score:', gm.score(X))

# Creating ANN neighbourhood
ann = hnswlib.Index(space='l2', dim=2)
ann.init_index(max_elements=len(X), ef_construction=50, M=16)
X_labels = np.arange(len(X))
ann.add_items(X, X_labels)
# nn_indices,distances=ann.knn_query(X,k=2)
# print(nn_indices)

# initialize k points by kMeans++
centroids, c_indices = kmeans_plusplus(X, n_clusters=k, random_state=0)

# print(centroids)
# Perform EM with incrementally updated coreset
core_size = 50
nn_indices, distances = ann.knn_query(centroids, k=core_size)
core_set = X[nn_indices.flatten()]
mu = centroids
cov = np.array([np.identity(len(X[0])) for i in range(k)])
eta = np.zeros((len(core_set), k))
w = np.ones(k) / k
print(np.exp(-np.dot(core_set[299] - mu[5], core_set[299] - mu[5]) / 2))

for itr in range(10):
    print('Iteration----', itr)
    # E step
    for i in range(len(core_set)):
        den = 0
        for j in range(k):
            eta[i][j] = w[j] * np.exp(-np.matmul((core_set[i] - mu[j]), (core_set[i] - mu[j]).T) / 2.0)
            den = den + eta[i][j]
        eta[i] = eta[i] / den

    # M step
    w = np.sum(eta, axis=0)
    mu = np.multiply(np.asmatrix(1 / w).T, np.matmul(eta.T, core_set))
    w = w / np.sum(w)

    print(w)
    print(mu)

# recompute core-set structure

#


# print(w)
# print(mu)
