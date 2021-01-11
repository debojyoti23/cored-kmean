import numpy as np
import hnswlib
from sklearn.mixture import GaussianMixture
from sklearn.cluster import kmeans_plusplus
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sys
from itertools import permutations
import warnings


# warnings.filterwarnings('error')


def getlabels(X, mu):
    # this function assigns each point to its nearest centroid
    dist = np.ones(len(X)) * sys.maxsize
    indices = np.zeros(len(X))
    for i in range(len(mu)):
        delta = X - mu[i]
        d_l2 = np.diag(np.matmul(delta, delta.T))
        indices[np.where(dist > d_l2)] = i
        dist[dist > d_l2] = d_l2[dist > d_l2]
    return indices


def plotclusters(index, X, y, ax, mu, c_means):
    ax[index].scatter(X[y == 0, 0], X[y == 0, 1], c='green', label='cluster_1')
    ax[index].scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='cluster_2')
    ax[index].scatter(X[y == 2, 0], X[y == 2, 1], c='orange', label='cluster_3')
    # ax[int(itr / 2)].scatter(c_means[:3, 0],c_means[:3, 1], c='red', marker='*', s=300, label='centroids')
    ax[index].scatter(mu[:, 0], mu[:, 1], c='red', marker='*', s=300, label='centroids')
    ax[index].legend(loc='lower right')


def updatecore(X, ann, centroids, core_size, rand_indices=[]):
    nn_indices, distances = ann.knn_query(centroids, k=core_size)
    core_set = X[nn_indices.flatten()]
    core_set = np.vstack((core_set, X[rand_indices]))
    return core_set


def compute_loss_metric(mu, mu_true):
    low = np.trace(np.matmul(mu - mu_true, (mu - mu_true).T))
    ind_perms = permutations([i for i in range(len(mu))])
    for ind in ind_perms:
        mu_cur = mu[ind, :]
        divergence = np.trace(np.matmul(mu_true - mu_cur, (mu_true - mu_cur).T))
        low = min(low, divergence)
    return low


def get_best_centroid(X, centroids_set):
    centroids_0 = centroids_set[0]
    l2_dist = np.ones(len(X)) * sys.maxsize
    for centroid in centroids_0:
        dif = X - centroid
        temp = np.diag(np.matmul(dif, dif.T))
        l2_dist = np.minimum(l2_dist, temp)
    dist = np.sum(l2_dist)
    idx = 0
    for i in range(1, len(centroids_set)):
        centroids_0 = centroids_set[i]
        l2_dist = np.ones(len(X)) * sys.maxsize
        for centroid in centroids_0:
            dif = X - centroid
            temp = np.diag(np.matmul(dif, dif.T))
            l2_dist = np.minimum(l2_dist, temp)
        dist_0 = np.sum(l2_dist)
        if dist > dist_0:
            dist = dist_0
            idx = i
    return centroids_set[idx]


def init_centroids(X, k):
    # initialize k points by kMeans++
    centroids_set = []
    for i in range(5):
        centroids, c_indices = kmeans_plusplus(X, n_clusters=k)
        # centroids = np.array([[2.12730188, 2.24326573], [-6.11226886, -5.73580923], [0.73587851, 1.51497656]])
        centroids_set.append(centroids)
    centroids = get_best_centroid(X, centroids_set)
    return centroids


def gen_coreset(X, ann, centroids, core_size, num_randomIds=0):
    nn_indices, distances = ann.knn_query(centroids, k=core_size)
    core_set = X[nn_indices.flatten()]
    rand_indices = []
    if num_randomIds > 0:
        all_indices = np.array([i for i in range(len(X))])
        unioncmp_indices = np.setdiff1d(all_indices, nn_indices.flatten())
        rand_indices = unioncmp_indices[np.unique(np.random.randint(len(unioncmp_indices), size=num_randomIds))]
        core_set = np.vstack((core_set, X[rand_indices]))

    return core_set, rand_indices


def run_EM(X, ax, core_set, mu, w, ann, core_size, rand_indices=[], maxIter=10):
    # Expectation Maximization algorithm
    k = len(mu)
    eta = np.zeros((len(core_set), k))
    for itr in range(maxIter):
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
        mu = np.array(mu)
        w = w / np.sum(w)

        # # Scatter plot
        # if itr % 2 == 0:
        #     y = getlabels(X, mu)
        #     plotclusters(int(itr / 2) + 1, X, y, ax, mu, c_means)
        #     # y = getlabels(core_set, mu)
        #     # plotclusters(int(itr / 2) + 1, core_set, y, ax, mu, c_means)

        # Update coreset
        core_set = updatecore(X, ann, mu, core_size, rand_indices)

    return mu, w


def run_EM_1(X, ax, core_set, mu, w, ann, maxIter=10):
    # Expectation Maximization algorithm
    k = len(mu)
    core_size_init = int(len(core_set) / k)
    core_size = core_size_init
    tau = 0.01  # Temperature parameter: starts low, gradually grows
    for itr in range(maxIter):
        eta = np.zeros((len(core_set), k))
        # E step
        for i in range(len(core_set)):
            den = 0
            for j in range(k):
                eta[i][j] = w[j] * np.exp(-np.matmul((core_set[i] - mu[j]), (core_set[i] - mu[j]).T) / (2.0 * tau))
                den = den + eta[i][j]
            if den > 0:
                eta[i] = eta[i] / den
            else:
                eta[i] = 0.00000000001

        # M step(modified): only the neighbourhood points take part in updating a centroid
        eta_masked = np.zeros_like(eta)
        for j in range(k):
            eta_masked[j * core_size:(j + 1) * core_size, j] = eta[j * core_size:(j + 1) * core_size, j]
        w = np.sum(eta_masked, axis=0)
        # print(w)
        mu = np.multiply(np.asmatrix(1 / w).T, np.matmul(eta_masked.T, core_set))
        mu = np.array(mu)
        w = w / np.sum(w)

        # # Scatter plot
        # if itr % 2 == 0:
        #     y = getlabels(X, mu)
        #     plotclusters(int(itr / 2) + 1, X, y, ax, mu, c_means)
        #     # y = getlabels(core_set, mu)
        #     # plotclusters(int(itr / 2) + 1, core_set, y, ax, mu, c_means)

        # Update coreset(modified): decreasing in size
        core_size = max(30, int(core_size_init * np.power(0.8, itr)))
        core_set = updatecore(X, ann, mu, core_size)

        # update temperature tau
        tau = min(1, tau * 2)

    return mu, w


def pre_process(n_cluster, c_means, c_sigma):
    # Create synthetic data points
    X = []
    for mean, sigma in zip(c_means, c_sigma):
        cov = sigma * np.identity(2)
        x = np.random.multivariate_normal(mean, cov, n_cluster)
        n_cluster *= 2
        X += list(x)

    X = np.array(X)
    np.random.shuffle(X)
    # print('Dataset shape:', X.shape)
    # print('#Clusters:', len(c_means))
    # print('Actual Centroids:', c_means)

    # Creating ANN neighbourhood
    ann = hnswlib.Index(space='l2', dim=2)
    ann.init_index(max_elements=len(X), ef_construction=50, M=16)
    X_labels = np.arange(len(X))
    ann.add_items(X, X_labels)

    return X, ann


# define the number of datapoints
n_cluster = 400

# define the number of cluster and the cluster centers
k = 3
# c_means = np.array([[0, 0], [1.5, 2], [-1, -2.5], [4.2, 5], [6.1, -3], [-5, 8.5]])
c_means = np.array([[-3, 1], [3, 2], [-6, -7]])

# define covariance for each cluster. Assume isotropic distr. i.e. identity covariance(scaled)
# c_sigma = np.array([1, 1, 1, 1, 1, 1])
c_sigma = np.array([1, 1, 1])

# # Learning Gaussian Mixture Model
# gm = GaussianMixture(n_components=k, random_state=0).fit(X)
# print('GMM params:')
# print(gm.means_)
# print(gm.weights_)
# print('GMM score:', gm.score(X))

fig, ax = plt.subplots(3, 3, figsize=(16, 16))
ax = np.ravel(ax)

MAXITER = 30
loss_rec = np.zeros([3, MAXITER])

for i in range(MAXITER):
    X, ann = pre_process(n_cluster, c_means, c_sigma)
    # Creating core-set
    centroids = init_centroids(X, k)
    core_size = 200
    core_set1, _ = gen_coreset(X, ann, centroids, core_size)
    core_set2, rand_indices = gen_coreset(X, ann, centroids, core_size=100, num_randomIds=3*30)  # with random points
    core_set3, _ = gen_coreset(X, ann, centroids, 100)

    # Initialize parameters of GMM
    mu = centroids
    cov = np.array([np.identity(len(X[0])) for i in range(k)])
    w = np.ones(k) / k

    # y = getlabels(X, mu)
    # plotclusters(0, X, y, ax, mu, c_means)
    # print('1----->',mu)

    # Perform EM
    mu1, w1 = run_EM_1(X, ax, core_set1, mu, w, ann, maxIter=10)  # graduated core-size EM
    mu2, w2 = run_EM(X, ax, core_set2, mu, w, ann, 100, rand_indices,
                     maxIter=10)  # Fixed core-size EM with random points
    mu3, w3 = run_EM(X, ax, core_set3, mu, w, ann, 100, maxIter=10)  # Fixed core-size EM

    loss_rec[0][i] = compute_loss_metric(mu1, c_means)
    loss_rec[1][i] = compute_loss_metric(mu2, c_means)
    loss_rec[2][i] = compute_loss_metric(mu3, c_means)

print(loss_rec[:, :20])
plt.figure()
plt.plot(np.linspace(1, MAXITER, MAXITER), loss_rec[0, :], c='red')
plt.plot(np.linspace(1, MAXITER, MAXITER), loss_rec[1, :], c='green')
plt.plot(np.linspace(1, MAXITER, MAXITER), loss_rec[2, :], c='yellow')
plt.xlabel('Iteration')
plt.ylabel('Loss(L2 distance between true and estimated centroids)')
# plt.ylim((0, 20))

# print('Result------------>')
# print('w', w)
# print('mu', mu)
#
# print('Loss:', compute_loss_metric(mu, c_means))

# # Perform K-mean
# km = KMeans(n_clusters=3, random_state=0).fit(X)
# y = km.labels_
# mu = km.cluster_centers_
# # print('K-Mean centroids----->', mu)
# plotclusters(7, X, y, ax, mu, c_means)

#
# plt.tight_layout()
plt.show()
