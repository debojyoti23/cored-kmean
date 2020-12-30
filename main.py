import numpy as np
import hnswlib
from sklearn.mixture import GaussianMixture
from sklearn.cluster import kmeans_plusplus

# define the number of datapoints
n_cluster=50


# define the number of cluster and the cluster centers
k=6
c_means=np.array([[0,0],[1.5,2],[-1,-2.5],[4.2,5],[6.1,-3],[-5,8.5]])

# define covariance for each cluster. Assume isotropic distr. i.e. identity covariance(scaled)
c_sigma=np.array([1,1,1,1,1,1])


def init(X,k):
	#initialization: choosing k starting points by kMeans++
	centroids=[]
	centroids.append(X[np.random.randint(len(X)),:])

	return centroids

# Create synthetic data points
X=[]
for mean,sigma in zip(c_means,c_sigma):
	cov=sigma*np.identity(2)
	x=np.random.multivariate_normal(mean,cov,n_cluster)
	n_cluster*=2
	X+=list(x)

X=np.array(X)
np.random.shuffle(X)
print('Dataset shape',X.shape)

# Learning Gaussian Mixture Model
gm=GaussianMixture(n_components=k, random_state=0).fit(X)
print('GMM params:')
print(gm.means_)
print(gm.weights_)
print('GMM score:',gm.score(X))

# Creating ANN neighbourhood
p=hnswlib.Index(space='l2',dim=2)
p.init_index(max_elements=len(X),ef_construction=50,M=16)
X_labels=np.arange(len(X))
p.add_items(X,X_labels)
labels,distances=p.knn_query(X,k=2)
# print(labels)

# initialize k points by kMeans++
# print(init(X,k))
centroids,indices=kmeans_plusplus(X,n_clusters=k,random_state=0)




