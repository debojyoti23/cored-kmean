import numpy as np
from sklearn.mixture import GaussianMixture

# define the number of datapoints
n_cluster=500


# define the number of cluster and the cluster centers
k=6
c_means=np.array([[0,0],[1.5,2],[-1,-2.5],[4.2,5],[6.1,-3],[-5,8.5]])

# define covariance for each cluster. Assume isotropic distr. i.e. identity covariance(scaled)
c_sigma=np.array([1,1,1,1,1,1])

# Create synthetic data points
X=[]
for mean,sigma in zip(c_means,c_sigma):
	cov=sigma*np.identity(2)
	x=np.random.multivariate_normal(mean,cov,n_cluster)
	X+=list(x)

X=np.array(X)
np.random.shuffle(X)
print('Dataset shape',X.shape)


gm=GaussianMixture(n_components=k, random_state=0).fit(X)
print(gm.means_)


