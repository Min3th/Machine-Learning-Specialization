import numpy as np
import matplotlib.pyplot as plt
from utils import *

def KMeans_init_centroids(X,K):
  randidx = np.random.permutation(X.shape[0])

  centroids = X[randidx[:K]]

  return centroids

def find_closest_centroids(X, centroids):
  K = centroids.shape[0]
  idx = np.zeros(X.shape[0],dtype=int)

  for i in range(X.shape[0]):
    dist = []
    for j in range(K):
      norm_ij = np.linalg.norm(X[i]-centroids[j])
      dist.append(norm_ij)
    
    idx[i] = np.argmin(dist)

  return idx

def compute_centroids(X,idx,K):
  m,n = X.shape
  centroids = np.zeros((K,n))

  for i in range(K):
    indexes = np.where(idx==i)[0]
    if len(indexes)> 0 :
      centroid_values = np.mean(X[indexes],axis=0)
      centroids[i] = centroid_values

  return centroids

def run_KMeans(X,intial_centroids,max_iters = 10,plot_progress = False):
  m,n = X.shape
  K = intial_centroids.shape[0]
  centroids = intial_centroids
  previous_centroids = centroids
  idx = np.zeros(m)
  plt.figure(figsize=(8,6))

  for i in range(max_iters):
    print("K-Means iteration %d/%d" % (i,max_iters-1))
    idx = find_closest_centroids(X,centroids)

    if plot_progress:
      plot_progress_kMeans(X,centroids,previous_centroids,idx,K,i)
      previous_centroids= centroids

    centroids = compute_centroids(X,idx,K)
    


  
  plt.show()
  return centroids,idx
