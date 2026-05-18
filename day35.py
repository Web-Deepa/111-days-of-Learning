#K-Means Clustering
import matplotlib.pyplot as plt
import  numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

#1.generate data
x,y_true=make_blobs(n_samples=300,centers=4,cluster_std=0.60,random_state=42)
kmeans=KMeans(n_clusters=4,init="k-means++",random_state=42) #2.define and fit data


y_kmeans=kmeans.fit_predict(x) #predict cluster assignment for each data 

centroids=kmeans.cluster_centers_ #calculate centers
print(f"Calcualted Centroid \n:{np.round(centroids, 3)} ")

#plot
plt.figure(figsize=(8,6))
plt.scatter(x[:,0],x[:,1],c=y_kmeans,s=50,cmap='viridis',alpha=0.7)
plt.scatter(centroids[:,0],centroids[:,1],c='red',s=200,marker="X",label="Centroids")
plt.title("K-Means Clustering Result")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True,linestyle='--',alpha=0.5)
plt.savefig("k-means.png")
plt.show()

