#Hierarchical Clustering and DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage

# 1.generate data
x, _ = make_moons(n_samples=100, noise=0.1, random_state=42)
x = StandardScaler().fit_transform(x)

# 2. Fit Models
hier = AgglomerativeClustering(n_clusters=2, linkage='ward') # Hierarchical Clustering 
hier_labels = hier.fit_predict(x)

dbscan = DBSCAN(eps=0.25, min_samples=5)  # DBSCAN 
dbscan_labels = dbscan.fit_predict(x)

z= linkage(x[:30], method='ward') # Create Dendrogram


fig, axes = plt.subplots(1, 3, figsize=(18, 5)) #plot

#  Dendrogram plot
dendrogram(z, ax=axes[0])
axes[0].set_title("Hierarchical Dendrogram  of 30 points")
axes[0].set_xlabel("Data Point Index")
axes[0].set_ylabel("Distance Threshold")

# Hierarchical Clustering Results plot
axes[1].scatter(x[:, 0], x[:, 1], c=hier_labels, cmap='viridis', s=40)
axes[1].set_title("Hierarchical Clustering  with k=2")

#  DBSCAN Results plot (black=noise)
axes[2].scatter(x[:, 0], x[:, 1], c=dbscan_labels, cmap='plasma', s=40)
axes[2].set_title("DBSCAN Clustering")

plt.savefig("hierdb.png")
plt.show()

