from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

iris = load_iris()
x = iris.data
print(x)
y = iris.target
print(y)

model = AgglomerativeClustering(n_clusters = 3)
labels = model.fit_predict(x)

fig, axes = plt.subplots(1,2, figsize = (10,8))

# print(x[:,0])

axes[0].scatter(x[:,0], x[:,1], c = y, cmap = "rainbow")
axes[0].set_xlabel(iris.feature_names[0])
axes[0].set_ylabel(iris.feature_names[1])



axes[1].scatter(x[:,0], x[:,1],c = labels, cmap = "rainbow")
axes[1].set_xlabel(iris.feature_names[0])
axes[1].set_ylabel(iris.feature_names[1])

plt.show()

plt.figure(figsize = (10,8))
z = linkage(x,"ward")
dendrogram(z)
plt.title("dendrogram-Hierarchical clustering")
plt.xlabel("sample data")
plt.ylabel("Distance")
plt.grid(True)
plt.show()