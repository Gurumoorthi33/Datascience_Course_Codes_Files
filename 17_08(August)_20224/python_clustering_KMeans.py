# from sklearn.cluster import KMeans
# from sklearn import datasets
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# iris = datasets.load_iris()
# df = pd.DataFrame(iris['data'], columns = iris["feature_names"])
# print(df.head())

from sklearn.cluster import KMeans
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", None)

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)
print(df.head())

iris_target = iris.target
print(iris_target)

iris_target_names = iris.target_names
print(iris_target_names)

model = KMeans(n_clusters = 3, random_state = 42)
model.fit(df)
predicted_value = model.predict(df)

df["target"] = iris.target

df["Kmeans"] = predicted_value
print(df.head())

# fig = plt.figure(figsize = (10,8))
# ax1 = fig.add_subplot(121, projection = "3d")
# ax1.scatter(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], c = df["target"], cmap = "rainbow")
# ax1.set_xlabel(iris_target_names[0])
# ax1.set_ylabel(iris_target_names[1])
# ax1.set_zlabel(iris_target_names[2])
#
# ax2 = fig.add_subplot(122, projection = "3d")
# ax2.scatter(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], c = df["Kmeans"], cmap = "rainbow")
# ax2.set_xlabel(iris_target_names[0])
# ax2.set_ylabel(iris_target_names[1])
# ax2.set_zlabel(iris_target_names[2])
# plt.show()

fig, axes = plt.subplots(1,2, figsize = (10,8))


axes[0].scatter(df.iloc[:,0], df.iloc[:,1], c = df["target"], cmap = "rainbow")
axes[0].set_xlabel(iris.feature_names[0])
axes[0].set_ylabel(iris.feature_names[1])



axes[1].scatter(df.iloc[:,0], df.iloc[:,1],c = df["Kmeans"], cmap = "rainbow")
axes[1].set_xlabel(iris.feature_names[0])
axes[1].set_ylabel(iris.feature_names[1])

plt.show()
