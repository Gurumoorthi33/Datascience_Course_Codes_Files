import pandas as pd
from sklearn.cluster import DBSCAN

x = [1,2,3,4,5,20]

y = [2,4,6,8,10,400]

df = pd.DataFrame(x,y)

db_outliers = DBSCAN(eps = 3,min_samples = 2).fit(df)

df['Cluster'] = db_outliers.labels_

outliers = df[df['Cluster'] == -1 ]
print(outliers)


'''lower the epsilon(eps) value - higher the Clustering Formation - Proper Outliers Formation'''