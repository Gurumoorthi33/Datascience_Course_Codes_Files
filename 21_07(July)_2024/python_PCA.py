import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = {
    "age":[21,22,23,24,25],
    "Height":[100,100,100,100,100],
    "Weight":[25,35,45,55,65]
}

df = pd.DataFrame(data)

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)
print(scaled_data)

model = PCA(n_components = 2)
principal_components = model.fit_transform(scaled_data)
print(principal_components)

new_df = pd.DataFrame(principal_components, columns = ["PC1" , "PC2"])
print(new_df)

explained_variance = model.explained_variance_ratio_
print(explained_variance)

