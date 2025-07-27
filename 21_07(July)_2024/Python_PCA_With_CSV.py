import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = pd.read_csv('quality.csv')
print(data.head())

y = data.pop("quality")
print(data.head())

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

new_df = pd.DataFrame(scaled_data, columns = data.columns)
print(new_df)

model = PCA()
principal_component = model.fit_transform(new_df)
print(principal_component)

pd.DataFrame(model.explained_variance_ratio_).plot.bar()
plt.xlabel("Principal Components")
plt.ylabel("Explained variance ")
plt.show()
#jenkins and concos


