import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = {
    "Age" : [25,30,35,40,45],
    "Height" : [150,168,170,190,180],
    "weight" : [55,65,75,85,95]
}
df = pd.DataFrame(data)

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

average = df['Age'][:4].mean()
print(average)

new_df = pd.DataFrame(scaled_data,columns = ["Age","Weight","Height"] )
print(new_df)

average = new_df["Age"].mean()
print(average)
a1 = new_df["Age"][:3].mean()
print(a1)
a2 = new_df['Age'][:4].mean()
print(a2)