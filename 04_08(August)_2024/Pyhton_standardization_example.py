import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.DataFrame({'score':[70,80,50,33,44,67,80]})
print(data.mean())

mean_one = data.iloc[[1,4]]['score'].mean()
print(mean_one)

mean_one = data.iloc[[1,3]]['score'].mean()
print(mean_one)

print('------------After Standardization-----------------------')

Standard_Scaler_data = StandardScaler()
data['score'] = Standard_Scaler_data.fit_transform((data[['score']]))

mean_one = data.iloc[[1,4]]['score'].mean()
print(mean_one)

mean_one = data.iloc[[1,3]]['score'].mean()
print(mean_one)
