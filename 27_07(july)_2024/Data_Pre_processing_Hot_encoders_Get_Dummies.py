import pandas as pd


pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
data = pd.read_csv('CarPredict_Dataset.csv')
print(data.head())

x = data[['brand','fuel_type']]
y = data['price']

output = pd.get_dummies(x['fuel_type'])
print(output.head())

