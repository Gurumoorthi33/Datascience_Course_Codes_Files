import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

data = pd.read_csv('housing.csv')

mean_imputer = SimpleImputer(strategy='mean')
data['latitude'] = mean_imputer.fit_transform(data[['median_house_value']])
print(data)

'''Various types of strategy can be used

strategy ='median'
strategy = 'most_frequent'
strategy = 'constant' '''

data['longitude'] = SimpleImputer(strategy = 'mean').fit_transform(data[['longitude']])
print(data)










