import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sl

data = pd.read_csv('housing.csv')
print(data)

finding_null = data.isnull()
print(finding_null)
'''below the two  lines code is fill all the empty cell with 1000'''

# data.fillna(1000,inplace = True)
# print(data.head(5))
'''below the two lines code is only fill with latitude column with 1000'''

# data['latitude'].fillna(1000,inplace = True)
# print(data.head(5))

'''below the two line will not work as this syntax is not correct '''

# data[['latitude','population']].fillna(1000,inplace = True)
# print(data.head(5))

'''below is the correct syntax to change the two column value at one short'''

# data.fillna({'latitude':1000,'longitude':1000},inplace = True)
# print(data)

'''filling the null with new functions'''
# data.fillna({'latitude':data['housing_median_age'].mean(),'longitude':data['housing_median_age'].median()},inplace = True)
# print(data)

'''Below the method will fill the previous data into the empty row'''

# data.ffill(inplace = True)
# print(data)

'''Below the method will fill the next data value into the empty row '''
# data.bfill(inplace = True)
# print(data)

'''delete the entire row that has null values'''

# data.dropna(inplace = True)
# print(data)















