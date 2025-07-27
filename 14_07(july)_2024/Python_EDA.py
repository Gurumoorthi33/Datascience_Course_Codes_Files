from calendar import c

import pandas as pd
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as ss
import numpy as np
import scipy.stats as st

# data = pd.read_csv('housing.csv')
# print(data.head())
# print(data.info())
# print(data.describe())
#
# '''Below the is to fetch all the column names in which data belongs to either int or float'''
# numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
# print(numerical_columns)
#
# '''Below the is to fetch all the column names in which data belongs to object '''
# categorical_columns = data.select_dtypes(include=['object']).columns
# print(categorical_columns)
#
# '''simple imputer for numerical based'''
# imputer_int = SimpleImputer(strategy='mean')
# data[numerical_columns] = imputer_int.fit_transform(data[numerical_columns])
# print(data)
#
# imputer_obj = SimpleImputer(strategy='most_frequent')
# data[categorical_columns] = imputer_obj.fit_transform(data[categorical_columns])
# print(data)
# # print(data.info())
# # #
# for columns in numerical_columns:
#     plt.figure(figsize = (10,8))
#     ss.histplot(data[columns])
#     plt.title('histogram of {columns}')
#     plt.show()
#
# for columns in numerical_columns:
#     plt.figure(figsize = (10,4))
#     ss.boxplot(x = data[columns])    #kernel density estimation
#     plt.title('histogram of {columns}')
#     plt.show()
#
# plt.figure(figsize = (10,4))
# ss.boxplot(x = data['population'])
# plt.title('histogram of population')
# plt.show()
#
# plt.figure(figsize = (10,8))
# ss.countplot(x = 'ocean_proximity',data = data)
# plt.title('count plot of hte ocean') # especially used for categorical data
# plt.show()
#
# plt.figure(figsize=(10, 8))
# var = ss.countplot(x='ocean_proximity', data=data)
# plt.title('count plot of hte ocean')  # especially used for categorical data
#
# for v in var.patches:
#     height = v.get_height()
#     var.annotate(f'{height}', (v.get_x() + v.get_width() / 2., height), ha='center', va='center', xytext = (0, 10),textcoords='offset points')
#
# plt.show()
#
#
# corr = data.drop(columns = ['ocean_proximity']).corr()
# plt.figure(figsize = (12,8))
# ss.heatmap(corr,annot = True, cmap = 'coolwarm') #viridis,rainbow,magma
# plt.title('correlation matrix')
# plt.show()
#
# plt.figure(figsize = (10,8))
# ss.countplot(x = data['population'])
# plt.title(f'histogram of population')
# plt.show()
#
# data = pd.read_csv('housing.csv')
#
# '''--------------------------------------------------------------------------------------------------'''
# mean = np.mean(data['median_house_value'])
# print(mean)
# mean = data['median_house_value']
# print(mean)
#
# mode = st.mode(data['median_house_value'])
# print(mode)
#
# skew = st.skew(data['median_house_value'])
# print(skew)
#
# kurtosis = st.kurtosis(data['median_house_value'])
# print(kurtosis)
#
# plt.figure(figsize = (8,6))
# ss.histplot(data['median_house_value'], kde = True)
# plt.title('Histogram for median house skew value')
# plt.xlabel('values')
# plt.ylabel('frequency')
# plt.show()
#
# print('---------------------------------------------------------------------------------------')
#
# data = pd.read_csv('housing.csv')
# print(data.info())
# print(data.describe())
#
# numerical_columns = data.select_dtypes(include = ['float64','int64']).columns
# print(numerical_columns)
#
# categorical_columns = data.select_dtypes(include = ['object']).columns
# print(categorical_columns)



# plt.figure(figsize = (8,6))
# ss.countplot(x = 'ocean_proximity', data = data)
# plt.title('Its time to start')
# plt.show()
#
# a = pd.read_csv('housing.csv')
# print(a.tail(20))
'''----------------------------------------------------------------------------------------------------------'''
data = pd.read_csv("housing.csv")


mode = st.mode(data["median_house_value"])
print(mode)

skew = st.skew(data["median_house_value"])
print(skew)

kurtosis = st.kurtosis(data["median_house_value"])
print(kurtosis)
