import matplotlib.pyplot as plt

var = ('gur\nu')
print(var)

var = (r'gur\nu')  #r ---- is a raw data which means that it gives the sentence without any changes in the
#given data.what will be the inpu without changes it give the exact output as what we gave.
print(var)

import pandas as pd

data = pd.read_csv('AdidasSalesdata.csv')
print(data.head())
print(data.tail())
print(data.ndim,"Good")
print(data.shape)
print(data.describe())
print(data.isnull().sum().sum())
print(data.info())
print(data.iloc[1,3])
print(data.loc[1:5,['State']])
print(data.nunique())
data.dropna(inplace = True)
print(data)





# import matplotlib.pyplot as plt
# import seaborn as sns
#
# x = data['Product Category']
# plt.bar(x, height = 20)
# plt.show()
#
# data = pd.read_csv("AdidasSalesdata.csv")
#
# party_count = data["State"].value_counts()
# #
# print(party_count)
# # #
# plt.figure(figsize=(10,8))
# #
# plt.pie(party_count, labels = party_count.index, autopct = "%5.1f%%", startangle = 180)
# #
# plt.show()
#
# from sklearn.impute import SimpleImputer
#
# numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
# print(numerical_columns)
#
# '''Below the is to fetch all the column names in which data belongs to object '''
# categorical_columns = data.select_dtypes(include=['object']).columns
# print(categorical_columns)
#
# # '''simple imputer for numerical based'''
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
#     sns.histplot(data[columns])
#     plt.title('histogram of {columns}')
#     plt.show()
#
# for rows in categorical_columns:
#     plt.figure(figsize=(10, 8))
#     sns.histplot(data[rows])
#     plt.title(f'histogram of {rows}')
#     plt.show()

# data['Retailer'] = data['Retailer'].str.replace(',','  ')  # it replaces comma or any spaced indentation and we can
# #Other things what we need.
# print(data)

# data['Price per Unit'] = data['Price per Unit'].str.replace("[$,]","",regex = True).str.replace(' ','').astype(float)  # it replaces comma or any spaced indentation and we can
# #Other things what we need.
# print(data)

#unstack method = compare the data one by one with Color Difference

# data['Gender'] = data['Product Category'].str.split().str(0).str.replace("'s","")
# print(data['Gender'])
# unit_sold_by_gender = data.groupby(['Gender','Product Category']) # It is to combine the two column for to find the difference

# data = {'date':['Monday, July, 2024 ']}
# df = pd.DataFrame(data)
# data_format = "%A,%B,%Y"
# df['converted'] = pd.to_datetime(df['date'],format = data_format)
# print(df)

units_sold_by_category_gender = data.groupby(["Product Category" , "Gender Type" ])["Units Sold"].sum().unstack()
print(f"Units sold by the gender category \n {units_sold_by_category_gender}" )

plt.figure(figsize = (10,8))
units_sold_by_category_gender.plot(kind = "bar", stacked = True)
plt.title("Units sold by Product category and Gender Type")
plt.xlabel("Product Category")
plt.ylabel("Units sold by gender types")
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()




