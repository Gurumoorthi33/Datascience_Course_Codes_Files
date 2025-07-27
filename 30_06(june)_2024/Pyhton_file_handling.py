'''File handling is the concept of handling the various manipulation
Files: csv,excel,json,pdf,text

library : pandas,csv'''

import pandas

#a = pandas.read_csv('housing.csv')
#print(a.head()) #head() will return first five data of csv file
#print(a.tail()) #tail() will return last five data of csv file

#print(a)

#output = pandas.read_csv('housing.csv')
#print(output.head())
#print(output['ocean_proximity'])

#a = pandas.read_csv('housing.csv')
#print(a['ocean_proximity'].head())

# a = pandas.read_csv('housing.csv')
# print(a.ocean_proximity.head())

#a = pandas.read_csv('housing.csv')
#print(a[['ocean_proximity','longitude']].head()) #double square braces needed for reading more than one column data


#a = pandas.read_csv('housing.csv')
#column = a.columns
#print(column)

#a = pandas.read_csv('housing.csv')
#index_output = a.index
#print(index_output)


#a = pandas.read_csv('housing.csv')
#data_output = a.values
#print(data_output)

#a = pandas.read_csv('housing.csv')
#print(a.values)

'''Dictionary is a key and value pair
interviwe tips: all the datatypes that we associate in python is single dimension '''


# data = {'name':['dhoni','virat','rohit','ashwin'],'age':[42,33,34,39]} #dictionary data
# print(data)
# print(type(data))
#
# print('----------------------------------------------')
#
# df = pandas.DataFrame(data)
# print(df)
# print(type(df))
#
# df.to_csv('cricket player.csv')

# output = pandas.read_csv('housing.csv')
# print(output.ndim) #checking the dimension of hte data(rows,column)
# print(output.shape) #checking the shape of their given dataset

# output = pandas.read_csv('housing.csv')
# data_description = output.describe()
# print(data_description)

# output = pandas.read_csv('housing.csv')
# data_description = output.info() #it si a statiscal function
# print(data_description)

# output = pandas.read_csv('housing.csv')
# data_description = output.isnull().sum().sum() #it is to check the nullset from entire dataset
# print(data_description)

# output = pandas.read_csv('housing.csv')
# data_description = output.loc[1,5]
# print(data_description)


# output = pandas.read_csv('housing.csv')
# data_description = output.iloc[1,3] # it picks first row and third column
# print(data_description)

# output = pandas.read_csv('housing.csv')
# data_description = output.iloc[1:3,3] # it picks first row ,3rd row and third column
# print(data_description)
#iloc = integer based loaction finder

# output = pandas.read_csv('housing.csv')
# data_description = output.loc[1:5,['total_rooms','ocean_proximity']] # it picks first,2nd,3rd row from third column
# print(data_description)

# a = pandas.read_csv('housing.csv')
# b = a.backfill
# print(b)

# a = pandas.read_csv('housing.csv')
# b = a.loc[1:3,'ocean_proximity']
# print(b)


# Bracket represents the functions


a = pandas.read_csv('housing.csv',header = None)
print(a.ndim)
print(a.shape)
print(a.describe())
print(a.info())
print(a.isnull().sum().sum())

output = a.iloc[1:3,3]
print(output)


b = pandas.read_csv('housing.csv')
out = b.loc[1:3,'ocean_proximity']
print(out)

out = b.loc[1:3,['latitude','ocean_proximity']]
print(out)

# the position of a student or trainee who works in an organization, sometimes without pay,
# in order to gain work experience or satisfy requirements for a qualification.

