import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# pd.set_option('display.max_columns',None)
# pd.set_option('display.max_rows',None)
data = pd.read_csv('CarPredict_Dataset.csv')
print(data.head())

x = data[['brand','fuel_type']]
y = data['price']

'''Below six lines of code is to execute the label encoding to each column Separately'''

encoder = OneHotEncoder(sparse_output = False)
encoded_data = encoder.fit_transform(data[['fuel_type']])
print(data)

data['fuel_type'] = encoder.fit_transform(data[['fuel_type']])
print(data['fuel_type'])
print(data.head(20))

new_encoded_column_names = encoder.get_feature_names_out(['fuel_type'])
print(new_encoded_column_names)

new_df = pd.DataFrame(encoded_data , columns = new_encoded_column_names)
print(new_df)

data = pd.concat([data.drop(columns=['fuel_type'], inplace = True), new_df], axis=1)  #axis = 1 --- append the data row vise
print(data.head())

#axis = 1 --------> merge the column one another one

'''-----------------------------------------------Co-pilot Code----------------------------------'''


# import pandas as pd
# from sklearn.preprocessing import OneHotEncoder
#
# # Set pandas display options
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
#
# # Load the dataset
# data = pd.read_csv('CarPredict_Dataset.csv')
# print(data.head())
#
# # Extract features and target
# x = data[['brand', 'fuel_type']]
# y = data['price']
#
# # One-hot encode the 'fuel_type' column
# encoder = OneHotEncoder(sparse_output=False)
# encoded_data = encoder.fit_transform(data[['fuel_type']])
#
# # Retrieve new column names
# new_encoded_column_names = encoder.get_feature_names_out(['fuel_type'])
# print(new_encoded_column_names)
#
# # Create a DataFrame from the encoded data
# new_df = pd.DataFrame(encoded_data, columns=new_encoded_column_names)
#
# # Merge the new encoded columns with the original dataset
# data = pd.concat([data.drop(columns=['fuel_type'],inplace = True), new_df], axis=1)
# print(data.head())
