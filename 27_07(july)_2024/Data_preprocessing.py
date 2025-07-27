# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
#
# pd.set_option('display.max_columns',None)
# data = pd.read_csv('CarPredict_Dataset.csv')
# print(data.head())
#
# x = data[['brand','fuel_type']]
# y = data['price']
#
# '''Below six lines of code is to execute the label encoding to each column Separately'''
#
# encoder = LabelEncoder()
# data['brand'] = encoder.fit_transform(data['brand'])
# print(data['brand'])
# print(data.head(20))
#
# data['fuel_type'] = encoder.fit_transform(data['fuel_type'])
# print(data['fuel_type'])
# print(data.head(20))
#
# ''''LabelEncoding is the single entity of data'''
#
# '''Below two lines of code is to execute the label encoding to each column together'''
#
# total_encoding = x.apply(encoder.fit_transform)
# print(total_encoding)
#
# data.drop(columns = ['brand','fuel_type'], inplace = True)
#
# '''Method - 2'''
#
# final_data = pd.concat([data,total_encoding], axis = 1) #Adding the two columns into one
# print(final_data)
#
# # total_categories = pd.unique()
# # print(total_categories)
#
# total_categories = final_data['brand'].value_counts()
# print(total_categories)

'''----------------------------Method - 2-----------------------------------------------'''

# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
#
# data = pd.read_csv('CarPredict_Dataset.csv')
#
# x = data[['brand','fuel_type']]
# y = data['price']
#
# encoder = LabelEncoder()
#
# data['encoded_brand'] = encoder.fit_transform(data['brand'])
# print(data)
# data.to_csv('output.csv')


'''Below two lines of code is to execute the label encoding to each column together'''

# total_encoding = x.apply(encoder.fit_transform)
# print(total_encoding)
#
# data.drop(columns = ['brand','fuel_type'], inplace = True)
# final_data = pd.concat([data,total_encoding], axis = 1) #Adding the two columns into one
# print(final_data)
'''One hand Encoding'''

# total_categories = pd.unique(data["price"])
# print(total_categories)
#
# total_categories = final_data['brand'].value_counts()
# print(total_categories)
#
# total_categories = pd.unique()
# print(total_categories)
#
# brand_value_counts = data[['encoded_brand']].value_counts()
# print(brand_value_counts)



'''----------------------------Mapping Categories  With Encoded Value------------------------------------'''

# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
#
# data = pd.read_csv('CarPredict_Dataset.csv')
#
# x = data[['brand','fuel_type']]
# y = data['price']
#
# encoder = LabelEncoder()
#
# data['encoded_brand'] = encoder.fit_transform(data['brand'])
# print(data)
# data.to_csv('output.csv')
#
#
'''Below two lines of code is to execute the label encoding to each column together'''




# brand_value_counts = data[['encoded_brand']].value_counts()
# print(brand_value_counts)
#
# class_of_encoder = encoder.classes_
# print(class_of_encoder)
#
# for x,categories in enumerate(encoder.classes_): #enumerate = index,value
#     print(f'Label{x}:{categories} count:{brand_value_counts[x]}')

'''-----------------How To Convert The all the Categorical Column of our Entire datasets-------------'''

# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
#
# data = pd.read_csv('CarPredict_Dataset.csv')
#
# cat_colm = data.select_dtypes(include = ['object']).columns
#
# encoder = LabelEncoder()
#
# data[cat_colm].apply(encoder.fit_transform)
# print(data)
# print(cat_colm)

# column encoding - Single shot encoding in the for all column

'''----------------------------------------Practice--------------------------------------------------------------'''
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)

data = pd.read_csv("CarPredict_Dataset.csv")

# x = data[["brand", "fuel_type"]]
# y = data["price"]
#
# encoder = LabelEncoder()
# data["brand"] = encoder.fit_transform(data["brand"])
# print(data["brand"])
# print(data.head(20))
#
# data["fuel_type"] = encoder.fit_transform(data["fuel_type"])
# print(data["fuel_type"])
#
# total_encoding = x.apply(encoder.fit_transform)
# print(total_encoding)
#
# total_categories = pd.unique(data["price"])
# print(total_categories)

dd = data.nunique()
print(dd)