# from sklearn.tree import DecisionTreeRegressor
# from sklearn.metrics import r2_score
# from sklearn.model_selection import train_test_split
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# df = pd.read_csv('petrol_consumption.csv')
# print(df.head())
#
# df_columns = df.columns
# print(df_columns)
#
# x = df[['Petrol_tax', 'Average_income', 'Paved_Highways', 'Population_Driver_licence(%)']]
# y = df['Petrol_Consumption']
#
# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.2 , train_size= 0.8, random_state= 42)
#
# model = DecisionTreeRegressor()
# model.fit(x_train, y_train)
# y_prediction = model.predict(x_test)
# print(y_test)
# print('-------------------------------------------------------------------------------------')
# print(y_prediction)
#
# r2_score_check = r2_score(y_test, y_prediction)
# print(r2_score_check)


# '''---------------------------------code optimization or Hyperparameter Tuning------------------------------------'''
#
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.metrics import r2_score
# from sklearn.model_selection import train_test_split
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# df = pd.read_csv('petrol_consumption.csv')
# print(df.head())
#
# df_columns = df.columns
# print(df_columns)
#
# x = df[['Petrol_tax', 'Average_income', 'Paved_Highways', 'Population_Driver_licence(%)']]
# y = df['Petrol_Consumption']
#
# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.2 , train_size= 0.8, random_state= 42)
#
# model = DecisionTreeRegressor(max_depth= 5, min_samples_split= 10, min_samples_leaf= 5, random_state= 42)
# model.fit(x_train, y_train)
# y_prediction = model.predict(x_test)
# print(y_test)
# print('-------------------------------------------------------------------------------------')
# print(y_prediction)
#
# r2_score_check = r2_score(y_test, y_prediction)
# print(r2_score_check)


'''---------------------------------Change of Mechanism Entropy------------------------------------'''

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../11_08(August)_2024/petrol_consumption.csv')
print(df.head())

df_columns = df.columns
print(df_columns)

x = df[['Petrol_tax', 'Average_income', 'Paved_Highways', 'Population_Driver_licence(%)']]
y = df['Petrol_Consumption']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.2 , train_size= 0.8, random_state= 42)

model = DecisionTreeRegressor(criterion= 'squared_error',max_depth= 5, min_samples_leaf= 5, random_state= 42)
#model = DecisionTreeClassifier(criterion= 'gini',max_depth= 5, min_samples_leaf= 5, random_state= 42)   #'entropy', 'log_loss', 'gini'
model.fit(x_train, y_train)
y_prediction = model.predict(x_test)
print(y_test)
print('-------------------------------------------------------------------------------------')
print(y_prediction)

r2_score_check = r2_score(y_test, y_prediction)
print(r2_score_check)