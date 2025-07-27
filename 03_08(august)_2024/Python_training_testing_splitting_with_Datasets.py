from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('housing.csv')

x = data[['total_rooms','total_bedrooms','population']]
y = data['median_house_value']


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,train_size=0.8,random_state=42)

print('-----------------------x_train')
print(x_train)

print('-----------------------x_test')
print(x_test)

print('-----------------------y_train')
print(y_train)

print('-----------------------y_test')
print(y_test)

# model = LinearRegression()
# model.fit(x_train,y_train)
# print(x_test)
# print(y_test)
#
# prediction = model.predict(x_test)
# print(prediction)









