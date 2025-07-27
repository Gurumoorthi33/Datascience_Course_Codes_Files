from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler , MinMaxScaler

data = pd.read_csv('Cellphone.csv')
print(data.head())

standard_scaler_data = StandardScaler()
#min_max_scaler_data = MinMaxScaler()
data['thickness'] = standard_scaler_data.fit_transform((data[['thickness']]))
#data['thickness'] = min_max_scaler_data.fit_transform((data[['thickness']]))

x = data[[ 'Product_id',  'Price',  'Sale']]
y = data['thickness']

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,train_size=0.8,random_state = 42)

model = LinearRegression()
model.fit(x_train,y_train)
print(x_test)
print(y_test)

prediction = model.predict(x_test)
print(prediction)

r2_score_output = r2_score(y_test,prediction)
r2_score_output1 = r2_score(y_test,pd.Series(prediction))
print(r2_score_output)
print(r2_score_output1)

mse_output = mean_squared_error(y_test, prediction)
print(mse_output)




