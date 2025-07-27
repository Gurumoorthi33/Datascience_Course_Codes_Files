import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("petrol_consumption.csv")
print(data.columns)

x = data[['Petrol_tax', 'Average_income', 'Paved_Highways', 'Population_Driver_licence(%)']]
y = data['Petrol_Consumption']

model = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 32)
cross_val_check = cross_val_score(model, x_train, y_train,cv = 5, scoring = "r2")
print(cross_val_check.max()) #Use this when were doing classification
print(cross_val_check.min()) #use this when were doing regression (mean_square)

model.fit(x_train, y_train)
prediction = model.predict(x_test)
print(prediction)