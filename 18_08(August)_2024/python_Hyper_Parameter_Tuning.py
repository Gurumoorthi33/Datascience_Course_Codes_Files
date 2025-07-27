import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = pd.read_csv("diabetes.csv")
print(data.columns)

Std = MinMaxScaler()
data["SkinThickness"] = Std.fit_transform(data[["SkinThickness"]])
data["Pregnancies"] = Std.fit_transform(data[["Pregnancies"]])
data["Glucose"] = Std.fit_transform(data[["Glucose"]])

x = data[['SkinThickness','Pregnancies', 'Glucose']]
y = data['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 42)
print(x_train.shape)
print(y_train.shape)
print(x_train.dtypes)

model = LogisticRegression()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
model.fit(x_train,y_train)
y_prediction = model.predict(x_test)
print(y_prediction)

metric_evaluation = accuracy_score(y_test, y_prediction)
print(metric_evaluation)



