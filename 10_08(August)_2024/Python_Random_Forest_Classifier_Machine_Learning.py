#Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('diabetes.csv')
print(df.head())

df_columns = df.columns
print(df_columns)

#df = df.fillna(1, inplace = True)

x = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']]
y = df['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.2 , train_size= 0.8, random_state= 42)

#model = RandomForestRegressor(random_state= 42)
model = RandomForestClassifier(criterion = "entropy", random_state= 42)



hyper_parameter_input = {
    'n_estimators': [50,100,200],
    'max_depth': [None, 10,20,30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1,2,4],
    'max_features': ["auto", "sqrt", "log2"]
}

grid_model = GridSearchCV(model, param_grid = hyper_parameter_input, cv = 5)
grid_model.fit(x_train, y_train)
best_parameter = grid_model.best_params_
best_estimators = grid_model.best_estimator_

y_prediction = best_estimators.predict(x_test)
print(y_test)
print('-------------------------------------------------------------------------------------')
print(y_prediction)

r2_score_check = r2_score(y_test, y_prediction)
print(r2_score_check)

