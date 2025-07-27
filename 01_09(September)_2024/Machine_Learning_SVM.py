import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("Raisin_Dataset.csv")
print(df.head())
print(df.info)
print(df.describe())
print(df.columns)

x = df[['MinorAxisLength','Extent', 'Perimeter',"Eccentricity",'ConvexArea']]
y = df['Class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 42)

model = SVC()
model.fit(x_train, y_train)
y_prediction = model.predict(x_test)
print(y_prediction)

accuracy_score_check = accuracy_score(y_test, y_prediction)
print(accuracy_score_check)




