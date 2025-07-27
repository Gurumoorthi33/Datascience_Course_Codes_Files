import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score

data = pd.read_csv("framingham.csv")
print(data.columns)

data.fillna(1, inplace = True)

x = data[['male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'diabetes', 'totChol', 'sysBP','diaBP', 'BMI']]
y = data['TenYearCHD']

skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
model = LogisticRegression()
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 32)
cross_val_check = cross_val_score(model, x_train, y_train,cv = skf, scoring = "accuracy")
print(cross_val_check.max()) #Use this when were doing classification
print(cross_val_check.min()) #use this when were doing regression (mean_square)

model.fit(x_train, y_train)
prediction = model.predict(x_test)
print(prediction)







