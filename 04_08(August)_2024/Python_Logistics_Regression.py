import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix, precision_score , recall_score , f1_score , classification_report
from sklearn.impute import SimpleImputer


df = pd.read_csv('framingham.csv')
#df.ffill(inplace = True)
'''-----------------------------------using simple imputer---------------------------------------------'''
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
imputer_int = SimpleImputer(strategy='mean')
df[numerical_columns] = imputer_int.fit_transform(df[numerical_columns])

'''----------------------using fill values----------------------------------------------------'''
# df.fillna(1 , inplace = True)
print(df.head())

x = df[['currentSmoker','BPMeds','cigsPerDay']]
y = df['TenYearCHD']


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.2, random_state= 42)

model = LogisticRegression()
model.fit(x_train,y_train)
prediction = model.predict(x_test)
print(prediction)

model_accuracy_check =  accuracy_score(y_test, prediction)
print(model_accuracy_check)

precision_score = precision_score(y_test, prediction)
print(precision_score)

'''In machine learning, precision measures the accuracy of positive predictions,
 specifically the proportion of true positive predictions among all positive predictions made by the model.
 It's calculated as True Positives / (True Positives + False Positives). '''

recall_score_check = recall_score(y_test , prediction)
print(recall_score_check)

f1_score_check = f1_score(y_test , prediction)
print(f1_score_check)

confusion_matrix = confusion_matrix(y_test , prediction)
print(confusion_matrix)

'''A confusion matrix is a table that is used to define the performance of a classification algorithm. 
A confusion matrix visualizes and summarizes the performance of a classification algorithm.'''

'''A confusion matrix is a table that visualizes the performance of a classification model 
by comparing its predictions to the actual values,
 showing the counts of true positives, true negatives, false positives, and false negatives. '''

classification_report_testing = classification_report(y_test, prediction)
print(classification_report_testing)


