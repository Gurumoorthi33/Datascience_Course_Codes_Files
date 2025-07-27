from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

data = {'year':[2000,2001,2002,2003,2004,2005],'price':[5000,6000,7000,8000,9000,10000]}
df = pd.DataFrame(data)
#print(df)

x = df[['year']]
y = df['price']

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,train_size=0.8,random_state = 42)


'''-------------------------train data sets--------------------------------------------'''
# print('--------------x_train')
# print(x_train)
# print('-------------------y_train')
# print(y_train)


model = LinearRegression()
model.fit(x_train,y_train)
print(x_test)
print(y_test)

prediction = model.predict(x_test)
print(prediction)


# prediction = model.predict(pd.DataFrame([2006]))
# prediction = model.predict(pd.DataFrame({'year':[2006]}))
# print(prediction)

r2_score_output = r2_score(y_test,prediction)
r2_score_output1 = r2_score(y_test,pd.Series(prediction))
print(r2_score_output)
print(r2_score_output1)

plt.scatter(x_test, y_test, color = 'blue', label = 'actual input data')
plt.plot(x_test, prediction, color = 'red', label = 'Predicted value')
plt.legend()
plt.show()

#AI involves advanced mathematics, algorithms, and programming,
# while Cyber Security demands a deep understanding of systems, vulnerabilities, and attack strategies.





