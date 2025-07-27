'''KNN is a Supervised learning
It Uses the distance metrics such as eculidean distance to find the near by data point
Load iris - Flower type setosa, versicolor and virgnica
k = 3, then the algorithm will took for 3 flowers which has similar attributes
if 2 belongs to virginca, one belongs to setosa

General: It calculates the distance to all points in the training data and select teh closesest one

classification -----> Majority
Regressor -------> Average value
Iris - This dataset is ideal for classification and clustering functions.'''

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)
y_prediction = model.predict(x_test)
print(y_prediction)

accuracy_score_check = accuracy_score(y_test, y_prediction)
print(accuracy_score_check)

