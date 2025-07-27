import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = {
    "word_count":[100,50,200,150,60],
    "contain_offer":[1,0,1,0,0],
    "contain_free":[0 ,1,1,0,1],
    "Is_spam":[1,0,1, 0, 1]
}

df = pd.DataFrame(data)

x = df[["word_count", "contain_offer", "contain_free"]]
y = df["Is_spam"]

# Feature scaling (KNN is distance-based, so scaling is important)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Instantiate and train KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)

# Predict and evaluate
y_prediction = model.predict(x_test)
print("Predictions:", y_prediction)

metric_evaluation = accuracy_score(y_test, y_prediction)
print("Accuracy:", metric_evaluation)