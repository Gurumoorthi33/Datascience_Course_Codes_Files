import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    "word_count":[100,50,200,150,60],
    "contain_offer":[1,0,1,0,0],
    "contain_free":[0 ,1,1,0,1],
    "Is_spam":[1,0,1, 0, 1]
}

df = pd.DataFrame(data)

x = df[["word_count", "contain_offer", "contain_free"]]
y = df["Is_spam"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 42)

# model = LogisticRegression()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = KNeighborsClassifier()
model = GaussianNB()
model.fit(x_train,y_train)
y_prediction = model.predict(x_test)
print(y_prediction)

metric_evaluation = accuracy_score(y_test, y_prediction)
print(metric_evaluation)
print(model.classes_)

cm = confusion_matrix(y_test,y_prediction)
display = ConfusionMatrixDisplay(cm, display_labels = model.classes_)
display.plot(cmap = plt.cm.Blues)
plt.grid(True)
plt.show()

plt.figure(figsize = (10,8))
for feature in x.columns:
    sns.kdeplot(df[df["Is_spam"] == 1][feature], label = f"{feature}(spam)", shade = True)
    sns.kdeplot(df[df["Is_spam"] == 0][feature], label=f"{feature}(not spam)", shade=True)
plt.title("feature distribution")
plt.xlabel("features")
plt.ylabel("Density")
plt.legend()
plt.show()

