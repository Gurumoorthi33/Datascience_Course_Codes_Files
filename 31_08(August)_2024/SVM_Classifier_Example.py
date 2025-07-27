from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# 1️⃣ Create synthetic data
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# 2️⃣ Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3️⃣ Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4️⃣ Train SVM classifier
svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale')  # You can experiment with 'linear', 'poly', or 'sigmoid' kernels
svm_clf.fit(X_train_scaled, y_train)

# 5️⃣ Evaluate the model
y_pred = svm_clf.predict(X_test_scaled)
print(classification_report(y_test, y_pred))