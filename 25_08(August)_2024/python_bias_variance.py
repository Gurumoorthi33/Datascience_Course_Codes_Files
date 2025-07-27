import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

x = np.linspace(1,10,100).reshape(-1, 1)
y = np.sin(x).ravel() + np.random.normal(0.2, size = x.shape[0])

plt.figure(figsize = (10,8))

plt.subplot(1,2,1)
model_bias = LinearRegression().fit(x,y)
y_prediction = model_bias.predict(x)

plt.scatter(x, y, color = "black", label = "Data")
plt.plot(x, y_prediction, color = "red", label = 'high bias model(Underfitting)')
plt.grid(True)
plt.legend()

plt.subplot(1,2,2)
model_var = PolynomialFeatures(degree=4)
x_poly = model_var.fit_transform(x)
model_variance = LinearRegression().fit(x_poly, y)
y_variance_prediction = model_variance.predict(x_poly)
plt.scatter(x, y, color = "black", label = "Data")
plt.plot(x, y_variance_prediction, color = "red", label = "High Variance Model(Overfitting)")
plt.legend()
plt.grid(True)
plt.show()



