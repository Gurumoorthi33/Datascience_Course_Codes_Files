import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

x = np.linspace(1,10,100).reshape(-1, 1)
y = np.sin(x).ravel() + np.random.normal(0.2, size = x.shape[0])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 42)

model_high_bias = LinearRegression()
poly_high_bias = PolynomialFeatures(degree = 1)
x_poly_train_bias = poly_high_bias.fit_transform(x_train)
x_poly_test_high_bias = poly_high_bias.transform(x_test)
model_high_bias.fit(x_poly_train_bias, y_train)
y_prediction_train_bias = model_high_bias.predict(x_poly_train_bias)
y_prediction_test_bias = model_high_bias.predict(x_poly_test_high_bias)


model_high_variance = LinearRegression()
poly_high_variance = PolynomialFeatures(degree = 15)
x_poly_train_variance = poly_high_variance.fit_transform(x_train)
x_poly_test_high_variance = poly_high_variance.transform(x_test)
model_high_variance.fit(x_poly_train_variance, y_train)
y_prediction_train = model_high_variance.predict(x_poly_train_variance)
y_prediction_test = model_high_variance.predict(x_poly_test_high_variance)


model_balanced = LinearRegression()
poly_balanced = PolynomialFeatures(degree = 4)
x_poly_train_balanced = poly_balanced.fit_transform(x_train)
x_poly_test_balanced = poly_balanced.transform(x_test)
model_balanced.fit(x_poly_train_balanced, y_train)
y_prediction_train_balance = model_balanced.predict(x_poly_train_balanced)
y_prediction_test_balance = model_balanced.predict(x_poly_test_balanced)


final_df = pd.DataFrame(
    {"model": ["High Bias", "High Variance", "Balanced"],
     "Train_MSE": [mean_squared_error(y_train, y_prediction_train_bias),
                   mean_squared_error(y_train, y_prediction_train),
                   mean_squared_error(y_prediction_train_balance, y_train)],
     "Test_MSE": [mean_squared_error(y_test, y_prediction_test_bias),
                  mean_squared_error(y_test, y_prediction_test),
                  mean_squared_error(y_prediction_test_balance, y_test)]}
)
print(final_df)


