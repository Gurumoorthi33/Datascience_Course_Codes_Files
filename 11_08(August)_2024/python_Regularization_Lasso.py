# from sklearn.linear_model import Lasso
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
#
# data = pd.DataFrame({"a":[1,2,3,4,5], "b":[2,4,6,8,10], "c":[2,3,4,5,6]})
# x = data[["a", "b"]]
# y = data["c"]
#
# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
# model = LinearRegression()
# model.fit(x_train, y_train)
# output = model.predict(x_test)
# print(output)
# metric_check = mean_squared_error(y_test, output)
# print(metric_check)
# model_coef = model.coef_
# print(model_coef)

'''With Lasso'''


from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.DataFrame({"a":[1,2,3,4,5], "b":[2,4,6,8,10], "c":[2,3,4,5,6]})
x = data[["a", "b"]]
y = data["c"]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
model = Lasso(alpha = 0.1)
model.fit(x_train, y_train)
output = model.predict(x_test)
print(output)
metric_check = mean_squared_error(y_test, output)
print(metric_check)
model_coef = model.coef_
print(model_coef)
# What model.coef_ Represents
#- Linear Regression: Each coefficient represents the weight of a feature in predicting the target.

# - Logistic Regression: Coefficients describe the effect of each feature on the log-odds of the outcome.
# Higher values mean stronger influence on predicting class probability.
# - In multivariate cases (e.g., multi-class classification), model.coef_ will be a 2D array —
# each row corresponds to a class, and each column to a feature.

