import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

np.random.seed(42)

x = np.random.rand(100)*10
y = 2.5 * x + np.random.randn(100) * 2 + 5

df = pd.DataFrame({'x_data':x , 'y_data': y})
print(df)

x_with_const = sm.add_constant(df['x_data'])
print(x_with_const)

model = sm.OLS(df['y_data'],x_with_const).fit()
print(model.summary())
print(model)

#rand = value ranges from 0 to 1
#randn = value ranges from -1 to 1






