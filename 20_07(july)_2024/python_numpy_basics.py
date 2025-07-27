import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# np.random.seed(42)
#
# x = np.random.rand(100)*10
# y = 2.5 * x + np.random.randn(100) * 2 + 5
#
# df = pd.DataFrame({'x_data':x , 'y_data': y})
# print(df)

'''Exploring numpy function'''

np.random.seed(42)
x = np.random.rand(100)
print(x)

y = np.random.randn(100)
print(y)

'''Column_stack function is basically used to convert into different 1D array into 2D array'''
data = np.column_stack([x,y])
df = pd.DataFrame(data,columns = ['Guru','Dhoni'])
print(df)

'''Below code is error code We can't convert into two separate entity into df'''
# df = pd.DataFrame([x,y],columns = ['x','x1'])
# print(df)

'''raise ValueError(err) from err
ValueError: 2 columns passed, passed data had 100 columns'''