import pandas as pd
import numpy as np

data = {'Flu':[0,1,0,1,0],
        'fever':[100,101.44,98,99.0,102],
        'smell':[1,1,0,0,1],
        'fatique':[1,2,3,2,1],
        'corona':[0,1,0,1,0]}
df = pd.DataFrame(data)
#print(df)
covariance_matrix = df.corr()
print(covariance_matrix)
print(covariance_matrix['corona'])