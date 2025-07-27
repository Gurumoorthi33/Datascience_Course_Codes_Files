import pandas as pd
import numpy as np

date_input = pd.date_range("2021-01-01", periods = 100, freq = "D")
data_input = np.random.randn(100)

df = pd.DataFrame({"Date":date_input, "value":data_input})
print(df.head())
df.to_csv("set_index-non.csv")

df.set_index("Date", inplace = True)
print(df.head())
df.to_csv("set_index.csv")
