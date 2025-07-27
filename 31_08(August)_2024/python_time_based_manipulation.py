import pandas as pd
import numpy as np

dates = pd.date_range("2024-01-01", periods = 10, freq = "D")
data = np.random.randn(10)

df = pd.DataFrame({"first":dates, "second":data})
print(df.head())
print('---------------------------------------------------------------------')
df.set_index("first", inplace = True)
print(df.head())

