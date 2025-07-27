import pandas as pd

data = {'date':['Monday, July, 2024']}
df = pd.DataFrame(data)
print(df)
data_format = "%A, %B, %Y"
df['converted'] = pd.to_datetime(df['date'],format = data_format)
print(df)
print(df.dtypes)


