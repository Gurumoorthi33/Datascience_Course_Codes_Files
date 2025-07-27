import numpy as np
import pandas as pd

data = {
    "my_date": pd.date_range("2024-01-01", periods = 10, freq = "D"),
    "Value": [21, 20, 22, 23, 34, 54, 65, 87, 23, 28]
}

df = pd.DataFrame(data)
df.set_index("my_date", inplace = True)

naive_forecasting_data = df['Value'].iloc[-1]
print(naive_forecasting_data)

future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days = 1), periods = 10, freq = "D")
print(future_dates)

future_forecasting = pd.Series(naive_forecasting_data, index = future_dates)
print(future_forecasting)