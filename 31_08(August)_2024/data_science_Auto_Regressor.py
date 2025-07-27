import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

data = {
    "my_date": pd.date_range("2024-01-01", periods = 10, freq = "D"),
    "Value": [21, 20, 22, 23, 34, 54, 65, 87, 23, 28]
}

df = pd.DataFrame(data)
df.set_index("my_date", inplace = True)

ar_model = AutoReg(df["Value"], lags = 3)
ar_fit = ar_model.fit()

future_forecast = ar_fit.predict(start = len(df), end = len(df)+4)

future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days = 1), periods = 5, freq = "D")
future_forecast_df = pd.DataFrame({"value": future_forecast}, index = future_dates)

print(future_forecast_df)