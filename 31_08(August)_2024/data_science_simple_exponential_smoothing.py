import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

data = {
    "my_date": pd.date_range("2024-01-01", periods = 10, freq = "D"),
    "Value": [21, 20, 22, 23, 34, 54, 65, 87, 23, 28]
}

df = pd.DataFrame(data)
df.set_index("my_date", inplace = True)

ses_model = SimpleExpSmoothing(df["Value"])
ses_fit = ses_model.fit(smoothing_level = 0.5) # Alpha ---------> It gives 50% importance to last data

future_forecast = ses_fit.forecast(5) # next 5 values

future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days = 1), periods = 5, freq = "D")
future_forecast_df = pd.DataFrame({"value": future_forecast}, index = future_dates)

print(future_forecast_df)

