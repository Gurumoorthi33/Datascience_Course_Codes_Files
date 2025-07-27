import numpy as np
import pandas as pd

data = {
    "my_date": pd.date_range("2024-01-01", periods = 10, freq = "D"),
    "Value": [21, 20, 22, 23, 34, 54, 65, 87, 23, 28]
}

df = pd.DataFrame(data)
df.set_index("my_date", inplace = True)

moving_average_input = df["Value"].rolling(window = 3).mean().iloc[-1]
print(moving_average_input)

future_forecast = []
window_size = 3

last_window = df["Value"].iloc[-window_size: ].tolist()
print(last_window)

for i in range(5):

    next_value = np.mean(last_window)
    future_forecast.append(next_value)

    last_window.pop(0)
    last_window.append(next_value)

print(future_forecast)

future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days = 1), periods = 5, freq = "D")
future_forecast_df = pd.DataFrame({"value": future_forecast}, index = future_dates)

print(future_forecast_df)

# Make sure your index is in datetime format
# future_forecast_df.index = pd.to_datetime(future_forecast_df.index)

# Access the row for a specific date
v = future_forecast_df.loc['2024-01-12']
print(v)






