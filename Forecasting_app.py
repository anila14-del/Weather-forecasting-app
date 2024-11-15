# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:23:10 2024

@author: acer
"""
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

#Load the CSV file
df = pd.read_csv(r'C:\Users\acer\Desktop\Weather Forecasting\weatherHistory.csv')

# Convert 'Formatted Date' to datetime with UTC to handle mixed time zones
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)

# Remove duplicate timestamps
df = df.sort_values('Formatted Date')  # Sort by date
df = df[~df['Formatted Date'].duplicated(keep='first')]  # Keep the first occurrence of each duplicate

# Set the date column as the index for time series analysis
df.set_index('Formatted Date', inplace=True)

# Infer frequency
freq = pd.infer_freq(df.index)
if freq:
    df = df.asfreq(freq)  # Set the inferred frequency
else:
    df = df.asfreq('D')  # Default to daily frequency if inference fails

# Fill missing values for numeric columns only
df.fillna(df.select_dtypes(include=['number']).mean(), inplace=True)

# Plot Temperature over Time
df['Temperature (C)'].plot()
plt.title('Temperature over Time')
plt.xlabel('Time')
plt.ylabel('Temperature (C)')
plt.show()

# Forecasting temperature using ARIMA
model = ARIMA(df['Temperature (C)'], order=(5, 1, 0))  # Adjust (p,d,q) as needed
model_fit = model.fit()
forecast = model_fit.forecast(steps=10)
print(f"Forecasted values:\n{forecast}")

# Calculate RMSE (Optional)
# Uncomment the following lines if you have actual observed values for the forecast period
# actual_values = [...]  # Replace with actual observed values for the forecast period
# if len(actual_values) == len(forecast):
#     mse = mean_squared_error(actual_values, forecast)
#     rmse = mse**0.5
#     print(f'RMSE: {rmse}')
# else:
#     print("Ensure actual_values and predicted_values have the same length.")






