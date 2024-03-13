#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load and preprocess the dataset
file_path = 'household_power_consumption.csv'
data = pd.read_csv(file_path, sep=',', low_memory=False)
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], dayfirst=True)
data.drop(['Date', 'Time'], axis=1, inplace=True)
numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data.dropna(inplace=True)
data.sort_values(by='Datetime', inplace=True)

# Resample the data to daily granularity
daily_data = data.resample('D', on='Datetime').mean()
daily_data.dropna(inplace=True)


# In[2]:


# Create lagged features for 1, 7, and 30 days
for lag in [1, 7, 30]:
    daily_data[f'lag_{lag}'] = daily_data['Global_active_power'].shift(lag)

# Create rolling window features (mean and std) for 7 and 30 days
for window in [7, 30]:
    daily_data[f'rolling_mean_{window}'] = daily_data['Global_active_power'].rolling(window=window).mean()
    daily_data[f'rolling_std_{window}'] = daily_data['Global_active_power'].rolling(window=window).std()

daily_data.dropna(inplace=True)

# Prepare features and target variable
X = daily_data.drop(['Global_active_power'], axis=1)
y = daily_data['Global_active_power']

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]


# In[3]:


# Train a Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Calculate error metrics
mae_rf = mean_absolute_error(y_test, y_pred)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred))
mape_rf = np.mean(np.abs(y_pred - y_test) / np.abs(y_test)) * 100

print(f"MAE: {mae_rf}, RMSE: {rmse_rf}, MAPE: {mape_rf}")


# In[4]:


# Plot actual vs predicted values
forecast_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=y_test.index)
plt.figure(figsize=(14, 7))
plt.plot(forecast_df['Actual'], label='Actual Power Consumption', color='blue', marker='o')
plt.plot(forecast_df['Predicted'], label='Predicted Power Consumption', color='red', linestyle='--', marker='x')
plt.title('Daily Global Active Power Consumption: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Global Active Power (kilowatts)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[ ]:





# Based on the development and evaluation of the predictive model for forecasting future power consumption, here are several recommendations and insights to enhance its utility and accuracy further:
# 
# ### Recommendations
# 
# 1. **Incorporate External Data**: Including additional relevant factors such as weather conditions (temperature, humidity, and precipitation), public holidays, and economic indicators could improve the model's predictive accuracy. Weather, in particular, has a significant impact on power usage patterns.
# 
# 2. **Advanced Feature Engineering**: Further develop the model's input features by exploring more sophisticated feature engineering techniques. This could include interactions between existing features, more granular temporal features (e.g., hour of the day for hourly models, day of the week for daily models), and polynomial features that could capture non-linear relationships.
# 
# 3. **Model Experimentation**: Experiment with various other models and techniques. Deep learning models like LSTM (Long Short Term Memory networks) are particularly well-suited for time series forecasting because they can capture long-term dependencies in sequence data. Hybrid models that combine traditional statistical methods with machine learning could also offer improved accuracy.
# 
# 4. **Hyperparameter Tuning**: For the Random Forest model and any other machine learning models used, perform thorough hyperparameter tuning. This process can often significantly improve model performance by finding the optimal configuration for the model's parameters.
# 
# 5. **Cross-Validation**: Implement time series cross-validation to more robustly evaluate the model's performance. This technique respects the chronological order of observations and provides a more reliable estimate of the model's predictive capability on unseen data.
# 
# 6. **Model Ensembling**: Consider ensembling multiple models to improve predictions. Ensembling methods, such as stacking or blending, combine the forecasts from several models, which can often lead to more accurate and robust predictions than any single model.
# 
# ### Insights
# 
# - **Seasonality and Trends**: The power consumption data exhibits clear seasonal patterns as well as longer-term trends. Understanding these can help in selecting appropriate models and features. For example, incorporating features that capture weekly and yearly seasonality can significantly improve forecasting accuracy.
# 
# - **Peak Demand Forecasting**: Identifying peak demand periods is crucial for ensuring grid stability and efficient energy distribution. The model's ability to accurately forecast these periods can be enhanced by focusing on extreme values analysis and potentially incorporating alert systems for unusual consumption patterns.
# 
# - **Energy Efficiency Programs**: Insights from the model can help in designing targeted energy efficiency programs. By understanding when and where energy usage peaks, utilities can devise strategies to flatten peak demand curves, such as through dynamic pricing or encouraging the use of energy-efficient appliances.
# 
# - **Renewable Energy Integration**: Forecasting power consumption accurately is vital for integrating renewable energy sources into the power grid. Predictive models can help in balancing supply and demand by anticipating periods of high consumption that may require additional energy from renewable sources.
# 
# Implementing these recommendations and leveraging the insights gained from the predictive modeling process can significantly enhance the effectiveness of forecasting models. This, in turn, supports more informed decision-making for energy providers, policymakers, and consumers, leading to more efficient energy use, cost savings, and a more sustainable energy future.

# In[ ]:





# In[ ]:




