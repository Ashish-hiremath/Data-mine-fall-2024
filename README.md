# Data-mine-fall-2024
dataset link
crime :https://hub.mph.in.gov/dataset/indiana-arrest-data
![image](https://github.com/user-attachments/assets/b15c0ea9-2618-4ce8-b0a3-929d048851b9)

ND:https://www.ngdc.noaa.gov/hazard/hazards.shtml
![image](https://github.com/user-attachments/assets/c7cce776-778b-4193-bc3a-cef0664a74a6)

# model 1:
Used prophet lib for forcasting crime counts  
Tuned Prophet MAE: 5729.2685
Tuned Prophet RMSE: 6708.8788
# necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Focusing on 'ARREST_YEAR' and 'ARREST_MONTH'
important_vars = ['ARREST_YEAR', 'ARREST_MONTH']

# a new DataFrame to store the necessary data
df_crime = df[important_vars].dropna().copy()

# Creating 'DATE' for time series and 'CRIME_COUNT'
df_crime['DATE'] = pd.to_datetime(df_crime['ARREST_YEAR'].astype(str) + '-' +
                                  df_crime['ARREST_MONTH'].astype(str).str.zfill(2) + '-01')

# monthly crime counts for time series
crime_counts = df_crime.groupby('DATE').size().reset_index(name='CRIME_COUNT')

# Prophet requires 'ds' for datetime and 'y' for the target var
crime_counts = crime_counts.rename(columns={'DATE': 'ds', 'CRIME_COUNT': 'y'})

# Split data into train/test (80% train, 20% test)
train, test = train_test_split(crime_counts, test_size=0.2, shuffle=False)

# --------------------------------------------
# Prophet Model: Tuning and Forecasting
# --------------------------------------------
# Initialize Prophet with tuning parameters
m_tuned = Prophet(
    yearly_seasonality=True,   # Enables yearly seasonality
    weekly_seasonality=False,  # Disable weekly seasonality
    daily_seasonality=False,   # Disable daily seasonality
    seasonality_mode='multiplicative',  # which allows for changes in seasonality that scale with the level of the trend.
    changepoint_prior_scale=0.05  # A lower value  makes the model less flexible, potentially capturing gradual trends better
)

# Fitting Prophet model to training data
m_tuned.fit(train)

# Creating future dataframe to predict for the test period
future = m_tuned.make_future_dataframe(periods=len(test), freq='M')
forecast = m_tuned.predict(future)

# Plotting forecasted data
plt.figure(figsize=(12, 6))
m_tuned.plot(forecast)
plt.title('Tuned Prophet Forecast of Crime Rates')
plt.show()

# Plot the components of the forecast (trend, yearly seasonality)
plt.figure(figsize=(12, 8))
m_tuned.plot_components(forecast)
plt.show()

# --------------------------------------------
# Evaluation Metrics
# --------------------------------------------
# Evaluate performance on the test set using MAE and RMSE
y_true = test['y']
y_pred = forecast['yhat'][-len(test):]  # Predicted values for test period

# Calculating MAE and RMSE for tuned Prophet model
mae_prophet = mean_absolute_error(y_true, y_pred)
rmse_prophet = np.sqrt(mean_squared_error(y_true, y_pred))

# Print results
print(f"Tuned Prophet MAE: {mae_prophet:.4f}")
print(f"Tuned Prophet RMSE: {rmse_prophet:.4f}")

# model 2:
Used ElasticNet with Bayesian optimization for ElasticNet and holidays lib for holidays season
ElasticNet with Seasonality - MAE: 1825.98, RMSE: 2436.17, R²: 0.90, MAPE: 2.76%
Best ElasticNet Parameters: OrderedDict([('alpha', 0.1), ('l1_ratio', 0.9)])
# Import  libraries
import pandas as pd
import numpy as np
import holidays  # For holiday info
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from sklearn.utils import resample  # For bootstrapping
import matplotlib.pyplot as plt

#  preprocessed DataFrame
important_vars = ['ARREST_YEAR', 'ARREST_MONTH','OFFENSE_COUNTY']

# new DataFrame for focused variables and drop missing data
df_crime = df[important_vars].dropna().copy()

#  'DATE' for time series and 'CRIME_COUNT'
df_crime['DATE'] = pd.to_datetime(df_crime['ARREST_YEAR'].astype(str) + '-' +
                                  df_crime['ARREST_MONTH'].astype(str).str.zfill(2) + '-01')

# Group data by date to get monthly crime counts
crime_counts = df_crime.groupby('DATE').size().reset_index(name='CRIME_COUNT')
crime_counts = crime_counts.rename(columns={'DATE': 'ds', 'CRIME_COUNT': 'y'})

# Add day of the week
crime_counts['day_of_week'] = crime_counts['ds'].dt.dayofweek  # Monday=0, Sunday=6

# Add holiday features
us_holidays = holidays.US()
crime_counts['is_holiday'] = crime_counts['ds'].isin(us_holidays).astype(int)

# Add seasonal features (sinusoidal transformation for seasonality)
crime_counts['month_sin'] = np.sin(2 * np.pi * crime_counts['ds'].dt.month / 12)
crime_counts['month_cos'] = np.cos(2 * np.pi * crime_counts['ds'].dt.month / 12)

# Create lag and rolling features
crime_counts['lag1'] = crime_counts['y'].shift(1)
crime_counts['lag2'] = crime_counts['y'].shift(2)
crime_counts['lag3'] = crime_counts['y'].shift(3)
crime_counts['rolling_mean_3'] = crime_counts['y'].rolling(window=3).mean()
crime_counts['rolling_mean_6'] = crime_counts['y'].rolling(window=6).mean()

# Incorporating recent trends using Exponentially Weighted Moving Average (EWMA)
crime_counts['ewm_12'] = crime_counts['y'].ewm(span=12).mean()

# Drop NaN values created by lagging, rolling features, and EWMA
crime_counts.dropna(inplace=True)

# Split the data into train and test sets (80% train, 20% test)
train, test = train_test_split(crime_counts, test_size=0.2, shuffle=False)

# Standardizing features
scaler = StandardScaler()

# Prepare training features
X_train_raw = pd.DataFrame({
    'year': train['ds'].dt.year,
    'month': train['ds'].dt.month,
    'day_of_week': train['day_of_week'],
    'is_holiday': train['is_holiday'],
    'month_sin': train['month_sin'],  # Adding seasonal feature
    'month_cos': train['month_cos'],  # Adding seasonal feature
    'lag1': train['lag1'],
    'lag2': train['lag2'],
    'lag3': train['lag3'],
    'rolling_mean_3': train['rolling_mean_3'],
    'rolling_mean_6': train['rolling_mean_6'],
    'ewm_12': train['ewm_12']  # EWMA feature
})

# Scaling features
X_train_scaled = scaler.fit_transform(X_train_raw)
y_train = train['y']

# Prepare test features
X_test_raw = pd.DataFrame({
    'year': test['ds'].dt.year,
    'month': test['ds'].dt.month,
    'day_of_week': test['day_of_week'],
    'is_holiday': test['is_holiday'],
    'month_sin': test['month_sin'],  # Adding seasonal feature
    'month_cos': test['month_cos'],  # Adding seasonal feature
    'lag1': test['lag1'],
    'lag2': test['lag2'],
    'lag3': test['lag3'],
    'rolling_mean_3': test['rolling_mean_3'],
    'rolling_mean_6': test['rolling_mean_6'],
    'ewm_12': test['ewm_12']  # EWMA feature
})

# Scale the test features
X_test_scaled = scaler.transform(X_test_raw)
y_test = test['y']

# Bayesian optimization for ElasticNet
param_dist_en = {
    'alpha': (0.1, 10.0, 'log-uniform'),  #alpha controls regularization strength.
    'l1_ratio': (0.1, 0.9, 'uniform')  # ElasticNet ratio   #l1_ratio balances between Lasso (L1) and Ridge (L2) regularization.
}

# TimeSeriesSplit cross-validator (no data leakage)
tscv = TimeSeriesSplit(n_splits=10)

# Bayesian search for hyperparameter tuning
bayes_search_en = BayesSearchCV(
    ElasticNet(max_iter=10000),  # Increased iterations
    search_spaces=param_dist_en,
    scoring='neg_mean_squared_error',
    cv=tscv,
    n_iter=50,  # Increased iterations for better search
    random_state=42,
    n_jobs=-1
)

# Fit the model using the Bayesian search
bayes_search_en.fit(X_train_scaled, y_train)

# Retrieve the best ElasticNet model
best_en_model = bayes_search_en.best_estimator_

# Predict on the test set using the tuned ElasticNet model
y_pred_en = best_en_model.predict(X_test_scaled)

# Bootstrapping to estimate uncertainty
n_bootstraps = 1000
y_boot_preds = []

for i in range(n_bootstraps):
    # Bootstrap resample
    X_resampled, y_resampled = resample(X_train_scaled, y_train, random_state=i)
    best_en_model.fit(X_resampled, y_resampled)
    y_boot_preds.append(best_en_model.predict(X_test_scaled))

# Convert to DataFrame for easier analysis
y_boot_preds_df = pd.DataFrame(y_boot_preds)

# Calculate mean prediction and confidence intervals
y_pred_mean = y_boot_preds_df.mean(axis=0)
y_pred_lower = y_boot_preds_df.quantile(0.025, axis=0)
y_pred_upper = y_boot_preds_df.quantile(0.975, axis=0)

# Plot actual vs predicted with uncertainty
plt.figure(figsize=(12, 6))
plt.plot(test['ds'], y_test, label='Actual', color='blue')
plt.plot(test['ds'], y_pred_mean, label='Predicted', color='orange', linestyle='--')
plt.fill_between(test['ds'], y_pred_lower, y_pred_upper, color='orange', alpha=0.3, label='95% Confidence Interval')
plt.title('Crime Count Predictions with Seasonality and Uncertainty')
plt.xlabel('Date')
plt.ylabel('Crime Count')
plt.legend()
plt.show()

# Fix for MAPE calculation to avoid division by zero
def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.where(y_true == 0, 1e-10, y_true)  # Replace zero values
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Evaluation: MAE, RMSE, R², and MAPE
mae_en = mean_absolute_error(y_test, y_pred_mean)  # Calculate MAE
mape_en = mean_absolute_percentage_error(y_test, y_pred_mean)  # Calculate MAPE
rmse_en = np.sqrt(mean_squared_error(y_test, y_pred_mean))  # Calculate RMSE
r2_en = r2_score(y_test, y_pred_mean)  # Calculate R²

# Print evaluation metrics
print(f"ElasticNet with Seasonality - MAE: {mae_en:.2f}, RMSE: {rmse_en:.2f}, R²: {r2_en:.2f}, MAPE: {mape_en:.2f}%")
print(f"Best ElasticNet Parameters: {bayes_search_en.best_params_}")



# forcasting crime counts 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Forecast Future Crime Counts - Iterative Approach with Alignment
forecast_horizon = 60  # Generate future dates for forecasting (5 years)
last_date = crime_counts['ds'].max()  # Last date from historical data
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_horizon, freq='MS')

# Start by using the last few known values for lag calculation
last_lags = {
    'lag1': crime_counts['y'].iloc[-1],
    'lag2': crime_counts['y'].iloc[-2],
    'lag3': crime_counts['y'].iloc[-3],
    'rolling_mean_3': crime_counts['y'].rolling(window=3).mean().iloc[-1],
    'rolling_mean_6': crime_counts['y'].rolling(window=6).mean().iloc[-1],
    'ewm_12': crime_counts['y'].ewm(span=12).mean().iloc[-1]
}

# Create a list for future predictions
future_predictions = []

# Iterative forecast loop for each future date
for date in future_dates:
    # Prepare features for the current date
    features = {
        'year': date.year,
        'month': date.month,
        'day_of_week': date.dayofweek,
        'is_holiday': date in us_holidays,
        'month_sin': np.sin(2 * np.pi * date.month / 12),
        'month_cos': np.cos(2 * np.pi * date.month / 12),
        'lag1': last_lags['lag1'],
        'lag2': last_lags['lag2'],
        'lag3': last_lags['lag3'],
        'rolling_mean_3': last_lags['rolling_mean_3'],
        'rolling_mean_6': last_lags['rolling_mean_6'],
        'ewm_12': last_lags['ewm_12']
    }

    # Convert features to a DataFrame and scale them
    features_df = pd.DataFrame([features])
    features_scaled = scaler.transform(features_df)  # Scale features

    # Predict crime count for this date
    predicted_crime_count = best_en_model.predict(features_scaled)[0]

    # Round up the predicted crime count and convert to an integer
    rounded_predicted_crime_count = int(np.ceil(predicted_crime_count))

    # Append the rounded prediction to the list of future predictions
    future_predictions.append(rounded_predicted_crime_count)

    # Update lag features iteratively
    last_lags['lag3'] = last_lags['lag2']
    last_lags['lag2'] = last_lags['lag1']
    last_lags['lag1'] = rounded_predicted_crime_count
    last_lags['rolling_mean_3'] = np.mean([last_lags['lag1'], last_lags['lag2'], last_lags['lag3']])
    last_lags['rolling_mean_6'] = np.mean([last_lags['lag1'], last_lags['lag2'], last_lags['lag3'],
                                           crime_counts['y'].iloc[-4], crime_counts['y'].iloc[-5], crime_counts['y'].iloc[-6]])
    last_lags['ewm_12'] = (0.92 * rounded_predicted_crime_count) + (0.08 * last_lags['ewm_12'])  # EWM update

# Create a DataFrame for the forecasted values
forecast_df = pd.DataFrame({'ds': future_dates, 'y': future_predictions})

# Add the last point of historical data to the beginning of the forecast
last_historical_point = pd.DataFrame({'ds': [crime_counts['ds'].iloc[-1]], 'y': [crime_counts['y'].iloc[-1]]})
forecast_df = pd.concat([last_historical_point, forecast_df]).reset_index(drop=True)

# Plot the historical and forecasted crime counts
plt.figure(figsize=(12, 6))

# Plot historical crime data in blue
plt.plot(crime_counts['ds'], crime_counts['y'], label='Historical Data', color='blue')

# Plot forecasted crime data in red with a dashed line
plt.plot(forecast_df['ds'], forecast_df['y'], label='Forecast', color='red', linestyle='--')

# Add title and labels
plt.title('Crime Count: Historical Data and Forecast')
plt.xlabel('Date')
plt.ylabel('Crime Count')

# Add legend to differentiate between historical and forecasted data
plt.legend()

# Ensure x-axis limits cover both historical and forecast data
plt.xlim(crime_counts['ds'].min(), forecast_df['ds'].max())

# Tight layout for better spacing
plt.tight_layout()

 # Show the plot
plt.show()

# Display the forecasted crime counts
print(forecast_df)
