import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def convert_float_to_date(x):
    year = int(x)
    quarter = int((x - year) * 4)
    month = quarter * 3 + 1
    return pd.Timestamp(f"{year}-{month:02d}-01")

url = "https://raw.githubusercontent.com/antoinecarme/TimeSeriesData/master/fpp2/uschange.csv"
df = pd.read_csv(url)

df['ds'] = df['Index'].apply(convert_float_to_date)
df['y'] = df['Consumption']
data = df[['ds', 'y']]

train_size = int(len(data) * 0.8)
train = data.iloc[:train_size]
test = data.iloc[train_size:]

model = Prophet()
model.fit(train)

future = model.make_future_dataframe(periods=len(test), freq='QS')
forecast = model.predict(future)

predictions = forecast.iloc[train_size:]['yhat']
actuals = test['y'].reset_index(drop=True)

r2 = r2_score(actuals, predictions)

print(f"R-squared Score: {r2}")

plt.figure(figsize=(12, 6))
plt.plot(train['ds'], train['y'], label='Train')
plt.plot(test['ds'], test['y'], label='Test (Actual)')
plt.plot(test['ds'], predictions.values, label='Prediction', linestyle='--')
plt.title(f'Consumption Forecast (R^2: {r2:.4f})')
plt.legend()
plt.show()