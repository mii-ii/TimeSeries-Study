import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 1. 데이터 로드 및 날짜 변환 함수
def convert_float_to_date(x):
    year = int(x)
    quarter = int((x - year) * 4)
    month = quarter * 3 + 1
    return pd.Timestamp(f"{year}-{month:02d}-01")

url = "https://raw.githubusercontent.com/antoinecarme/TimeSeriesData/master/fpp2/uschange.csv"
df = pd.read_csv(url)

# 2. 데이터 준비 (변수 추가!)
df['ds'] = df['Index'].apply(convert_float_to_date)
df['y'] = df['Consumption']
# 추가 변수(Regressor) 준비
df['Income'] = df['Income']
df['Unemployment'] = df['Unemployment']
df['Savings'] = df['Savings'] # 저축도 넣어볼게요, 효과가 좋습니다.

# 학습에 필요한 컬럼만 선택
data = df[['ds', 'y', 'Income', 'Unemployment', 'Savings']]

# 3. Train/Test 분리 (80% : 20%)
train_size = int(len(data) * 0.8)
train = data.iloc[:train_size]
test = data.iloc[train_size:]

# 4. 모델 생성 및 추가 변수 등록 (핵심!)
model = Prophet()
model.add_regressor('Income')
model.add_regressor('Unemployment')
model.add_regressor('Savings')

model.fit(train)

# 5. 예측을 위한 데이터 프레임 생성
# 테스트 기간의 'Income', 'Unemployment', 'Savings' 값을 미래 데이터프레임에 넣어줘야 함
future = model.make_future_dataframe(periods=len(test), freq='QS')
# 미래(테스트) 기간의 외부 변수 값들을 원본 데이터에서 가져와 붙이기
future['Income'] = df['Income'].values
future['Unemployment'] = df['Unemployment'].values
future['Savings'] = df['Savings'].values

forecast = model.predict(future)

# 6. 평가 및 시각화
predictions = forecast.iloc[train_size:]['yhat']
actuals = test['y'].reset_index(drop=True)

r2 = r2_score(actuals, predictions)

print(f"Improved R-squared Score: {r2}")

plt.figure(figsize=(12, 6))
plt.plot(train['ds'], train['y'], label='Train')
plt.plot(test['ds'], test['y'], label='Test (Actual)')
plt.plot(test['ds'], predictions.values, label='Prediction (Multivariate)', linestyle='--', color='red')
plt.title(f'Consumption Forecast with Regressors (R^2: {r2:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()