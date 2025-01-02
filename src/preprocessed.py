import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 데이터 불러오기
file_path = 'data/raw/train.csv'
data = pd.read_csv(file_path)

# 범주형 변수 인코딩
label_encoders = {}
for col in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])  # 데이터 전체에 대해 레이블 인코딩
    label_encoders[col] = le

# 데이터 분리 : 결측값이 있는 데이터와 없는 데이터
data_with_battery = data.dropna(subset=['배터리용량'])  # 배터리용량이 있는 데이터
data_without_battery = data[data['배터리용량'].isnull()].copy()  # 배터리용량이 없는 데이터

# 학습용 데이터 준비
X = data_with_battery.drop(columns=['배터리용량', 'ID', '가격(백만원)'])
y = data_with_battery['배터리용량']

# 결측값 데이터 준비
X_missing = data_without_battery.drop(columns=['배터리용량', 'ID', '가격(백만원)'])

# 랜덤 포레스트 모델 학습
battery_model = RandomForestRegressor(random_state=42, n_estimators=100)
battery_model.fit(X, y)

# 결측값 예측
data_without_battery['배터리용량'] = battery_model.predict(X_missing)

# 결측값 대체
data.loc[data['배터리용량'].isnull(), '배터리용량'] = data_without_battery['배터리용량']

# 새로운 데이터 저장
output_path = 'data/processed/processed_data.csv'
data.to_csv(output_path, index=False)

print(f"Processed data saved to {output_path}")
