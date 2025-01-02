import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
font_path = 'C:/Users/min13/AppData/Local/Microsoft/Windows/Fonts/NanumGothic.ttf'
font_prop = fm.FontProperties(fname=font_path)
plt.rc('font', family=font_prop.get_name())

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split  # train_test_split 가져오기

# 모델 및 데이터 경로 설정
model_path = 'models/trained_random_forest.pkl'
test_data_path = 'data/processed/processed_data.csv'

# 모델 로드
best_model = joblib.load(model_path)

# 데이터 불러오기
data = pd.read_csv(test_data_path)
X = data.drop(columns=['가격(백만원)', 'ID'])
y = data['가격(백만원)']

# 데이터 분리 (테스트 세트 사용)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 잔차 분석
y_pred = best_model.predict(X_test)
residuals = y_test - y_pred

# 1. 잔차 분포 시각화
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, color='skyblue', edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--')
plt.title('Residual Distribution', fontsize=16)
plt.xlabel('Residuals', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.show()

# 2. 잔차 vs 예측값
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.6, color='purple', edgecolor='black')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals vs Predicted Values', fontsize=16)
plt.xlabel('Predicted Values', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.show()

# 3. 변수 중요도
feature_importances = best_model.feature_importances_
sorted_idx = np.argsort(feature_importances)[::-1]  # 중요도가 높은 순서로 정렬
sorted_features = X.columns[sorted_idx]
sorted_importances = feature_importances[sorted_idx]

# 변수 중요도 시각화
plt.figure(figsize=(12, 8))
plt.barh(sorted_features, sorted_importances, color='skyblue', edgecolor='black')
plt.title('Feature Importances', fontsize=16)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.gca().invert_yaxis()
plt.show()
