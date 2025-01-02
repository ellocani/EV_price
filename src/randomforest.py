import os  # 폴더 경로 확인 및 생성
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib  # 모델 저장 및 로드
from scipy.stats import randint

# 데이터 불러오기
file_path = 'data/processed/processed_data.csv'
model_save_path = 'models/optimized_random_forest.pkl'  # 모델 저장 경로
data = pd.read_csv(file_path)

# 모델 저장 디렉토리 생성
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# 데이터 준비
X = data.drop(columns=['가격(백만원)', 'ID'])
y = data['가격(백만원)']

# 범주형 변수 인코딩
label_encoders = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# 학습 및 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 및 하이퍼파라미터 범위 설정
rf_model = RandomForestRegressor(random_state=42)
param_dist = {
    'n_estimators': randint(100, 500),  # 트리 개수
    'max_depth': [None, 10, 20, 30, 40],  # 최대 깊이
    'min_samples_split': randint(2, 20),  # 노드 분할 최소 샘플 수
    'min_samples_leaf': randint(1, 10),  # 리프 노드 최소 샘플 수
    'max_features': ['auto', 'sqrt', 'log2'],  # 분할에 사용할 특성 수
    'bootstrap': [True, False]  # 부트스트랩 샘플링
}

# RandomizedSearchCV 설정
random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_dist,
    n_iter=50,  # 탐색 횟수
    scoring='neg_mean_absolute_error',
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# 하이퍼파라미터 탐색 수행
random_search.fit(X_train, y_train)

# 최적의 하이퍼파라미터와 교차 검증 점수 확인
print(f"Best Parameters: {random_search.best_params_}")
print(f"Best CV Score (MAE): {-random_search.best_score_}")

# 최적의 모델로 테스트 세트 평가
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MAE: {mae}")
print(f"Test R²: {r2}")

# 학습된 모델 저장
joblib.dump(best_model, model_save_path)
print(f"Optimized model saved to {model_save_path}")
