import os  # 디렉토리 생성용 모듈
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib  # 모델 저장 및 로드

# 파일 경로 설정
test_file_path = 'data/raw/test.csv'
submission_format_path = 'data/raw/sample_submission.csv'
model_file_path = 'models/trained_random_forest.pkl'  # 모델 파일 경로
output_submission_path = 'data/submission/submission.csv'

# 저장 경로에 디렉토리가 없으면 생성
os.makedirs(os.path.dirname(output_submission_path), exist_ok=True)

# 데이터 불러오기
test_data = pd.read_csv(test_file_path)
sample_submission = pd.read_csv(submission_format_path)

# 학습된 랜덤 포레스트 모델 로드
rf_model = joblib.load(model_file_path)

# 필요한 전처리
# 범주형 변수 인코딩 (훈련 시 사용한 LabelEncoder와 동일한 매핑 사용)
label_encoders = {}
for col in test_data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    test_data[col] = le.fit_transform(test_data[col])  # 여기서는 새롭게 인코딩
    label_encoders[col] = le

# 테스트 데이터 준비 (ID는 제외)
X_test = test_data.drop(columns=['ID'])

# 학습된 모델을 사용해 예측 수행
predictions = rf_model.predict(X_test)

# sample_submission에 예측값 추가
sample_submission['가격(백만원)'] = predictions

# 예측 결과 저장
sample_submission.to_csv(output_submission_path, index=False)
print(f"Submission file saved to {output_submission_path}")
