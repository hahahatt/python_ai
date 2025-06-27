import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

# 1. 데이터 로드
df = pd.read_csv('./PBL2/pbl_5/customer_data_balanced.csv')

# 2. 특성과 라벨 분리
X = df.drop('IsChurn', axis=1)
y = df['IsChurn']

# 3. 전처리: One-Hot Encoding & Standard Scaling
categorical = ['ContractType']
numerical = ['Age', 'Tenure', 'MonthlySpending_KRW', 'CustomerServiceCalls']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical),
    ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical)
])

X_processed = preprocessor.fit_transform(X)

# 4. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

# 5. 클래스 가중치 설정
class_weight = {0: 1.0, 1: 2.0}  # 불균형 고려

# 6. 모델 구성
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

# 7. 모델 학습
history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.2, class_weight=class_weight, verbose=1)

# 8. 예측 및 평가
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype('int32')

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\n✅ Accuracy: {acc:.4f}")
print(f"✅ F1 Score: {f1:.4f}")
print("\n🔍 Confusion Matrix:")
print(cm)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# 9. 학습 과정 시각화 (선택)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()