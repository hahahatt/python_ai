import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix


df = pd.read_csv('./PBL2/pbl_5/customer_data_balanced.csv')
print(df.shape)
print(df.isnull().sum())

df = pd.get_dummies(df, columns=['ContractType'])

X = df.drop(columns=['IsChurn'])
y = df['IsChurn']

scaler = StandardScaler()


X_train, X_temp, y_train, y_temp = train_test_split(X,y,test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_train = scaler.fit_transform(X_train)
X_val = scaler.fit_transform(X_val)
X_test = scaler.fit_transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


class_weight = {0: 1.0, 1: 1.5}
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=70, batch_size=16, class_weight=class_weight)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss : {test_loss:.4f}, Test Accuracy : {test_accuracy:.4f}")


predict = model.predict(X_test)
predicted_classes = (predict>0.5).astype(int)


print(classification_report(y_test, predicted_classes))

accuracy = accuracy_score(y_test, predicted_classes)
f1 = f1_score(y_test, predicted_classes)
conf = confusion_matrix(y_test, predicted_classes)

print(f"Accuracy : {accuracy:.4f}")
print(f"F1 Score : {f1:.4f}")
print("Confusion Matrix :")
print(conf)

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))

# 1. Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('📉 Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 2. Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('📈 Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

