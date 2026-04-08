import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# ================== LOAD DATA ==================
df = pd.read_csv("data/PS_20174392719_1491204439457_log.csv")

# ================== CLEAN DATA ==================
df = df.drop(['nameOrig', 'nameDest'], axis=1)

# Convert type to numeric
df['type'] = df['type'].astype('category').cat.codes

# ================== TARGET ==================
y = df['isFraud']

# Drop target columns
X = df.drop(['isFraud', 'isFlaggedFraud'], axis=1)

# ================== SPLIT ==================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ================== SCALING ==================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================== PCA ==================
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# ================== MODEL ==================
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train_pca.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train_pca, y_train, epochs=5, batch_size=64)

# ================== SAVE ==================
os.makedirs("model", exist_ok=True)

model.save("model/fraud_model.h5")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(pca, "model/pca.pkl")

print("✅ REAL MODEL TRAINED & SAVED")