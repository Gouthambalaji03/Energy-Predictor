import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.losses import Huber


# --- 1. Data Loading ---
print("Loading and processing data...")
df = pd.read_csv("energydata_complete.csv", parse_dates=["date"], index_col="date")

# Log Transformation (Keep this, it works well)
df['Appliances_Log'] = np.log1p(df['Appliances'])

dataset_features = [
    'Appliances_Log', 'lights',
    'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 
    'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8', 'T9', 'RH_9',
    'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 
    'Tdewpoint', 'rv1', 'rv2'
]

df_hourly = df[dataset_features].resample("H").mean().dropna()

# Cyclical Features
df_hourly['Hour_Sin'] = np.sin(2 * np.pi * df_hourly.index.hour / 24)
df_hourly['Hour_Cos'] = np.cos(2 * np.pi * df_hourly.index.hour / 24)
df_hourly['Day_Sin']  = np.sin(2 * np.pi * df_hourly.index.dayofweek / 7)
df_hourly['Day_Cos']  = np.cos(2 * np.pi * df_hourly.index.dayofweek / 7)

all_features = dataset_features + ['Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos']
dataset = df_hourly[all_features]

# --- 2. Normalization ---
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

scaled_features = scaler_features.fit_transform(dataset)
scaler_target.fit(dataset[['Appliances_Log']])

# --- 3. Sequence Creation (The Big Change) ---
# We look back 168 hours (1 Week) to capture weekly patterns
SEQ_LENGTH = 168 

def create_sequences(data, seq_length=SEQ_LENGTH, target_idx=0):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, target_idx])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_features, seq_length=SEQ_LENGTH, target_idx=0)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# --- 4. Model Architecture (Bidirectional) ---
model = Sequential([
    # Bidirectional allows the LSTM to learn from both past and "future" context in training
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
    BatchNormalization(),
    Dropout(0.3),
    
    Bidirectional(LSTM(64, return_sequences=False)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

optimizer = Adam(learning_rate=0.0001) 
model.compile(optimizer=optimizer, loss=Huber()) 

# --- 5. Training ---
early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

print("Starting training (7-Day Context)...")
history = model.fit(
    X_train, y_train, 
    epochs=150, 
    batch_size=32, 
    validation_split=0.2, 
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# --- 6. Evaluation ---
y_pred = model.predict(X_test)
y_pred_log = scaler_target.inverse_transform(y_pred) 
y_test_log = scaler_target.inverse_transform(y_test.reshape(-1, 1))

# Convert back from Log scale
y_pred_actual = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test_log)

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
mae = mean_absolute_error(y_test_actual, y_pred_actual)

print(f"\nFinal Score (7-Day Context + Bidirectional):")
print(f"RMSE: {rmse:.4f} Wh")
print(f"MAE:  {mae:.4f} Wh")

# --- 7. Save ---
model.save("energy_predictor_lstm.h5")
scalers = {"scaler_features": scaler_features, "scaler_target": scaler_target}
with open("preprocessing_scalers.pkl", "wb") as f:
    pickle.dump(scalers, f)
print("Saved artifacts.")