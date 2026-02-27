import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Bidirectional,
    Input, Attention, Concatenate, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber


# --- 1. Data Loading ---
print("Loading and processing data...")
df = pd.read_csv("energydata_complete.csv", parse_dates=["date"], index_col="date")

# Log Transformation
df['Appliances_Log'] = np.log1p(df['Appliances'])

dataset_features = [
    'Appliances_Log', 'lights',
    'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4',
    'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8', 'T9', 'RH_9',
    'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility',
    'Tdewpoint', 'rv1', 'rv2'
]

df_hourly = df[dataset_features].resample("h").mean().dropna()

# Cyclical Features
df_hourly['Hour_Sin'] = np.sin(2 * np.pi * df_hourly.index.hour / 24)
df_hourly['Hour_Cos'] = np.cos(2 * np.pi * df_hourly.index.hour / 24)
df_hourly['Day_Sin']  = np.sin(2 * np.pi * df_hourly.index.dayofweek / 7)
df_hourly['Day_Cos']  = np.cos(2 * np.pi * df_hourly.index.dayofweek / 7)

# Month cyclical (captures seasonal patterns)
df_hourly['Month_Sin'] = np.sin(2 * np.pi * df_hourly.index.month / 12)
df_hourly['Month_Cos'] = np.cos(2 * np.pi * df_hourly.index.month / 12)

# Weekend flag
df_hourly['Is_Weekend'] = (df_hourly.index.dayofweek >= 5).astype(float)

# Peak hours flag (7-9 AM and 17-21 PM)
hour = df_hourly.index.hour
df_hourly['Is_Peak'] = ((hour >= 7) & (hour <= 9) | (hour >= 17) & (hour <= 21)).astype(float)

# Rolling statistics
df_hourly['Rolling_Mean_6h'] = df_hourly['Appliances_Log'].rolling(6, min_periods=1).mean()
df_hourly['Rolling_Std_6h'] = df_hourly['Appliances_Log'].rolling(6, min_periods=1).std().fillna(0)
df_hourly['Rolling_Mean_24h'] = df_hourly['Appliances_Log'].rolling(24, min_periods=1).mean()

# Temperature differential (indoor vs outdoor)
df_hourly['Temp_Diff'] = df_hourly['T1'] - df_hourly['T_out']

# Lag features
df_hourly['Lag_24h'] = df_hourly['Appliances_Log'].shift(24).bfill()
df_hourly['Lag_168h'] = df_hourly['Appliances_Log'].shift(168).bfill()

df_hourly = df_hourly.dropna()

all_features = dataset_features + [
    'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos',
    'Month_Sin', 'Month_Cos',
    'Is_Weekend', 'Is_Peak',
    'Rolling_Mean_6h', 'Rolling_Std_6h', 'Rolling_Mean_24h',
    'Temp_Diff',
    'Lag_24h', 'Lag_168h'
]
dataset = df_hourly[all_features]

# --- 2. Normalization ---
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

scaled_features = scaler_features.fit_transform(dataset)
scaler_target.fit(dataset[['Appliances_Log']])

# --- 3. Sequence Creation ---
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

# --- 4. Model Architecture (Bidirectional LSTM + Attention) ---
n_features = X_train.shape[2]

inputs = Input(shape=(SEQ_LENGTH, n_features))

# First Bidirectional LSTM
x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

# Self-attention on sequence
attn_output = Attention()([x, x])

# Second Bidirectional LSTM processes attention-enriched sequence
x = Bidirectional(LSTM(64, return_sequences=False))(attn_output)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

# Dense layers
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(1)(x)

model = Model(inputs=inputs, outputs=outputs)

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss=Huber())

model.summary()

# --- 5. Training ---
early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

print("\nStarting training (7-Day Context + Attention)...")
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

y_pred_actual = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test_log)

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
mae = mean_absolute_error(y_test_actual, y_pred_actual)
r2 = r2_score(y_test_actual, y_pred_actual)
mape = np.mean(np.abs((y_test_actual - y_pred_actual) / np.clip(y_test_actual, 1, None))) * 100

print(f"\nFinal Score (7-Day Context + Bidirectional + Attention):")
print(f"RMSE: {rmse:.4f} Wh")
print(f"MAE:  {mae:.4f} Wh")
print(f"R²:   {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

# --- 7. Save Artifacts ---
model.save("energy_predictor_lstm.h5")

scalers = {"scaler_features": scaler_features, "scaler_target": scaler_target}
with open("preprocessing_scalers.pkl", "wb") as f:
    pickle.dump(scalers, f)

# Save training history
with open("training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

# Save test predictions for Model Performance page
test_predictions = {
    "y_test": y_test_actual.flatten(),
    "y_pred": y_pred_actual.flatten(),
    "rmse": rmse,
    "mae": mae,
    "r2": r2,
    "mape": mape
}
with open("test_predictions.pkl", "wb") as f:
    pickle.dump(test_predictions, f)

print("Saved all artifacts.")
