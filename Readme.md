# Energy Predictor

A deep learning-based energy consumption forecasting system that uses a **Bidirectional LSTM** neural network to predict household appliance energy usage one hour ahead, given 7 days (168 hours) of historical context.

## Overview

This project trains a Bidirectional LSTM model on the [Appliances Energy Prediction dataset](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction) (shifted to 2026) and serves predictions through an interactive Streamlit web dashboard.

**Key highlights:**

- 7-day sliding window (168 hours) captures weekly occupancy and usage patterns
- Bidirectional LSTM learns temporal dependencies in both directions
- Log-scaled target variable for robustness against outliers
- 42 input features including cyclical encodings, rolling statistics, lag features, and contextual flags
- Huber loss function for outlier-resistant training
- Interactive Streamlit app for real-time forecast visualization

## Project Structure

```
Energy-Predictor/
├── main.py                      # Model training pipeline
├── app.py                       # Streamlit web application
├── energydata_complete.csv      # Dataset (19,735 records, 10-min intervals, Jan–May 2026)
├── energy_predictor_lstm.h5     # Pre-trained LSTM model
├── preprocessing_scalers.pkl    # Saved MinMaxScaler objects
├── requirements.txt             # Python dependencies
└── Readme.md
```

## Dataset

The dataset contains ~4.6 months of sensor readings (January–May 2026) sampled every 10 minutes from a residential building, including:

| Category | Features |
|----------|----------|
| Target | `Appliances` — energy consumption in Wh |
| Indoor climate | Temperature (`T1`–`T9`) and humidity (`RH_1`–`RH_9`) from 9 rooms |
| Outdoor weather | Temperature, humidity, pressure, wind speed, visibility, dew point |

The 10-minute data is resampled to **hourly averages** during preprocessing, and the target is **log-transformed** (`log1p`) to reduce skew.

## Feature Engineering

The model uses **42 features** in total — 28 from the raw dataset plus 14 engineered features:

| Feature | Type | Purpose |
|---------|------|---------|
| `Hour_Sin` / `Hour_Cos` | Cyclical | 24-hour daily periodicity |
| `Day_Sin` / `Day_Cos` | Cyclical | 7-day weekly periodicity |
| `Month_Sin` / `Month_Cos` | Cyclical | 12-month seasonal periodicity |
| `Is_Weekend` | Binary flag | Weekend vs. weekday usage patterns |
| `Is_Peak` | Binary flag | Peak usage hours (7–9 AM, 5–9 PM) |
| `Rolling_Mean_6h` | Rolling stat | Short-term trend (6-hour average) |
| `Rolling_Std_6h` | Rolling stat | Short-term volatility |
| `Rolling_Mean_24h` | Rolling stat | Daily trend (24-hour average) |
| `Temp_Diff` | Derived | Indoor–outdoor temperature gap (HVAC load proxy) |
| `Lag_24h` | Lag | Energy usage at the same hour yesterday |
| `Lag_168h` | Lag | Energy usage at the same hour last week |

## Model Architecture

```
Input (168 timesteps x 42 features)
  → Bidirectional LSTM (128 units, return sequences)
  → BatchNormalization → Dropout (0.3)
  → Bidirectional LSTM (64 units)
  → BatchNormalization → Dropout (0.3)
  → Dense (64, ReLU) → Dropout (0.2)
  → Dense (1, Linear)
Output: Predicted energy (log-scaled Wh)
```

**Training config:** Adam optimizer (lr=1e-4), Huber loss, early stopping (patience=12), LR reduction on plateau, batch size 32, up to 150 epochs.

## Getting Started

### Prerequisites

- Python 3.8+

### Installation

```bash
git clone https://github.com/<your-username>/Energy-Predictor.git
cd Energy-Predictor
pip install -r requirements.txt
```

### Train the Model

```bash
python main.py
```

This will:
1. Load and preprocess the dataset
2. Engineer all 42 features (cyclical, flags, rolling stats, lags, temperature differential)
3. Train the Bidirectional LSTM
4. Evaluate on a 20% held-out test set (prints RMSE and MAE)
5. Save `energy_predictor_lstm.h5` and `preprocessing_scalers.pkl`

### Run the Web App

```bash
streamlit run app.py
```

The dashboard lets you:
- Select any timestamp from the dataset (after the initial 168-hour warm-up)
- View the predicted vs. actual energy consumption
- See outdoor conditions at the selected time
- Explore an interactive chart of the last 24 hours plus the forecast

## Tech Stack

- **Deep Learning:** TensorFlow / Keras
- **Data Processing:** Pandas, NumPy, scikit-learn
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Web App:** Streamlit

## Evaluation

The model is evaluated on the original Wh scale (after inverse log-transform) using:

- **RMSE** — 78.11 Wh
- **MAE** — 51.45 Wh

## License

This project is for educational and research purposes.
![alt text](image.png)
