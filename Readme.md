# Energy Predictor

A deep learning-based energy consumption forecasting system that uses a **Bidirectional LSTM with Self-Attention** to predict household appliance energy usage, featuring multi-step forecasting, confidence intervals, anomaly detection, and an interactive multi-page Streamlit dashboard.

**Live Demo:** [energy-predictor-irx5dsbkakuufwd5nojgjb.streamlit.app](https://energy-predictor-irx5dsbkakuufwd5nojgjb.streamlit.app/)

## Overview

This project trains a Bidirectional LSTM model with a self-attention mechanism on the [Appliances Energy Prediction dataset](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction) (shifted to 2026) and serves predictions through an interactive 4-tab Streamlit dashboard.

**Key highlights:**

- 7-day sliding window (168 hours) captures weekly occupancy and usage patterns
- Bidirectional LSTM with self-attention learns complex temporal dependencies
- Monte Carlo Dropout for prediction confidence intervals
- Multi-step forecasting (1h, 6h, 12h, 24h ahead)
- Anomaly detection with z-score thresholding
- Log-scaled target variable for robustness against outliers
- 42 input features including cyclical encodings, rolling statistics, lag features, and contextual flags
- Huber loss function for outlier-resistant training
- 4-tab interactive Streamlit dashboard with dark theme

## Project Structure

```
Energy-Predictor/
├── main.py                      # Model training pipeline (Attention + Bi-LSTM)
├── app.py                       # Streamlit multi-page dashboard (4 tabs)
├── energydata_complete.csv      # Dataset (19,735 records, 10-min intervals, Jan-May 2026)
├── energy_predictor_lstm.h5     # Pre-trained LSTM model with attention
├── preprocessing_scalers.pkl    # Saved MinMaxScaler objects
├── training_history.pkl         # Training/validation loss history
├── test_predictions.pkl         # Test set predictions and metrics
├── requirements.txt             # Python dependencies
├── runtime.txt                  # Python version for Streamlit Cloud (3.11)
├── .streamlit/config.toml       # Streamlit server and dark theme config
└── Readme.md
```

## Dataset

The dataset contains ~4.6 months of sensor readings (January-May 2026) sampled every 10 minutes from a residential building, including:

| Category | Features |
|----------|----------|
| Target | `Appliances` - energy consumption in Wh |
| Indoor climate | Temperature (`T1`-`T9`) and humidity (`RH_1`-`RH_9`) from 9 rooms |
| Outdoor weather | Temperature, humidity, pressure, wind speed, visibility, dew point |

The 10-minute data is resampled to **hourly averages** during preprocessing, and the target is **log-transformed** (`log1p`) to reduce skew.

## Feature Engineering

The model uses **42 features** in total - 28 from the raw dataset plus 14 engineered features:

| Feature | Type | Purpose |
|---------|------|---------|
| `Hour_Sin` / `Hour_Cos` | Cyclical | 24-hour daily periodicity |
| `Day_Sin` / `Day_Cos` | Cyclical | 7-day weekly periodicity |
| `Month_Sin` / `Month_Cos` | Cyclical | 12-month seasonal periodicity |
| `Is_Weekend` | Binary flag | Weekend vs. weekday usage patterns |
| `Is_Peak` | Binary flag | Peak usage hours (7-9 AM, 5-9 PM) |
| `Rolling_Mean_6h` | Rolling stat | Short-term trend (6-hour average) |
| `Rolling_Std_6h` | Rolling stat | Short-term volatility |
| `Rolling_Mean_24h` | Rolling stat | Daily trend (24-hour average) |
| `Temp_Diff` | Derived | Indoor-outdoor temperature gap (HVAC load proxy) |
| `Lag_24h` | Lag | Energy usage at the same hour yesterday |
| `Lag_168h` | Lag | Energy usage at the same hour last week |

## Model Architecture

```
Input (168 timesteps x 42 features)
  -> Bidirectional LSTM (128 units, return sequences)
  -> BatchNormalization -> Dropout (0.3)
  -> Self-Attention Layer
  -> Bidirectional LSTM (64 units)
  -> BatchNormalization -> Dropout (0.3)
  -> Dense (64, ReLU) -> Dropout (0.2)
  -> Dense (1, Linear)
Output: Predicted energy (log-scaled Wh)
```

The self-attention layer (`tf.keras.layers.Attention`) allows the model to weigh the importance of different time steps in the 168-hour input window, focusing on the most relevant historical patterns for each prediction.

**Training config:** Adam optimizer (lr=1e-4), Huber loss, early stopping (patience=12), LR reduction on plateau, batch size 32, up to 150 epochs.

## Dashboard

The Streamlit app features a **4-tab interface** with a dark theme:

### Tab 1: Forecasting
- **Single-step prediction** - Select any timestamp and get a 1-hour-ahead forecast
- **Multi-step forecasting** - Predict 1h, 6h, 12h, or 24h ahead with interactive charts
- **Confidence intervals** - Monte Carlo Dropout (10 forward passes) provides uncertainty bands
- **Anomaly detection** - Flags predictions where actual values deviate >2 standard deviations from the rolling mean
- **Energy cost estimation** - Configurable electricity rate ($/kWh) in sidebar

### Tab 2: Batch Prediction
- Select a date range and generate predictions for every hour
- Results table with predicted vs. actual values, error, and anomaly flags
- Summary statistics (average error, max error, anomaly count)
- **CSV export** via download button

### Tab 3: Data Explorer
- Feature statistics table (min, max, mean, std)
- Energy consumption distribution (histogram + box plot)
- Seasonal pattern analysis (hourly, daily, weekly averages)
- Interactive correlation heatmap of all features

### Tab 4: Model Performance
- Training/validation loss curves
- Metrics dashboard: RMSE, MAE, R², MAPE
- Actual vs. Predicted scatter plot
- Error distribution histogram
- Residual plot over time

## Getting Started

### Prerequisites

- Python 3.11+

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
3. Train the Bidirectional LSTM with self-attention
4. Evaluate on a 20% held-out test set (prints RMSE, MAE, R², MAPE)
5. Save artifacts: `energy_predictor_lstm.h5`, `preprocessing_scalers.pkl`, `training_history.pkl`, `test_predictions.pkl`

### Run the Web App Locally

```bash
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`.

## Deployment

The app is deployed on **Streamlit Community Cloud**.

### How to Deploy

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/<your-username>/Energy-Predictor.git
   git branch -M main
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
   - Click **"New app"**
   - Select your repository, branch (`main`), and main file (`app.py`)
   - Click **"Deploy"**

3. **Key deployment files:**
   - `runtime.txt` - sets Python 3.11 (required for TensorFlow compatibility)
   - `requirements.txt` - pinned minimum versions to avoid dependency conflicts
   - `.streamlit/config.toml` - headless server config and dark theme for cloud environment

The app will be live at `https://<your-app-name>.streamlit.app`

## Tech Stack

- **Deep Learning:** TensorFlow / Keras (Bidirectional LSTM + Attention)
- **Data Processing:** Pandas, NumPy, scikit-learn
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Web App:** Streamlit
- **Deployment:** Streamlit Community Cloud

## Evaluation

The model is evaluated on the original Wh scale (after inverse log-transform):

| Metric | Value |
|--------|-------|
| **RMSE** | 64.44 Wh |
| **MAE** | 43.51 Wh |
| **R²** | -0.2122 |
| **MAPE** | 50.91% |

## License

This project is for educational and research purposes.

![alt text](image.png)
