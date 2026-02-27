import streamlit as st
import numpy as np
import pickle
import pandas as pd
import plotly.graph_objects as go
from keras.models import load_model
import os

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Deep Energy Forecaster",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Deep Learning Energy Forecaster")
st.markdown("**Bidirectional LSTM | 7-Day Context (168 hrs) | Log-Scaled Target**")
st.markdown("---")

SEQ_LENGTH = 168

# ----------------------------
# Load Resources
# ----------------------------
@st.cache_resource
def load_resources():
    try:
        if not os.path.exists("energy_predictor_lstm.h5"):
            st.error("❌ energy_predictor_lstm.h5 not found")
            return None, None, None

        if not os.path.exists("preprocessing_scalers.pkl"):
            st.error("❌ preprocessing_scalers.pkl not found")
            return None, None, None

        if not os.path.exists("energydata_complete.csv"):
            st.error("❌ energydata_complete.csv not found")
            return None, None, None

        # Load model (SAFE)
        model = load_model("energy_predictor_lstm.h5", compile=False)

        # Load scalers
        with open("preprocessing_scalers.pkl", "rb") as f:
            scalers = pickle.load(f)

        # Load data
        df = pd.read_csv(
            "energydata_complete.csv",
            parse_dates=["date"],
            index_col="date"
        )

        # Log target
        df["Appliances_Log"] = np.log1p(df["Appliances"])

        dataset_features = [
            'Appliances_Log', 'lights',
            'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4',
            'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8',
            'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out',
            'Windspeed', 'Visibility', 'Tdewpoint', 'rv1', 'rv2'
        ]

        df_hourly = df[dataset_features].resample("H").mean().dropna()

        # Cyclical features
        df_hourly["Hour_Sin"] = np.sin(2 * np.pi * df_hourly.index.hour / 24)
        df_hourly["Hour_Cos"] = np.cos(2 * np.pi * df_hourly.index.hour / 24)
        df_hourly["Day_Sin"]  = np.sin(2 * np.pi * df_hourly.index.dayofweek / 7)
        df_hourly["Day_Cos"]  = np.cos(2 * np.pi * df_hourly.index.dayofweek / 7)

        final_cols = dataset_features + [
            "Hour_Sin", "Hour_Cos", "Day_Sin", "Day_Cos"
        ]

        return model, scalers, df_hourly[final_cols]

    except Exception as e:
        st.error(f"❌ Loading error: {e}")
        return None, None, None


model, scalers, df = load_resources()

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("🕒 Simulation Control")

if df is not None:
    valid_times = df.index[SEQ_LENGTH:]
    selected_time = st.sidebar.selectbox(
        "Select Timestamp",
        valid_times.strftime("%Y-%m-%d %H:%M:%S")
    )

    selected_time = pd.to_datetime(selected_time)
    idx = df.index.get_loc(selected_time)

    past_data = df.iloc[idx-SEQ_LENGTH:idx]
    actual_log = df.iloc[idx]["Appliances_Log"]
    actual_real = np.expm1(actual_log)

    st.sidebar.markdown("### 🌤 Conditions")
    st.sidebar.metric("Outdoor Temp", f"{df.iloc[idx]['T_out']:.1f} °C")
    st.sidebar.metric("Humidity", f"{df.iloc[idx]['RH_out']:.0f} %")

    if st.button("🔮 Run Prediction"):
        scaler_features = scalers["scaler_features"]
        scaler_target = scalers["scaler_target"]

        X_scaled = scaler_features.transform(past_data)
        X_input = X_scaled.reshape(1, SEQ_LENGTH, X_scaled.shape[1])

        pred_scaled = model.predict(X_input)
        pred_log = scaler_target.inverse_transform(pred_scaled)
        pred_real = np.expm1(pred_log)[0][0]

        c1, c2, c3 = st.columns(3)
        c1.metric("🤖 Forecast", f"{pred_real:.2f} Wh")
        c2.metric("✅ Actual", f"{actual_real:.2f} Wh")
        c3.metric(
            "⚠ Error",
            f"{abs(pred_real - actual_real):.2f} Wh",
            delta_color="inverse"
        )

        # ----------------------------
        # Plot
        # ----------------------------
        plot_df = past_data.iloc[-24:]
        plot_y = np.expm1(plot_df["Appliances_Log"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_df.index,
            y=plot_y,
            name="Last 24h Usage",
            line=dict(color="#2563eb")
        ))
        fig.add_trace(go.Scatter(
            x=[selected_time],
            y=[actual_real],
            mode="markers",
            name="Actual",
            marker=dict(size=12, color="green")
        ))
        fig.add_trace(go.Scatter(
            x=[selected_time],
            y=[pred_real],
            mode="markers",
            name="Prediction",
            marker=dict(size=12, color="red", symbol="x")
        ))

        fig.update_layout(
            title="⚡ Energy Forecast using 7-Day Context",
            template="plotly_white",
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("⚠ Resources not loaded. Check files.")
