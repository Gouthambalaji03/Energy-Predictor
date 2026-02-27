import streamlit as st
import numpy as np
import pickle
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from tensorflow.keras.models import load_model
import tensorflow as tf
import os
import io

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Energy Forecaster",
    page_icon="⚡",
    layout="wide"
)

SEQ_LENGTH = 168
ELECTRICITY_RATE_DEFAULT = 0.12  # $/kWh

# ----------------------------
# Load Resources
# ----------------------------
@st.cache_resource(ttl=300)
def load_resources():
    try:
        for f in ["energy_predictor_lstm.h5", "preprocessing_scalers.pkl", "energydata_complete.csv"]:
            if not os.path.exists(f):
                st.error(f"File not found: {f}")
                return None, None, None, None, None

        model = load_model("energy_predictor_lstm.h5", compile=False)

        with open("preprocessing_scalers.pkl", "rb") as f:
            scalers = pickle.load(f)

        # Load optional artifacts
        history = None
        if os.path.exists("training_history.pkl"):
            with open("training_history.pkl", "rb") as f:
                history = pickle.load(f)

        test_preds = None
        if os.path.exists("test_predictions.pkl"):
            with open("test_predictions.pkl", "rb") as f:
                test_preds = pickle.load(f)

        df = pd.read_csv("energydata_complete.csv", parse_dates=["date"], index_col="date")
        df["Appliances_Log"] = np.log1p(df["Appliances"])

        dataset_features = [
            'Appliances_Log', 'lights',
            'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4',
            'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8',
            'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out',
            'Windspeed', 'Visibility', 'Tdewpoint', 'rv1', 'rv2'
        ]

        df_hourly = df[dataset_features].resample("h").mean().dropna()

        df_hourly["Hour_Sin"] = np.sin(2 * np.pi * df_hourly.index.hour / 24)
        df_hourly["Hour_Cos"] = np.cos(2 * np.pi * df_hourly.index.hour / 24)
        df_hourly["Day_Sin"]  = np.sin(2 * np.pi * df_hourly.index.dayofweek / 7)
        df_hourly["Day_Cos"]  = np.cos(2 * np.pi * df_hourly.index.dayofweek / 7)
        df_hourly["Month_Sin"] = np.sin(2 * np.pi * df_hourly.index.month / 12)
        df_hourly["Month_Cos"] = np.cos(2 * np.pi * df_hourly.index.month / 12)
        df_hourly["Is_Weekend"] = (df_hourly.index.dayofweek >= 5).astype(float)
        hour = df_hourly.index.hour
        df_hourly["Is_Peak"] = ((hour >= 7) & (hour <= 9) | (hour >= 17) & (hour <= 21)).astype(float)
        df_hourly["Rolling_Mean_6h"] = df_hourly["Appliances_Log"].rolling(6, min_periods=1).mean()
        df_hourly["Rolling_Std_6h"] = df_hourly["Appliances_Log"].rolling(6, min_periods=1).std().fillna(0)
        df_hourly["Rolling_Mean_24h"] = df_hourly["Appliances_Log"].rolling(24, min_periods=1).mean()
        df_hourly["Temp_Diff"] = df_hourly["T1"] - df_hourly["T_out"]
        df_hourly["Lag_24h"] = df_hourly["Appliances_Log"].shift(24).bfill()
        df_hourly["Lag_168h"] = df_hourly["Appliances_Log"].shift(168).bfill()
        df_hourly = df_hourly.dropna()

        final_cols = dataset_features + [
            "Hour_Sin", "Hour_Cos", "Day_Sin", "Day_Cos",
            "Month_Sin", "Month_Cos",
            "Is_Weekend", "Is_Peak",
            "Rolling_Mean_6h", "Rolling_Std_6h", "Rolling_Mean_24h",
            "Temp_Diff",
            "Lag_24h", "Lag_168h"
        ]

        return model, scalers, df_hourly[final_cols], history, test_preds

    except Exception as e:
        st.error(f"Loading error: {e}")
        return None, None, None, None, None


def predict_single(model, scalers, past_data):
    """Run a single prediction."""
    scaler_features = scalers["scaler_features"]
    scaler_target = scalers["scaler_target"]
    X_scaled = scaler_features.transform(past_data)
    X_input = X_scaled.reshape(1, SEQ_LENGTH, X_scaled.shape[1])
    pred_scaled = model.predict(X_input, verbose=0)
    pred_log = scaler_target.inverse_transform(pred_scaled)
    return np.expm1(pred_log)[0][0]


def predict_with_confidence(model, scalers, past_data, n_passes=10):
    """Monte Carlo dropout for confidence intervals."""
    scaler_features = scalers["scaler_features"]
    scaler_target = scalers["scaler_target"]
    X_scaled = scaler_features.transform(past_data)
    X_input = X_scaled.reshape(1, SEQ_LENGTH, X_scaled.shape[1])

    predictions = []
    for _ in range(n_passes):
        pred = model(X_input, training=True)  # dropout active
        pred_log = scaler_target.inverse_transform(pred.numpy())
        predictions.append(np.expm1(pred_log)[0][0])

    predictions = np.array(predictions)
    return predictions.mean(), predictions.std(), predictions.min(), predictions.max()


def detect_anomaly(actual, rolling_mean, rolling_std):
    """Flag if actual is >2 std devs from rolling mean."""
    if rolling_std == 0:
        return False
    z_score = abs(actual - rolling_mean) / rolling_std
    return z_score > 2


# ----------------------------
# Load Data
# ----------------------------
model, scalers, df, history, test_preds = load_resources()

if df is None:
    st.error("Could not load resources. Check files.")
    st.stop()

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("⚡ Energy Forecaster")
st.sidebar.markdown("---")

electricity_rate = st.sidebar.number_input(
    "Electricity Rate ($/kWh)", min_value=0.01, value=ELECTRICITY_RATE_DEFAULT, step=0.01
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info**")
st.sidebar.caption("Bidirectional LSTM + Attention")
st.sidebar.caption(f"Context: {SEQ_LENGTH}h (7 days)")
st.sidebar.caption(f"Features: {df.shape[1]}")

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Forecasting", "📦 Batch Prediction", "📊 Data Explorer", "📈 Model Performance"
])

# ============================================================
# TAB 1: FORECASTING
# ============================================================
with tab1:
    st.header("🔮 Energy Forecasting")

    col_ctrl, col_info = st.columns([1, 2])

    with col_ctrl:
        valid_times = df.index[SEQ_LENGTH:]
        selected_time = st.selectbox(
            "Select Timestamp", valid_times.strftime("%Y-%m-%d %H:%M:%S"),
            index=len(valid_times) // 2
        )
        selected_time = pd.to_datetime(selected_time)
        idx = df.index.get_loc(selected_time)

        forecast_mode = st.radio(
            "Forecast Mode",
            ["Single Step (1h)", "Multi-Step (6h)", "Multi-Step (12h)", "Multi-Step (24h)"]
        )

        run_btn = st.button("Run Prediction", type="primary", use_container_width=True)

    with col_info:
        row = df.iloc[idx]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🌡 Outdoor Temp", f"{row['T_out']:.1f} °C")
        c2.metric("💧 Humidity", f"{row['RH_out']:.0f} %")
        c3.metric("💨 Wind", f"{row['Windspeed']:.1f} m/s")
        c4.metric("📊 Pressure", f"{row['Press_mm_hg']:.0f} mmHg")

    if run_btn:
        past_data = df.iloc[idx - SEQ_LENGTH:idx]
        actual_log = df.iloc[idx]["Appliances_Log"]
        actual_real = np.expm1(actual_log)

        # Anomaly detection
        rolling_mean_val = np.expm1(row["Rolling_Mean_24h"])
        rolling_std_val = np.expm1(row["Rolling_Std_6h"]) if row["Rolling_Std_6h"] > 0 else 20
        is_anomaly = detect_anomaly(actual_real, rolling_mean_val, rolling_std_val)

        steps_map = {
            "Single Step (1h)": 1,
            "Multi-Step (6h)": 6,
            "Multi-Step (12h)": 12,
            "Multi-Step (24h)": 24
        }
        n_steps = steps_map[forecast_mode]

        if n_steps == 1:
            # Single step with confidence interval
            mean_pred, std_pred, min_pred, max_pred = predict_with_confidence(model, scalers, past_data)
            error = abs(mean_pred - actual_real)
            cost_pred = (mean_pred / 1000) * electricity_rate

            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🤖 Forecast", f"{mean_pred:.1f} Wh")
            m2.metric("✅ Actual", f"{actual_real:.1f} Wh")
            m3.metric("⚠ Error", f"{error:.1f} Wh")
            m4.metric("💰 Est. Cost", f"${cost_pred:.4f}")

            if is_anomaly:
                st.warning(f"🚨 Anomaly Detected! Actual ({actual_real:.0f} Wh) deviates significantly from the 24h rolling average ({rolling_mean_val:.0f} Wh).")

            st.info(f"📊 Confidence: {mean_pred:.1f} Wh ± {std_pred:.1f} Wh (range: {min_pred:.1f} – {max_pred:.1f} Wh)")

            # Chart
            plot_df = past_data.iloc[-24:]
            plot_y = np.expm1(plot_df["Appliances_Log"])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=plot_df.index, y=plot_y, name="Last 24h Usage",
                line=dict(color="#3b82f6", width=2),
                fill="tozeroy", fillcolor="rgba(59,130,246,0.1)"
            ))
            # Confidence band
            fig.add_trace(go.Scatter(
                x=[selected_time], y=[max_pred], mode="markers",
                marker=dict(size=1, color="rgba(0,0,0,0)"), showlegend=False
            ))
            fig.add_shape(
                type="rect",
                x0=selected_time - pd.Timedelta(hours=0.5),
                x1=selected_time + pd.Timedelta(hours=0.5),
                y0=min_pred, y1=max_pred,
                fillcolor="rgba(251,191,36,0.2)", line=dict(width=0)
            )
            fig.add_trace(go.Scatter(
                x=[selected_time], y=[actual_real], mode="markers",
                name="Actual", marker=dict(size=14, color="#10b981", symbol="circle")
            ))
            fig.add_trace(go.Scatter(
                x=[selected_time], y=[mean_pred], mode="markers",
                name="Prediction", marker=dict(size=14, color="#f59e0b", symbol="diamond")
            ))
            fig.update_layout(
                title="Energy Forecast with Confidence Interval",
                template="plotly_dark", height=450,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
                xaxis=dict(gridcolor="rgba(51,65,85,0.5)"),
                yaxis=dict(title="Energy (Wh)", gridcolor="rgba(51,65,85,0.5)"),
                legend=dict(orientation="h", y=1.08)
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            # Multi-step forecasting
            predictions = []
            timestamps = []
            actuals = []
            costs = []

            with st.spinner(f"Forecasting {n_steps} hours..."):
                for step in range(n_steps):
                    step_idx = idx + step
                    if step_idx >= len(df):
                        break
                    step_past = df.iloc[step_idx - SEQ_LENGTH:step_idx]
                    pred = predict_single(model, scalers, step_past)
                    actual = np.expm1(df.iloc[step_idx]["Appliances_Log"])
                    predictions.append(pred)
                    actuals.append(actual)
                    timestamps.append(df.index[step_idx])
                    costs.append((pred / 1000) * electricity_rate)

            results_df = pd.DataFrame({
                "Timestamp": timestamps,
                "Predicted (Wh)": predictions,
                "Actual (Wh)": actuals,
                "Error (Wh)": [abs(p - a) for p, a in zip(predictions, actuals)],
                "Est. Cost ($)": costs
            })

            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Avg Prediction", f"{np.mean(predictions):.1f} Wh")
            m2.metric("Avg Actual", f"{np.mean(actuals):.1f} Wh")
            m3.metric("Avg Error", f"{results_df['Error (Wh)'].mean():.1f} Wh")
            m4.metric("Total Cost", f"${sum(costs):.4f}")

            # Multi-step chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=timestamps, y=actuals, name="Actual",
                line=dict(color="#10b981", width=2)
            ))
            fig.add_trace(go.Scatter(
                x=timestamps, y=predictions, name="Predicted",
                line=dict(color="#f59e0b", width=2, dash="dash")
            ))
            fig.update_layout(
                title=f"Multi-Step Forecast ({n_steps}h)",
                template="plotly_dark", height=400,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
                xaxis=dict(gridcolor="rgba(51,65,85,0.5)"),
                yaxis=dict(title="Energy (Wh)", gridcolor="rgba(51,65,85,0.5)"),
                legend=dict(orientation="h", y=1.08)
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(results_df.set_index("Timestamp"), use_container_width=True)


# ============================================================
# TAB 2: BATCH PREDICTION
# ============================================================
with tab2:
    st.header("📦 Batch Prediction")

    valid_dates = df.index[SEQ_LENGTH:]
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=valid_dates[0].date(),
                                    min_value=valid_dates[0].date(),
                                    max_value=valid_dates[-1].date())
    with col2:
        end_date = st.date_input("End Date", value=min(valid_dates[0].date() + pd.Timedelta(days=2), valid_dates[-1].date()),
                                  min_value=valid_dates[0].date(),
                                  max_value=valid_dates[-1].date())

    max_hours = st.slider("Max hours to predict", 1, 168, 48)

    if st.button("Run Batch Prediction", type="primary"):
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date) + pd.Timedelta(hours=23)

        mask = (df.index >= start_ts) & (df.index <= end_ts) & (df.index.isin(valid_dates))
        batch_indices = [df.index.get_loc(t) for t in df.index[mask][:max_hours]]

        if not batch_indices:
            st.warning("No valid timestamps in selected range.")
        else:
            predictions = []
            progress = st.progress(0)

            for i, bidx in enumerate(batch_indices):
                past = df.iloc[bidx - SEQ_LENGTH:bidx]
                pred = predict_single(model, scalers, past)
                actual = np.expm1(df.iloc[bidx]["Appliances_Log"])
                rolling_m = np.expm1(df.iloc[bidx]["Rolling_Mean_24h"])
                rolling_s = np.expm1(df.iloc[bidx]["Rolling_Std_6h"]) if df.iloc[bidx]["Rolling_Std_6h"] > 0 else 20
                is_anom = detect_anomaly(actual, rolling_m, rolling_s)

                predictions.append({
                    "Timestamp": df.index[bidx],
                    "Predicted (Wh)": round(pred, 2),
                    "Actual (Wh)": round(actual, 2),
                    "Error (Wh)": round(abs(pred - actual), 2),
                    "Error (%)": round(abs(pred - actual) / max(actual, 1) * 100, 1),
                    "Est. Cost ($)": round((pred / 1000) * electricity_rate, 5),
                    "Anomaly": "Yes" if is_anom else "No"
                })
                progress.progress((i + 1) / len(batch_indices))

            batch_df = pd.DataFrame(predictions)

            # Summary
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total Predictions", len(batch_df))
            s2.metric("Avg Error", f"{batch_df['Error (Wh)'].mean():.1f} Wh")
            s3.metric("Max Error", f"{batch_df['Error (Wh)'].max():.1f} Wh")
            anomaly_count = len(batch_df[batch_df["Anomaly"] == "Yes"])
            s4.metric("Anomalies", anomaly_count)

            # Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=batch_df["Timestamp"], y=batch_df["Actual (Wh)"],
                name="Actual", line=dict(color="#10b981", width=2)
            ))
            fig.add_trace(go.Scatter(
                x=batch_df["Timestamp"], y=batch_df["Predicted (Wh)"],
                name="Predicted", line=dict(color="#3b82f6", width=2, dash="dash")
            ))
            # Mark anomalies
            anomalies = batch_df[batch_df["Anomaly"] == "Yes"]
            if len(anomalies) > 0:
                fig.add_trace(go.Scatter(
                    x=anomalies["Timestamp"], y=anomalies["Actual (Wh)"],
                    mode="markers", name="Anomaly",
                    marker=dict(size=12, color="#ef4444", symbol="triangle-up")
                ))
            fig.update_layout(
                title="Batch Predictions vs Actual",
                template="plotly_dark", height=450,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
                xaxis=dict(gridcolor="rgba(51,65,85,0.5)"),
                yaxis=dict(title="Energy (Wh)", gridcolor="rgba(51,65,85,0.5)"),
                legend=dict(orientation="h", y=1.08)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Table
            st.dataframe(
                batch_df.set_index("Timestamp").style.applymap(
                    lambda v: "color: #ef4444; font-weight: bold" if v == "Yes" else "",
                    subset=["Anomaly"]
                ),
                use_container_width=True
            )

            # Export
            csv = batch_df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV", csv, "batch_predictions.csv", "text/csv",
                use_container_width=True
            )


# ============================================================
# TAB 3: DATA EXPLORER
# ============================================================
with tab3:
    st.header("📊 Data Explorer")

    # Date filter
    explore_col1, explore_col2 = st.columns(2)
    with explore_col1:
        explore_start = st.date_input("From", value=df.index[0].date(), key="exp_start",
                                       min_value=df.index[0].date(), max_value=df.index[-1].date())
    with explore_col2:
        explore_end = st.date_input("To", value=df.index[-1].date(), key="exp_end",
                                     min_value=df.index[0].date(), max_value=df.index[-1].date())

    df_filtered = df[(df.index >= pd.Timestamp(explore_start)) & (df.index <= pd.Timestamp(explore_end) + pd.Timedelta(hours=23))]
    energy_wh = np.expm1(df_filtered["Appliances_Log"])

    # Feature statistics
    st.subheader("Feature Statistics")
    stats_cols = ["Appliances_Log", "T1", "T_out", "RH_1", "RH_out", "Windspeed", "Press_mm_hg", "Temp_Diff"]
    available_stats = [c for c in stats_cols if c in df_filtered.columns]
    stats_df = df_filtered[available_stats].describe().T
    stats_df.columns = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    st.dataframe(stats_df.round(2), use_container_width=True)

    # Distribution + Box plot side by side
    st.subheader("Energy Consumption Distribution")
    dist_col1, dist_col2 = st.columns(2)

    with dist_col1:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=energy_wh, nbinsx=50,
            marker=dict(color="rgba(59,130,246,0.6)", line=dict(color="#3b82f6", width=1))
        ))
        fig_hist.update_layout(
            title="Distribution (Wh)", template="plotly_dark", height=350,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            xaxis=dict(title="Energy (Wh)", gridcolor="rgba(51,65,85,0.5)"),
            yaxis=dict(title="Frequency", gridcolor="rgba(51,65,85,0.5)")
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with dist_col2:
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=energy_wh, name="Energy",
            marker_color="#3b82f6", boxmean=True
        ))
        fig_box.update_layout(
            title="Box Plot (Wh)", template="plotly_dark", height=350,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            yaxis=dict(title="Energy (Wh)", gridcolor="rgba(51,65,85,0.5)")
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Seasonal patterns
    st.subheader("Seasonal Patterns")
    season_col1, season_col2, season_col3 = st.columns(3)

    with season_col1:
        hourly_avg = energy_wh.groupby(energy_wh.index.hour).mean()
        fig_hourly = go.Figure()
        fig_hourly.add_trace(go.Bar(
            x=hourly_avg.index, y=hourly_avg.values,
            marker_color="#3b82f6"
        ))
        fig_hourly.update_layout(
            title="Avg by Hour", template="plotly_dark", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            xaxis=dict(title="Hour", gridcolor="rgba(51,65,85,0.5)"),
            yaxis=dict(title="Wh", gridcolor="rgba(51,65,85,0.5)")
        )
        st.plotly_chart(fig_hourly, use_container_width=True)

    with season_col2:
        daily_avg = energy_wh.groupby(energy_wh.index.dayofweek).mean()
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Bar(
            x=day_names, y=daily_avg.values,
            marker_color="#10b981"
        ))
        fig_daily.update_layout(
            title="Avg by Day of Week", template="plotly_dark", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            xaxis=dict(gridcolor="rgba(51,65,85,0.5)"),
            yaxis=dict(title="Wh", gridcolor="rgba(51,65,85,0.5)")
        )
        st.plotly_chart(fig_daily, use_container_width=True)

    with season_col3:
        weekly_avg = energy_wh.resample("W").mean()
        fig_weekly = go.Figure()
        fig_weekly.add_trace(go.Scatter(
            x=weekly_avg.index, y=weekly_avg.values,
            line=dict(color="#f59e0b", width=2), fill="tozeroy",
            fillcolor="rgba(245,158,11,0.1)"
        ))
        fig_weekly.update_layout(
            title="Weekly Trend", template="plotly_dark", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            xaxis=dict(gridcolor="rgba(51,65,85,0.5)"),
            yaxis=dict(title="Wh", gridcolor="rgba(51,65,85,0.5)")
        )
        st.plotly_chart(fig_weekly, use_container_width=True)

    # Correlation Heatmap
    st.subheader("Feature Correlation Heatmap")
    corr_features = ["Appliances_Log", "lights", "T1", "T_out", "RH_1", "RH_out",
                     "Windspeed", "Press_mm_hg", "Temp_Diff", "Is_Weekend", "Is_Peak",
                     "Rolling_Mean_6h", "Lag_24h"]
    available_corr = [c for c in corr_features if c in df_filtered.columns]
    corr_matrix = df_filtered[available_corr].corr()

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont=dict(size=10)
    ))
    fig_corr.update_layout(
        template="plotly_dark", height=500,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)"
    )
    st.plotly_chart(fig_corr, use_container_width=True)


# ============================================================
# TAB 4: MODEL PERFORMANCE
# ============================================================
with tab4:
    st.header("📈 Model Performance")

    if test_preds is not None:
        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RMSE", f"{test_preds['rmse']:.2f} Wh")
        m2.metric("MAE", f"{test_preds['mae']:.2f} Wh")
        m3.metric("R² Score", f"{test_preds['r2']:.4f}")
        m4.metric("MAPE", f"{test_preds['mape']:.2f}%")

        perf_col1, perf_col2 = st.columns(2)

        # Actual vs Predicted scatter
        with perf_col1:
            st.subheader("Actual vs Predicted")
            y_test = test_preds["y_test"]
            y_pred = test_preds["y_pred"]

            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=y_test, y=y_pred, mode="markers",
                marker=dict(color="#3b82f6", size=4, opacity=0.5),
                name="Predictions"
            ))
            # Perfect prediction line
            max_val = max(y_test.max(), y_pred.max())
            fig_scatter.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode="lines", name="Perfect",
                line=dict(color="#ef4444", dash="dash", width=2)
            ))
            fig_scatter.update_layout(
                template="plotly_dark", height=400,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
                xaxis=dict(title="Actual (Wh)", gridcolor="rgba(51,65,85,0.5)"),
                yaxis=dict(title="Predicted (Wh)", gridcolor="rgba(51,65,85,0.5)"),
                legend=dict(orientation="h", y=1.08)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Error distribution
        with perf_col2:
            st.subheader("Error Distribution")
            errors = y_pred - y_test

            fig_err = go.Figure()
            fig_err.add_trace(go.Histogram(
                x=errors, nbinsx=50,
                marker=dict(color="rgba(239,68,68,0.6)", line=dict(color="#ef4444", width=1))
            ))
            fig_err.add_vline(x=0, line_dash="dash", line_color="#10b981", line_width=2)
            fig_err.update_layout(
                template="plotly_dark", height=400,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
                xaxis=dict(title="Prediction Error (Wh)", gridcolor="rgba(51,65,85,0.5)"),
                yaxis=dict(title="Frequency", gridcolor="rgba(51,65,85,0.5)")
            )
            st.plotly_chart(fig_err, use_container_width=True)

        # Residual plot
        st.subheader("Residuals Over Test Set")
        fig_resid = go.Figure()
        fig_resid.add_trace(go.Scatter(
            x=list(range(len(errors))), y=errors,
            mode="markers", marker=dict(color="#f59e0b", size=3, opacity=0.5),
            name="Residual"
        ))
        fig_resid.add_hline(y=0, line_dash="dash", line_color="#10b981", line_width=2)
        fig_resid.update_layout(
            template="plotly_dark", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            xaxis=dict(title="Sample Index", gridcolor="rgba(51,65,85,0.5)"),
            yaxis=dict(title="Error (Wh)", gridcolor="rgba(51,65,85,0.5)")
        )
        st.plotly_chart(fig_resid, use_container_width=True)

    else:
        st.info("No test predictions found. Run `python main.py` to generate model performance data.")

    # Training curves
    if history is not None:
        st.subheader("Training History")
        hist_col1, hist_col2 = st.columns(2)

        with hist_col1:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                y=history["loss"], name="Train Loss",
                line=dict(color="#3b82f6", width=2)
            ))
            if "val_loss" in history:
                fig_loss.add_trace(go.Scatter(
                    y=history["val_loss"], name="Val Loss",
                    line=dict(color="#ef4444", width=2)
                ))
            fig_loss.update_layout(
                title="Loss Curves", template="plotly_dark", height=350,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
                xaxis=dict(title="Epoch", gridcolor="rgba(51,65,85,0.5)"),
                yaxis=dict(title="Loss (Huber)", gridcolor="rgba(51,65,85,0.5)"),
                legend=dict(orientation="h", y=1.08)
            )
            st.plotly_chart(fig_loss, use_container_width=True)

        with hist_col2:
            if "lr" in history:
                fig_lr = go.Figure()
                fig_lr.add_trace(go.Scatter(
                    y=history["lr"], name="Learning Rate",
                    line=dict(color="#10b981", width=2)
                ))
                fig_lr.update_layout(
                    title="Learning Rate Schedule", template="plotly_dark", height=350,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
                    xaxis=dict(title="Epoch", gridcolor="rgba(51,65,85,0.5)"),
                    yaxis=dict(title="LR", gridcolor="rgba(51,65,85,0.5)")
                )
                st.plotly_chart(fig_lr, use_container_width=True)
            else:
                st.info("Learning rate history not available.")
    else:
        st.info("No training history found. Run `python main.py` to generate training data.")
