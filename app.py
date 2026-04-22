import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Crypto Analysis Dashboard | BTC-LSTM", 
    page_icon="📈", 
    layout="wide"
)

# 2. PROFESSIONAL HEADER & STUDENT DETAILS
def render_header():
    with st.container():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image("https://img.icons8.com/fluency/96/university.png", width=100)
        with col2:
            st.title("Final Year Project: Cryptocurrency Price Prediction")
            st.markdown("### Deep Learning (LSTM) & Market Sentiment Analysis")
    
    st.divider()
    
    with st.sidebar:
        st.markdown("### 🎓 Student Information")
        # GYARA NAN DA BAYANANKA
        st.info(f"""
        **Name:** Your Full Name  
        **Matric No:** AFIT/XYZ/2026/000  
        **Faculty:** Computing  
        **Department:** Computer Science  
        **Supervisor:** Prof./Dr. Name
        """)
        
        st.divider()
        st.markdown("### ⚙️ Model Parameters")
        years_scope = st.slider("Historical Data Scope (Years)", 5, 10, 8)
        predict_days = st.slider("Prediction Horizon (Days)", 7, 60, 30)
        return years_scope, predict_days

years, future_days = render_header()

# ---------------- Load Model ----------------
@st.cache_resource
def load_trained_model():
    try:
        return load_model("BTC_LSTM_sentiment_model.keras")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_trained_model()

# ---------------- Data Processing ----------------
def get_data(years_scope):
    start_date = f"{2026 - years_scope}-01-01"
    try:
        df = yf.download("BTC-USD", start_date)
        if df.empty: return None
        df = df.reset_index()[['Close']]
        df['Close'] = df['Close'].astype(float)
        return df
    except:
        return None

def process_sentiment(price_data):
    try:
        fg = pd.read_csv("fear_greed.csv")
        fg_value = fg.iloc[:, 1].astype(float).values / 100
        if len(fg_value) < len(price_data):
            padding = np.full(len(price_data) - len(fg_value), 0.5)
            fg_value = np.append(fg_value, padding)
        else:
            fg_value = fg_value[:len(price_data)]
        return fg_value
    except:
        return np.full(len(price_data), 0.5)

# ---------------- Execution ----------------
data = get_data(years)

if data is not None and model is not None:
    fg_value = process_sentiment(data)
    
    # Scaling
    dataset = np.column_stack((data['Close'].values.flatten(), fg_value))
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(dataset)

    base_days = 100
    x_test = []
    for i in range(base_days, len(scaled_data)):
        x_test.append(scaled_data[i-base_days:i])
    x_test = np.array(x_test)

    # Historical Prediction
    with st.spinner('Analyzing market trends...'):
        pred_scaled = model.predict(x_test, verbose=0)
        combined_pred = np.hstack([pred_scaled, scaled_data[base_days:, 1].reshape(-1, 1)])
        pred_price = scaler.inverse_transform(combined_pred)[:, 0]
        actual_price = data['Close'].values[base_days:]

    # ---------------- DASHBOARD METRICS (FIXED) ----------------
    # Muna amfani da .flatten() da float() don magance "TypeError"
    rmse = np.sqrt(mean_squared_error(actual_price, pred_price))
    mae = mean_absolute_error(actual_price, pred_price)
    
    current_price_scalar = float(actual_price.flatten()[-1])
    rmse_scalar = float(np.array(rmse).flatten()[0])
    mae_scalar = float(np.array(mae).flatten()[0])

    m1, m2, m3 = st.columns(3)
    m1.metric("Current BTC Price", f"${current_price_scalar:,.2f}")
    m2.metric("Model RMSE (Error)", f"${rmse_scalar:,.2f}")
    m3.metric("Model MAE (Error)", f"${mae_scalar:,.2f}")

    st.divider()

    # ---------------- VISUALIZATION ----------------
    st.subheader("📊 Market Analysis & Forecast Visualization")
    
    sns.set_style("whitegrid")
    fig, ax1 = plt.subplots(figsize=(12, 5))

    x_hist = np.arange(len(actual_price))
    x_future = np.arange(len(actual_price), len(actual_price) + future_days)

    # Future Forecasting
    last_sequence = scaled_data[-base_days:].copy()
    last_sentiment = float(last_sequence[-1, 1])
    future_preds_scaled = []
    
    for i in range(future_days):
        input_seq = last_sequence.reshape(1, base_days, 2)
        p_scaled = model.predict(input_seq, verbose=0)
        next_step = np.array([p_scaled[0, 0], last_sentiment])
        future_preds_scaled.append(next_step)
        last_sequence = np.vstack([last_sequence[1:], next_step])
    
    future_preds_scaled = np.array(future_preds_scaled)
    future_price = scaler.inverse_transform(future_preds_scaled)[:, 0]
    
    # Plotting
    ax1.plot(x_hist, actual_price, label="Actual Price", color="#1f77b4")
    ax1.plot(x_hist, pred_price, "--", label="LSTM Prediction", color="#d62728", alpha=0.7)
    ax1.plot(x_future, future_price, "-.", label=f"{future_days}-Day Forecast", color="#2ca02c", linewidth=2)
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.fill_between(np.arange(len(fg_value)), 0, fg_value, color="orange", alpha=0.1)
    ax2.set_ylabel("Sentiment Score")
    
    st.pyplot(fig)

    # ---------------- DATA TABLES ----------------
    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📝 Accuracy (Last 10 Days)")
        history_df = pd.DataFrame({
            "Actual": actual_price.flatten(),
            "Predicted": pred_price.flatten()
        }).tail(10)
        st.table(history_df.style.format("${:,.2f}"))

    with c2:
        st.subheader(f"🔮 Next {future_days} Days Forecast")
        future_df = pd.DataFrame({
            "Day": [f"Day {i+1}" for i in range(future_days)],
            "Price": future_price.flatten()
        })
        st.dataframe(future_df.style.format({"Price": "${:,.2f}"}), use_container_width=True)

    st.markdown("### 💾 Export Research Data")
    st.download_button("Download Forecast CSV", future_df.to_csv(index=False), "btc_forecast.csv", "text/csv")

else:
    st.error("Missing Data: Please ensure 'BTC_LSTM_sentiment_model.keras' and 'fear_greed.csv' are in the project folder.")

st.markdown("---")
st.caption("BSc Computer Science Thesis Evaluation - 2026 Academic Session")