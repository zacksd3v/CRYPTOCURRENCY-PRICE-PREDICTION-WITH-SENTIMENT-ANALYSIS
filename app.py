import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(page_title="Crypto Price Prediction", layout="wide")
st.title("📈 Bitcoin Price Prediction using LSTM & Sentiment Analysis")

# ---------------- Load model ----------------
@st.cache_resource
def load_trained_model():
    return load_model("BTC_LSTM_sentiment_model.keras")

model = load_trained_model()

# ---------------- User controls ----------------
years = st.slider("Select years of historical data", 5, 10, 10)
future_days = st.slider("Future prediction days", 7, 60, 30)

# ---------------- Load BTC data ----------------
start_date = f"{2026-years}-01-01"
data = yf.download("BTC-USD", start_date)
data = data.reset_index()[['Close']]
data['Close'] = data['Close'].astype(float)

# ---------------- Load sentiment ----------------
fg = pd.read_csv("fear_greed.csv")
fg_value = fg.iloc[:, 1].astype(float).values / 100

# Align sentiment length with price
fg_value = fg_value[:len(data)]
if len(fg_value) < len(data):
    fg_value = np.append(fg_value, [0.5] * (len(data) - len(fg_value)))

# ---------------- Prepare input ----------------
dataset = np.column_stack((data['Close'].values, fg_value))
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dataset)

base_days = 100
x = []
for i in range(base_days, len(scaled_data)):
    x.append(scaled_data[i-base_days:i])
x = np.array(x)

# ---------------- Predict historical ----------------
pred_scaled = model.predict(x, verbose=0)
pred_price = scaler.inverse_transform(
    np.hstack([
        pred_scaled,
        scaled_data[base_days:, 1].reshape(-1, 1)
    ])
)[:, 0]

actual_price = data['Close'].values[base_days:]

rmse = np.sqrt(mean_squared_error(actual_price, pred_price))
mae = mean_absolute_error(actual_price, pred_price)

col1, col2 = st.columns(2)
col1.metric("RMSE (USD)", f"{rmse:,.2f}")
col2.metric("MAE (USD)", f"{mae:,.2f}")

# ---------------- Future prediction ----------------
last_sequence = scaled_data[-base_days:].copy()
last_sentiment = float(last_sequence[-1, 1])

future_sentiment = np.array([last_sentiment] * future_days)

future_predictions = []
for i in range(future_days):
    input_seq = last_sequence.reshape(1, base_days, 2)
    pred_scaled = model.predict(input_seq, verbose=0)
    next_day_scaled = np.array([pred_scaled[0, 0], future_sentiment[i]])
    future_predictions.append(next_day_scaled)
    last_sequence = np.vstack([last_sequence[1:], next_day_scaled])

future_predictions = np.array(future_predictions)

future_price = scaler.inverse_transform(
    np.hstack([
        future_predictions[:, 0].reshape(-1, 1),
        future_predictions[:, 1].reshape(-1, 1)
    ])
)[:, 0]

# ---------------- Plot ----------------
fig, ax1 = plt.subplots(figsize=(12, 6))

x_hist = np.arange(len(actual_price))
x_future = np.arange(len(actual_price), len(actual_price) + future_days)

ax1.plot(x_hist, actual_price, label="Actual Price", color="blue")
ax1.plot(x_hist, pred_price, "--", label="Predicted Price", color="red")
ax1.plot(x_future, future_price, "-.", label="Future Prediction", color="green")
ax1.set_ylabel("BTC Price (USD)")
ax1.legend(loc="upper left")

ax2 = ax1.twinx()
ax2.plot(np.arange(len(fg_value)), fg_value, color="orange", alpha=0.4, label="Sentiment")
ax2.set_ylabel("Sentiment (0–1)")
ax2.legend(loc="upper right")

st.pyplot(fig)

# ---------------- Tables ----------------
st.subheader("📄 Historical Prediction Data")

history_df = pd.DataFrame({
    "Actual Price (USD)": actual_price.reshape(-1),
    "Predicted Price (USD)": pred_price.reshape(-1)
})


st.dataframe(history_df.tail(20))

st.subheader("🔮 Future Price Predictions")

future_df = pd.DataFrame({
    "Day": np.arange(1, future_days + 1),
    "Predicted BTC Price (USD)": future_price
})

st.dataframe(future_df)

# ---------------- Downloads ----------------
st.download_button(
    "⬇️ Download Historical Predictions",
    history_df.to_csv(index=False),
    "historical_predictions.csv",
    "text/csv"
)

st.download_button(
    "⬇️ Download Future Predictions",
    future_df.to_csv(index=False),
    "future_predictions.csv",
    "text/csv"
)
