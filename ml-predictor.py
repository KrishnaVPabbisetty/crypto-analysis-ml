import sys
import joblib
import pandas as pd

# Load the trained models
model_buy = joblib.load('model_buy_strict.pkl')
model_sell = joblib.load('model_sell_strict.pkl')

# Get real-time OHLC data from Node.js
open_price = float(sys.argv[1])
high_price = float(sys.argv[2])
low_price = float(sys.argv[3])
close_price = float(sys.argv[4])
volume = float(sys.argv[5])

# Prepare the data for the model
df_real_time = pd.DataFrame([{
    'open': open_price,
    'high': high_price,
    'low': low_price,
    'close': close_price,
    'volume': volume
}])

# Apply the same technical indicators used during training
df_real_time['EMA_3'] = df_real_time['close'].ewm(span=3, adjust=False).mean()
df_real_time['EMA_7'] = df_real_time['close'].ewm(span=7, adjust=False).mean()
df_real_time['BB_Middle'] = df_real_time['close'].rolling(window=10).mean()
df_real_time['BB_Upper'] = df_real_time['BB_Middle'] + (2 * df_real_time['close'].rolling(window=10).std())
df_real_time['BB_Lower'] = df_real_time['BB_Middle'] - (2 * df_real_time['close'].rolling(window=10).std())
df_real_time = df_real_time.dropna()

# Predict the buy/sell signal
X_real_time = df_real_time[['open', 'high', 'low', 'close', 'volume', 'EMA_3', 'EMA_7', 'BB_Upper', 'BB_Lower']]
buy_signal = model_buy.predict(X_real_time)
sell_signal = model_sell.predict(X_real_time)

# Output the predictions for Node.js
print(f'{buy_signal[0]},{sell_signal[0]}')
