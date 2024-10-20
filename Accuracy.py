import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
import asyncio
import websockets
from flask import Flask, jsonify

app = Flask(__name__)

# 1. Preprocess WebSocket data to extract features for the model
def preprocess_data(ticker_info):
    processed_data = {
        'best_ask_price': float(ticker_info['a'][0]),
        'best_bid_price': float(ticker_info['b'][0]),
        'close_price': float(ticker_info['c'][0]),
        'volume_today': float(ticker_info['v'][0]),
        'open_price_today': float(ticker_info['o'][0]),
        'high_price_today': float(ticker_info['h'][0]),
        'low_price_today': float(ticker_info['l'][0]),
        'vwap_today': float(ticker_info['p'][0]),
        'Change %': ((float(ticker_info['c'][0]) - float(ticker_info['o'][0])) / float(ticker_info['o'][0])) * 100
    }
    return processed_data

# 2. Signal Generation for Buy/Sell/Hold
def generate_signals(data):
    conditions = [
        (data['Change %'] > 0.5),
        (data['Change %'] < -0.5)
    ]
    choices = [1, -1]  # Buy: 1, Sell: -1
    data['Signal'] = np.select(conditions, choices, default=0)
    return data

# 3. Load historical data and generate buy/sell signals for training
def load_historical_data(file_path):
    historical_data = pd.read_csv(file_path)
    historical_data = generate_signals(historical_data)
    return historical_data

# 4. Train the RandomForest model using historical data
def train_model(data):
    features = ['best_ask_price', 'best_bid_price', 'close_price', 'volume_today', 'open_price_today', 'high_price_today', 'low_price_today', 'vwap_today', 'Change %']
    target = 'Signal'

    X = data[features]
    y = data[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Hold', 'Buy', 'Sell'])

    buy_accuracy = accuracy_score(y_test == 1, y_pred == 1)
    sell_accuracy = accuracy_score(y_test == -1, y_pred == -1)

    return model, accuracy, buy_accuracy, sell_accuracy, report

# 5. WebSocket Listener to get real-time stock data and predict Buy/Sell signals
async def listen_to_websocket(model):
    url = "wss://ws.kraken.com"
    
    async with websockets.connect(url) as ws:
        # Subscription message to subscribe to ticker data
        subscribe_message = json.dumps({
            "event": "subscribe",
            "pair": ["XBT/USD"],  # Adjust this to track a different pair
            "subscription": {"name": "ticker"}
        })
        await ws.send(subscribe_message)
        
        while True:
            try:
                message = await ws.recv()
                message = json.loads(message)
                
                if isinstance(message, list) and len(message) > 1:
                    ticker_info = message[1]  # Extract the ticker info from WebSocket message
                    df_real_time = pd.DataFrame([preprocess_data(ticker_info)])
                    
                    features = ['best_ask_price', 'best_bid_price', 'close_price', 'volume_today', 'open_price_today', 'high_price_today', 'low_price_today', 'vwap_today', 'Change %']
                    signal_prediction = model.predict(df_real_time[features])

                    # Output the signal to frontend or console
                    result = {
                        'Signal': int(signal_prediction[0]),  # 1: Buy, -1: Sell, 0: Hold
                        'Prediction': 'Buy' if signal_prediction[0] == 1 else 'Sell' if signal_prediction[0] == -1 else 'Hold'
                    }
                    print(result)  # You can send this to frontend via an API

            except Exception as e:
                print(f"Error receiving message: {e}")

# 6. Endpoint to start the WebSocket listener and use the model for real-time predictions
@app.route('/start', methods=['GET'])
def start_websocket_listener():
    # Load historical data and train the model
    historical_data = load_historical_data('historical_stock_data.csv')
    model, overall_accuracy, buy_accuracy, sell_accuracy, report = train_model(historical_data)
    
    # Start WebSocket listening in an async event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(listen_to_websocket(model))

    return jsonify({
        'message': 'WebSocket listener started!',
        'Overall Accuracy': overall_accuracy,
        'Buy Accuracy': buy_accuracy,
        'Sell Accuracy': sell_accuracy
    })

if __name__ == '__main__':
    app.run(debug=True)
