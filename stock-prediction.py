# stock-prediction.py

import traceback
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from collections import deque

# Initialize FastAPI app
app = FastAPI()

# Initialize a buffer to store the last 60 time steps
buffer = deque(maxlen=60)  # Stores the last 60 time steps

# 1. Fetch Historical Data from Kraken's OHLC API
def fetch_historical_data(pair='XBTUSD', interval=21600):
    """
    Fetches historical OHLC data from Kraken's public API.

    Args:
        pair (str): Trading pair, e.g., 'XBTUSD'.
        interval (int): Time interval in minutes.

    Returns:
        pd.DataFrame: DataFrame containing 'open', 'high', 'low', 'close' prices.
    """
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}"
    response = requests.get(url)
    data = response.json()

    if data['error']:
        raise Exception(f"Error fetching data: {data['error']}")

    ohlc_key = list(data['result'].keys())[0]  # Dynamically get the pair key
    ohlc_data = data['result'][ohlc_key]
    df = pd.DataFrame(ohlc_data, columns=[
        'time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
    ])
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    return df[['open', 'high', 'low', 'close']]

# 2. Preprocess Data
def preprocess_data(df, sequence_length=60):
    """
    Scales the data and creates sequences for RNN training.

    Args:
        df (pd.DataFrame): DataFrame containing stock data.
        sequence_length (int): Number of time steps in each input sequence.

    Returns:
        tuple: Scaled training and testing data along with the scaler.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    X = []
    y = []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        # Binary classification: 1 if close price increases, else 0
        y.append(1 if scaled_data[i, 3] > scaled_data[i-1, 3] else 0)

    X, y = np.array(X), np.array(y)
    return X, y, scaler

# 3. Split Data into Training and Testing Sets
def split_data(X, y, train_ratio=0.8):
    """
    Splits the data into training and testing sets.

    Args:
        X (np.array): Feature sequences.
        y (np.array): Target labels.
        train_ratio (float): Proportion of data to use for training.

    Returns:
        tuple: Training and testing datasets.
    """
    split_index = int(len(X) * train_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test

# 4. Build the RNN Model
def build_rnn(input_shape):
    """
    Constructs the RNN (LSTM) model.

    Args:
        input_shape (tuple): Shape of the input data (timesteps, features).

    Returns:
        Sequential: Compiled Keras model.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))  # Binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 5. Train the Model
def train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Trains the RNN model.

    Args:
        X_train (np.array): Training feature sequences.
        y_train (np.array): Training target labels.
        X_val (np.array): Validation feature sequences.
        y_val (np.array): Validation target labels.
        epochs (int): Number of training epochs.
        batch_size (int): Size of each training batch.

    Returns:
        Sequential: Trained Keras model.
        History: Training history.
    """
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_rnn(input_shape)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )
    return model, history

# 6. Evaluate the Model with Separate Buy and Sell Accuracy
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test set and prints separate accuracy for buy and sell signals.

    Args:
        model (Sequential): Trained Keras model.
        X_test (np.array): Testing feature sequences.
        y_test (np.array): Testing target labels.

    Returns:
        None
    """
    # Predict probabilities
    y_pred_prob = model.predict(X_test)
    # Convert probabilities to binary predictions
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate per-class accuracy
    accuracy_sell = tn / (tn + fp) if (tn + fp) != 0 else 0
    accuracy_buy = tp / (tp + fn) if (tp + fn) != 0 else 0

    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Sell (0)', 'Buy (1)']))

    print(f"Accuracy for Sell (Class 0): {accuracy_sell:.4f}")
    print(f"Accuracy for Buy (Class 1): {accuracy_buy:.4f}")

# 7. Save the Model and Scaler
def save_model(model, scaler, model_path='rnn_model.h5', scaler_path='scaler.pkl'):
    """
    Saves the trained model and scaler to disk.

    Args:
        model (Sequential): Trained Keras model.
        scaler (MinMaxScaler): Fitted scaler.
        model_path (str): Path to save the model.
        scaler_path (str): Path to save the scaler.
    """
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

# 8. FastAPI Real-Time Prediction Endpoint
class StockData(BaseModel):
    open_price_today: float
    high_price_today: float
    low_price_today: float
    close_price: float

# Load trained models and scaler (for real-time prediction)
model_buy = load_model('rnn_model.h5')  # Example: Use the same model for buy/sell prediction
scaler = joblib.load('scaler.pkl')

# @app.post("/predict")
# async def predict(stock_data: StockData):
#     # Prepare real-time data
#     input_data = np.array([[
#         stock_data.open_price_today,
#         stock_data.high_price_today,
#         stock_data.low_price_today,
#         stock_data.close_price
#     ]])

# Initialize a buffer to store the last 60 time steps
buffer = deque(maxlen=60)  # Stores the last 60 time steps

@app.post("/predict")
async def predict(stock_data: StockData):
    try:
        # Log received data
        print(f"Received data: {stock_data}")

        # Prepare the input data for prediction
        input_data = [[
            stock_data.open_price_today,
            stock_data.high_price_today,
            stock_data.low_price_today,
            stock_data.close_price
        ]]

        # Append the new data to the buffer
        buffer.append(input_data[0])
        # Check if we have 60 time steps
        if len(buffer) < 60:
            print(f"Not enough data yet. Currently have {len(buffer)} time steps.")
            return {"message": "Not enough data yet for prediction."}

        # Once we have 60 time steps, prepare the input for the model
        
        input_sequence = np.array(buffer).reshape(1, 60, 4)  # Reshape to (1, 60, 4)

        # Scale the input sequence
        scaled_data = scaler.transform(input_sequence[0])  # Transform only the features
        scaled_data = scaled_data.reshape(1, 60, 4)  # Reshape back to (1, 60, 4)

        # Make the prediction
        buy_prob = model_buy.predict(scaled_data)[0][0]
        print(f"Prediction probability: {buy_prob}")

        # Create a buy/sell signal
        buy_signal = "Buy Signal" if buy_prob > 0.5 else "No Buy"

        # Return the result
        return {
            "buy_signal": buy_signal,
            "buy_probability": buy_prob
        }

    except Exception as e:
        print(f"Error during prediction: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")


# 9. Main Function
def main():
    # Step 1: Fetch Data
    print("Fetching historical data...")
    df = fetch_historical_data()
    print(f"Data fetched: {df.shape[0]} records.")
    
    # Step 2: Preprocess Data
    print("Preprocessing data...")
    X, y, scaler = preprocess_data(df)
    print(f"Data preprocessed: {X.shape}, {y.shape}")
    
    # Step 3: Split Data
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")
    
    # Step 4: Train Model
    print("Training the RNN model...")
    model, history = train_model(X_train, y_train, X_test, y_test)
    print("Model training completed.")
    
    # Step 5: Evaluate Model
    print("Evaluating the model on the test set...")
    evaluate_model(model, X_test, y_test)
    
    # Step 6: Save Model and Scaler
    print("Saving the model and scaler...")
    save_model(model, scaler)
    print("All tasks completed successfully.")

if __name__ == "__main__":
    main()
