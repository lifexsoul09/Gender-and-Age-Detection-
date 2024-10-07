from flask import Flask, render_template, request
import pandas as pd
from model.preprocess import preprocess_data
from model.model import create_model, predict_stock_price
import numpy as np
import yfinance as yf

def fetch_stock_data(stock_ticker):
    data = yf.download(stock_ticker, start='2019-01-01', end='2024-01-01')
    return data



app = Flask(__name__)

@app.route('/preprocessed')
def preprocessed():
    preprocessed_data = "Your preprocessed data here"  # Replace with actual preprocessed data
    return render_template('preprocessed.html', preprocessed_data=preprocessed_data)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    stock_ticker = request.form['stock_ticker'].upper()
    try:
        # Fetch stock data
        data = fetch_stock_data(stock_ticker)  # Ensure this function is defined
        X_train, y_train, X_test, y_test, scaler = preprocess_data(data)

        # Create and train the LSTM model
        model = create_model(X_train)
        model.fit(X_train, y_train, epochs=5, batch_size=32)

        # Make predictions
        predicted_stock_price = predict_stock_price(model, X_test)

        # Inverse scale the predicted values back to the original price scale
        predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

        # Align the actual test data
        valid_test_data = y_test[60:]  # Skip the first 'time_step' points

        # Convert arrays to lists for rendering in HTML
        predicted_prices_list = predicted_stock_price.flatten().tolist()  # Flatten the predictions
        valid_test_data_list = valid_test_data.flatten().tolist()  # Flatten actual prices

        # Check lengths for debug purposes
        print(f"Length of predictions: {len(predicted_prices_list)}")
        print(f"Length of actual stock prices: {len(valid_test_data_list)}")

        # Prepare data for visualization
        return render_template('index.html', predicted_price=predicted_prices_list, actual_prices=valid_test_data_list)
    except Exception as e:
        return f"Error during model training/prediction: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
