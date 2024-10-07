import matplotlib.pyplot as plt

def plot_predictions(data, predictions, ticker):
    train = data[:int(len(data) * 0.8)]
    valid = data[int(len(data) * 0.8):].copy()
    valid['Predictions'] = predictions
    
    plt.figure(figsize=(16, 8))
    plt.title(f'{ticker} Stock Price Prediction')
    plt.plot(train['Close'], label='Training Data')
    plt.plot(valid['Close'], label='Actual Prices')
    plt.plot(valid['Predictions'], label='Predicted Prices')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.savefig(f'static/{ticker}_prediction.png')  # Save plot as image
    plt.close()
