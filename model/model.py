from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Function to create LSTM model
def create_model(X_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to make predictions
def predict_stock_price(model, X_test):
    return model.predict(X_test)
