import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Function to preprocess the data
def preprocess_data(data, time_step=60):
    # Ensure we are only using the 'Close' column for price prediction
    data = data[['Close']]

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Split data into training (80%) and testing (20%)
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    # Use the helper function to create sequences
    X_train, y_train = create_sequences(train_data, time_step)
    X_test, y_test = create_sequences(test_data, time_step)

    # Reshape data for LSTM: [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test, y_test, scaler

# Function to create sequences for LSTM
def create_sequences(data, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)
