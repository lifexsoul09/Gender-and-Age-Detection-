def train_model(model, X_train, y_train):
    print("Starting model training...")
    model.fit(X_train, y_train, batch_size=64, epochs=5)
    print("Model training completed.")
    return model

def make_predictions(model, X_test, scaler):
    print("Making predictions...")
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    print("Predictions made.")
    return predictions
