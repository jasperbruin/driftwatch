from src.models.model_training import train_model

def retrain_model(new_data):
    X_train, y_train = new_data
    model = train_model(X_train, y_train)
    # Save or deploy the updated model
    return model
