import joblib

# Load the model
model = joblib.load('model.pkl')

def predict(data):
    prediction = model.predict(data)
    return prediction
