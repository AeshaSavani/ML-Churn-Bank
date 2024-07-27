from flask import Flask, request, jsonify
import pandas as pd
from app.model import predict

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()  # Assuming data is sent as JSON
    df = pd.DataFrame(data)
    prediction = predict(df)
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(debug=True)
