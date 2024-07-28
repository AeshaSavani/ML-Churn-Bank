from flask import Flask, jsonify, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load models
with open('log_reg_model.pkl', 'rb') as f:
    log_reg_model = pickle.load(f)
with open('svm.pkl', 'rb') as f:
    svm_model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    log_reg_pred = log_reg_model.predict(features)[0]
    svm_pred = svm_model.predict(features)[0]
    return jsonify({'log_reg_prediction': log_reg_pred, 'svm_prediction': svm_pred})

if __name__ == '__main__':
    app.run(debug=True)
