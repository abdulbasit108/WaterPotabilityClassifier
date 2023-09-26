from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS
import numpy as np

app = Flask(__name__)

CORS(app)


def predict_potability(ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity):

    user_input = np.array([ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity])

    reshaped_array = user_input.reshape(1, -1)

    prediction = model.predict(reshaped_array)

    return int(prediction[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    required_params = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
    for param in required_params:
        if param not in data:
            return jsonify({'error': f'Missing parameter: {param}'}), 400

    ph = data['ph']
    Hardness = data['Hardness']
    Solids = data['Solids']
    Chloramines = data['Chloramines']
    Sulfate = data['Sulfate']
    Conductivity = data['Conductivity']
    Organic_carbon = data['Organic_carbon']
    Trihalomethanes = data['Trihalomethanes']
    Turbidity = data['Turbidity']

    prediction = predict_potability(float(ph), float(Hardness), float(Solids), float(Chloramines), float(Sulfate), float(Conductivity), float(Organic_carbon), float(Trihalomethanes), float(Turbidity))
    
    return jsonify({'prediction': prediction}), 200

if __name__ == '__main__':
     
     filename = "task_model.pickle"

     model = pickle.load(open(filename, "rb"))

     app.run(host='192.168.56.1', debug=True)
