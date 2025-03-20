from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load models
linear_regression_model = pickle.load(open(r'C:\Users\farha\Desktop\linear_regression_model.pkl', 'rb'))
random_forest_model = pickle.load(open(r'C:\Users\farha\Desktop\random_forest_model.pkl', 'rb'))
decision_tree_model = pickle.load(open(r'C:\Users\farha\Desktop\decision_tree_model.pkl', 'rb'))
neural_network_model = pickle.load(open(r'C:\Users\farha\Desktop\neural_network_model.pkl', 'rb'))


@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received JSON:", data)  # Debugging output

        required_keys = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
        
        # Check if all required fields exist
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            return jsonify({'error': f'Missing required fields: {missing_keys}'}), 400

        # Convert input to DataFrame
        features = pd.DataFrame([data])

        print("DataFrame:\n", features)  # Debugging

        # Make predictions
        linear_regression_prediction = linear_regression_model.predict(features)[0]
        random_forest_prediction = random_forest_model.predict(features)[0]
        decision_tree_prediction = decision_tree_model.predict(features)[0]
        neural_network_prediction = neural_network_model.predict(features)[0]

        return jsonify({
            'linear_regression_prediction': linear_regression_prediction,
            'random_forest_prediction': random_forest_prediction,
            'decision_tree_prediction': decision_tree_prediction,
            'neural_network_prediction': neural_network_prediction
        })

    except Exception as e:
        print("🔥 ERROR:", str(e))  # Print error in Flask console
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
