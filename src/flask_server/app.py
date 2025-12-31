from flask import Flask, request, jsonify, render_template
from machine_learning.inference import inference_data_catboost, inference_data_RNN
import requests

app = Flask(__name__)

@app.get("/")
def hello():
    return "<p>Server is up and running!</p>"

@app.get("/fetch_data")
def fetch_external_data():
    API_URL = "http://nodered:1880/api/classification"
    """
    Helper function to fetch data from the external URL.
    Returns a list of 2 dictionaries.
    """
    # Default placeholder data
    data_structure = [
        {"title": "Catboost", "label": "Waiting...", "confidence": 0.0},
        {"title": "SimpleRNN", "label": "Waiting...", "confidence": 0.0}
    ]

    try:
        response = requests.get(API_URL, timeout=2)
        response.raise_for_status()
        external_data = response.json()
        
        # Update our structure with the fetched values
        if isinstance(external_data, list) and len(external_data) >= 2:
            data_structure[0]["label"] = external_data[0].get("label", "Unknown")
            data_structure[0]["confidence"] = external_data[0].get("confidence", 0.0)
            
            data_structure[1]["label"] = external_data[1].get("label", "Unknown")
            data_structure[1]["confidence"] = external_data[1].get("confidence", 0.0)

    except Exception as e:
        print(f"Error fetching data: {e}")
    
    return data_structure

@app.get("/dashboard")
def classification_display():
    displays_data = fetch_external_data()
    return render_template('index.html', displays=displays_data)

@app.route('/live-data')
def live_data():
    """
    Background route used by JavaScript. 
    Returns JSON data instead of HTML.
    """
    data = fetch_external_data()
    return jsonify(data)

@app.post("/api/inference/catboost")
def inference_catboost():
    req_body = request.get_json()
    current = req_body['current']
    voltage = req_body['voltage']
    power = req_body['power']
    pf = req_body['power_factor']
    energy = req_body['energy']
    data = {'current': current, 'voltage': voltage, 'energy': energy, 'power_factor': pf, 'power': power}
    pred_label, prob = inference_data_catboost(data)
    result = {'prediction': pred_label[0], 'probability': prob}
    return result

@app.post("/api/inference/rnn")
def inference_rnn():
    req_body = request.get_json()
    label, conf = inference_data_RNN(req_body)
    result = {'prediction': label, 'confidence': float(conf)}
    return result
    

    
    