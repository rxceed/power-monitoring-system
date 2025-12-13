from flask import Flask, request, jsonify
from machine_learning.inference import inference_data

app = Flask(__name__)

@app.get("/")
def hello():
    return "<p>Server is up and running!</p>"

@app.post("/api/inference")
def inference():
    req_body = request.get_json()
    current = req_body['current']
    voltage = req_body['voltage']
    power = req_body['power']
    pf = req_body['power_factor']
    energy = req_body['energy']
    data = {'current': current, 'voltage': voltage, 'power': power, 'power_factor': pf, 'energy': energy}
    pred_label, prob = inference_data(data)
    result = {'prediction': pred_label[0], 'probability': prob}
    return result

    
    