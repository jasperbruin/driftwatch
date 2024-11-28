import json
from flask import Flask, request, jsonify
from prometheus_client import Counter, Gauge, start_http_server
import numpy as np
from skmultiflow.drift_detection.adwin import ADWIN
from collections import deque

# Initialize Flask app
app = Flask(__name__)

# Initialize ADWIN for drift detection
adwin = ADWIN()
queue = deque(maxlen=1000)  # Queue for accuracy calculation

# Prometheus metrics
adwin_alerts = Counter('adwin_alerts_total', 'Total number of ADWIN drift alerts')
accuracy_metric = Gauge('model_accuracy', 'Current model accuracy')

# Start Prometheus exporter on port 5005
start_http_server(5005)

# API route
@app.route('/', methods=['GET'])
def detection():
    # Get the 'value' parameter
    is_correct = request.args.get('value', type=int)

    # Validate input (only 0 or 1 are allowed)
    if is_correct not in [0, 1]:
        return jsonify({'error': 'Only send the values 0 and 1'}), 400

    # Append value to the queue for accuracy calculation
    queue.append(is_correct)

    # Update ADWIN and check for drift
    adwin.add_element(is_correct)
    if adwin.detected_change():
        adwin_alerts.inc()  # Increment alert counter if drift is detected

    # Calculate accuracy and update Prometheus metric
    current_accuracy = np.mean(queue)
    accuracy_metric.set(current_accuracy)

    return jsonify({'message': 'success', 'accuracy': current_accuracy})

# Run Flask app on all IPs
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
