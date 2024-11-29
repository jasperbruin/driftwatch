import json
import logging
from flask import Flask, request, jsonify
from prometheus_client import Counter, Gauge, start_http_server
import numpy as np
from skmultiflow.drift_detection.adwin import ADWIN
from collections import deque

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

# Initialize ADWIN for drift detection
logging.debug("Initializing ADWIN drift detection.")
adwin = ADWIN()
queue = deque(maxlen=10)  # Queue for accuracy calculation
logging.debug(
    "Queue initialized with max length of 10 for accuracy calculation.")

# Prometheus metrics
logging.debug("Initializing Prometheus metrics.")
adwin_alerts = Counter('adwin_alerts_total',
                       'Total number of ADWIN drift alerts')
accuracy_metric = Gauge('model_accuracy', 'Current model accuracy')

# Start Prometheus exporter on port 5005
logging.info("Starting Prometheus HTTP server on port 5005.")
start_http_server(5005)


# API route
@app.route('/', methods=['GET'])
def detection():
    logging.debug("Received request on '/' endpoint.")

    # Get the 'value' parameter
    is_correct = request.args.get('value', type=int)
    logging.debug(f"Parameter 'value' received: {is_correct}")

    # Validate input (only 0 or 1 are allowed)
    if is_correct not in [0, 1]:
        logging.warning(
            f"Invalid value received: {is_correct}. Must be 0 or 1.")
        return jsonify({'error': 'Only send the values 0 and 1'}), 400

    # Append value to the queue for accuracy calculation
    queue.append(is_correct)
    logging.debug(
        f"Appended value {is_correct} to the queue. Current queue size: {len(queue)}")

    # Update ADWIN and check for drift
    adwin.add_element(is_correct)
    if adwin.detected_change():
        adwin_alerts.inc()  # Increment alert counter if drift is detected
        logging.info("ADWIN detected a change. Incremented alert counter.")

    # Calculate accuracy and update Prometheus metric
    current_accuracy = np.mean(queue)
    accuracy_metric.set(current_accuracy)
    logging.debug(
        f"Calculated current accuracy: {current_accuracy}. Updated Prometheus gauge.")

    return jsonify({'message': 'success', 'accuracy': current_accuracy})


# Run Flask app on all IPs
if __name__ == '__main__':
    logging.info("Starting Flask app on host '0.0.0.0', port 5000.")
    app.run(host='0.0.0.0', port=5000)
