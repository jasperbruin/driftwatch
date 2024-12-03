# src/detector/detector.py

import logging
from flask import Flask, request, jsonify
from prometheus_client import Counter, Gauge, start_http_server
import numpy as np
from collections import deque
from scipy.stats import ks_2samp

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

# Initialize drift detection
logging.debug("Initializing drift detection.")
reference_window_size = 50
detection_window_size = 50
reference_window = deque(maxlen=reference_window_size)
detection_window = deque(maxlen=detection_window_size)
drift_detected = False

# Prometheus metrics
logging.debug("Initializing Prometheus metrics.")
drift_alerts = Counter('drift_alerts_total', 'Total number of drift alerts')
p_value_metric = Gauge('drift_p_value', 'P-value of the drift test')

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

    # Append value to the detection window
    detection_window.append(is_correct)
    logging.debug(
        f"Appended value {is_correct} to the detection window. Current size: {len(detection_window)}")

    # Once both windows are full, perform drift detection
    if len(reference_window) == reference_window_size and len(detection_window) == detection_window_size:
        # Perform statistical test (e.g., Kolmogorov-Smirnov test)
        stat, p_value = ks_2samp(reference_window, detection_window)
        p_value_metric.set(p_value)
        logging.debug(f"Performed KS test: stat={stat}, p-value={p_value}")

        # If p-value is below a threshold, we detect drift
        if p_value < 0.05:
            drift_alerts.inc()
            logging.warning(f"Drift detected! p-value={p_value}")
            # Reset reference window to current detection window
            reference_window.clear()
            reference_window.extend(detection_window)
            detection_window.clear()
        else:
            # No drift detected; shift the windows
            # Move half of detection window to reference window
            half_size = detection_window_size // 2
            for _ in range(half_size):
                reference_window.append(detection_window.popleft())

    elif len(reference_window) < reference_window_size:
        # Fill the reference window first
        reference_window.append(is_correct)
        logging.debug(
            f"Filling reference window. Current size: {len(reference_window)}")

    return jsonify({'message': 'success'})

# Run Flask app on all IPs
if __name__ == '__main__':
    logging.info("Starting Flask app on host '0.0.0.0', port 5000.")
    app.run(host='0.0.0.0', port=5000)
