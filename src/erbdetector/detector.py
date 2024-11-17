# /src/erbdetector/detector.py

from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics
import numpy as np
np.float = float
from skmultiflow.drift_detection.adwin import ADWIN
from collections import deque

# Configure server name
app = Flask(__name__)
metrics = PrometheusMetrics(app)

# Initialize ADWIN
adwin = ADWIN()

# Queue for accuracy calculation
queue = deque(maxlen=1000)

# Metrics
current_accuracy = metrics.info("current_accuracy", "Current Accuracy of the Detector")
adwin_alerts_counter = metrics.counter("adwin_alerts", "Number of ADWIN drift alerts")

@app.route('/', methods=['GET'])
def detection():
    # Store argument
    is_correct = int(request.args.get('value'))

    # Ensure only 0 and 1 are accepted
    if is_correct != 0 and is_correct != 1:
        return jsonify({'error': 'only send the values 0 and 1'})

    # Append for accuracy calculation
    queue.append(is_correct)

    # Append value to adwin for drift detection
    adwin.add_element(is_correct)

    # Increase alert counter if ADWIN detected change
    if adwin.detected_change():
        adwin_alerts_counter.inc()

    # Calculate current accuracy
    accuracy = np.mean(queue)
    current_accuracy.set(accuracy)

    # Success message
    return jsonify({'message': 'success'})

# Listen on all IPs so that other containers can reach
app.run(host='0.0.0.0')
