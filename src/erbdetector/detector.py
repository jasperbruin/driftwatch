from flask import Flask, request, jsonify, abort
from prometheus_flask_exporter import PrometheusMetrics
import numpy as np
np.float = float
from skmultiflow.drift_detection.adwin import ADWIN
from collections import deque
from prometheus_client import Counter, Gauge, Histogram

app = Flask(__name__)
metrics = PrometheusMetrics(app)

adwin = ADWIN()
queue = deque(maxlen=1000)

# Metrics
current_accuracy = Gauge("detector_current_accuracy_ratio", "Current accuracy of the detector (ratio)")
adwin_alerts_counter = Counter("detector_adwin_alerts_total", "Total number of ADWIN drift alerts")
accuracy_histogram = Histogram(
    'detector_accuracy_histogram',
    'Histogram of accuracy values',
    buckets=[i * 0.1 for i in range(11)]
)
queue_size = Gauge("detector_queue_size", "Current size of the accuracy queue")
total_predictions = Counter("detector_total_predictions", "Total number of predictions processed")

@app.route('/', methods=['GET'])
def detection():
    # Get and validate input
    value = request.args.get('value')
    if value not in ['0', '1']:
        abort(400, description="Invalid value: only 0 and 1 are accepted.")
    is_correct = int(value)

    # Update metrics
    total_predictions.inc()
    queue.append(is_correct)
    queue_size.set(len(queue))

    # Update ADWIN and check for drift
    adwin.add_element(is_correct)
    if adwin.detected_change():
        adwin_alerts_counter.inc()

    # Calculate and update accuracy metrics
    accuracy = np.mean(queue)
    current_accuracy.set(accuracy)
    accuracy_histogram.observe(accuracy)

    return jsonify({'message': 'success'})

if __name__ == '__main__':
    app.run(host='0.0.0.0')
