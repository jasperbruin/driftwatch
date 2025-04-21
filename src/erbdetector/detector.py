from flask import Flask, request, jsonify
import numpy as np

np.float = float
from skmultiflow.drift_detection.adwin import ADWIN
from collections import deque
from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.metrics import CallbackOptions, Observation

# configure OTLP service name
resource = Resource(attributes={"service.name": "erbdetector"})

# configure server name
app = Flask(__name__)

# initialize ADWIN
adwin = ADWIN()
# queue for accuracy calculation
queue = deque(maxlen=1000)

current_accuracy = 0


@app.route("/", methods=["GET"])
def detection():
    # store argument
    is_correct = int(request.args.get("value"))

    # ensure only 0 and 1 are accepted
    if is_correct != 0 and is_correct != 1:
        return jsonify({"error": "only send the values 0 and 1"})

    # append for accuracy calculation
    queue.append(is_correct)

    # append value to adwin for drift detection
    adwin.add_element(is_correct)

    # increase alert counter if ADWIN detected change
    if adwin.detected_change():
        adwin_alerts.add(1)
    else:
        adwin_alerts.add(0)

    # calculate current accuracy
    signal.set_current_value(np.mean(queue))

    # success message
    return jsonify({"message": "success"})


# callback function for gauge metric
def get_accuracy(options: CallbackOptions):
    yield Observation(current_accuracy)


# initialize metric reader
reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(endpoint="otelcollector:4317", insecure=True), 5
)

# configure meterProvider
meterProvider = MeterProvider(resource=resource, metric_readers=[reader])

# set prefered meterProvider
metrics.set_meter_provider(meterProvider)

# create meter
meter = metrics.get_meter("erb-detector")

# create counter metric for the alerts
adwin_alerts = meter.create_counter(name="adwin_alert_counter")


# observable class for drift score export
class Signal:
    current_value = 0

    def __init__(self, attribute):
        self.attribute = attribute

    def set_current_value(self, i):
        self.current_value = i

    def get_current_value(self):
        return self.current_value


# init class for observation
signal = Signal("accuracy_attribute")


# callback function for gauge export
def read_gauge(options: CallbackOptions):
    yield Observation(signal.get_current_value(), {"attribute": signal.attribute})


# create gauge metric for the accuracy
ddb_gauge = meter.create_observable_gauge("accuracy", [read_gauge])

# listen on all ips so that other containers can reach
app.run(host="0.0.0.0")
