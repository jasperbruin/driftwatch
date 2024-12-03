# src/recommendationservice/recommendation_server.py

import os
import random
import time
import traceback
from concurrent import futures

import googlecloudprofiler
from google.auth.exceptions import DefaultCredentialsError
import grpc

import demo_pb2
import demo_pb2_grpc
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc

from opentelemetry import trace
from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient, \
    GrpcInstrumentorServer
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import \
    OTLPSpanExporter

from evidently.calculations.stattests import hellinger_stat_test

from evidently.core import ColumnType
from prometheus_client import start_http_server, Gauge


import numpy as np

np.float = float
import pandas as pd
import sklearn
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

from logger import getJSONLogger


class RecommendationModel:
    def __init__(self):
        # Prepare ML Model
        self.ratings = pd.read_json('ratings.json', orient='records')
        self.ratings_test = pd.read_json('ratings_small.json', orient='records')
        self.X, self.user_mapper, self.product_mapper, self.user_inv_mapper, self.product_inv_mapper = self.create_matrix(
            self.ratings)

        self.kNN = NearestNeighbors(n_neighbors=5, algorithm="brute", metric="cosine")
        self.kNN.fit(self.X)

    def create_matrix(self, df):
        N = len(df['user_id'].unique())
        M = len(df['product_id'].unique())

        # Map Ids to indices
        user_mapper = dict(zip(np.unique(df["user_id"]), list(range(N))))
        product_mapper = dict(zip(np.unique(df["product_id"]), list(range(M))))

        # Map indices to IDs
        user_inv_mapper = dict(zip(list(range(N)), np.unique(df["user_id"])))
        product_inv_mapper = dict(zip(list(range(M)), np.unique(df["product_id"])))

        user_index = [user_mapper[i] for i in df['user_id']]
        product_index = [product_mapper[i] for i in df['product_id']]

        X = csr_matrix((df["rating"], (product_index, user_index)), shape=(M, N))

        return X, user_mapper, product_mapper, user_inv_mapper, product_inv_mapper

    def find_similar_products(self, product_id, k, show_distance=False):
        neighbour_ids = []

        product_ind = self.product_mapper[product_id]
        product_vec = self.X[product_ind]
        product_vec = product_vec.reshape(1, -1)
        neighbour = self.kNN.kneighbors(product_vec, return_distance=show_distance)

        for i in range(0, k):
            n = neighbour.item(i)
            neighbour_ids.append(self.product_inv_mapper[n])

        neighbour_ids.pop(0)

        return neighbour_ids


class Signal:
    current_value = 0

    def __init__(self, attribute):
        self.attribute = attribute

    def set_current_value(self, i):
        self.current_value = i

    def get_current_value(self):
        return self.current_value


class RecommendationService(demo_pb2_grpc.RecommendationServiceServicer):
    def __init__(self, model, signal, tracer, accuracy_gauge):
        self.model = model
        self.signal = signal
        self.tracer = tracer
        self.accuracy_gauge = accuracy_gauge
        self.total_predictions = 0
        self.correct_predictions = 0

    def ListRecommendations(self, request, context):
        max_responses = 5
        self.tracer.start_span("ml-model")

        self.logger.debug(f"Received request: {request}")
        response = demo_pb2.ListRecommendationsResponse()
        with self.tracer.start_span("ml-model"):
            current_span = trace.get_current_span()
            if not list(request.product_ids):
                current_span.add_event("No context for ml-model provided, returning no recommendations")
                self.logger.warning("No product IDs received, returning empty recommendations.")
            else:
                product_id = list(request.product_ids)[0]
                self.logger.debug(f"Processing product ID: {product_id}")
                recommendations = self.model.find_similar_products(product_id, max_responses)
                actual_products = self.get_actual_products(product_id)  # Simulate ground truth
                self.update_accuracy(recommendations, actual_products)

                current_span.set_attribute("input", product_id)
                current_span.set_attribute("output", recommendations)
                self.logger.info(f"Generated recommendations: {recommendations} for product ID: {product_id}")
                response.product_ids.extend(recommendations)

        return response

    def get_actual_products(self, product_id):
        """Fetch actual user preferences for comparison."""
        # Fetch products rated highly by the same user who rated `product_id`
        user_id = next((uid for pid, uid in
                        zip(self.model.ratings["product_id"],
                            self.model.ratings["user_id"]) if
                        pid == product_id), None)
        if user_id:
            # Fetch all products rated highly by this user
            actual_products = self.model.ratings[
                self.model.ratings["user_id"] == user_id]
            actual_products = actual_products[actual_products["rating"] >= 4][
                "product_id"].tolist()
            return actual_products
        return []

    def update_accuracy(self, predicted, actual):
        """Update accuracy based on predictions and actual values."""
        self.logger.debug(f"Predicted: {predicted}, Actual: {actual}")
        self.total_predictions += len(predicted)
        self.correct_predictions += len(set(predicted) & set(actual))
        if self.total_predictions > 0:
            accuracy = self.correct_predictions / self.total_predictions
            self.accuracy_gauge.set(accuracy)
            self.logger.info(f"Updated model accuracy: {accuracy:.2f}")

    def Check(self, request, context):
        """Implements the health check method."""
        return health_pb2.HealthCheckResponse(
            status=health_pb2.HealthCheckResponse.SERVING
        )

class RecommendationServer:
    def __init__(self):
        self.logger = getJSONLogger('recommendationservice-server')
        self.logger.info("Initializing recommendationservice")

        self.signal = Signal("drift_score_attribute")
        self.initMetrics()
        self.initTracing()

        self.logger.info("Loading recommendation model...")
        self.model = RecommendationModel()
        self.logger.info("Recommendation model loaded successfully.")

        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        # Register recommendation service
        self.logger.info("Registering recommendation service...")
        self.service = RecommendationService(self.model, self.signal, self.tracer, self.accuracy_gauge)
        demo_pb2_grpc.add_RecommendationServiceServicer_to_server(self.service, self.server)
        self.logger.info("Recommendation service registered successfully.")

        # Register health check service
        self.logger.info("Registering health check service...")
        self.health_service = HealthService()
        health_pb2_grpc.add_HealthServicer_to_server(self.health_service, self.server)
        self.logger.info("Health check service registered successfully.")

        self.startServer()

    def initMetrics(self):
        """Initialize Prometheus metrics."""
        # Gauge for model accuracy
        self.logger.info("Initializing Prometheus metrics...")
        self.accuracy_gauge = Gauge('model_accuracy', 'Accuracy of the model predictions')

        # Gauge for drift score
        self.drift_score_gauge = Gauge('drift_score', 'Current drift score', ['attribute'])

        # Start Prometheus server
        prometheus_port = int(os.getenv('PROMETHEUS_PORT', '9464'))
        start_http_server(prometheus_port)
        self.logger.info(f"Prometheus exporter running on port {prometheus_port}")


    def initTracing(self):
        try:
            grpc_client_instrumentor = GrpcInstrumentorClient()
            grpc_client_instrumentor.instrument()
            grpc_server_instrumentor = GrpcInstrumentorServer()
            grpc_server_instrumentor.instrument()
            if os.environ.get("ENABLE_TRACING", "0") == "1":
                trace.set_tracer_provider(TracerProvider(resource=self.resource))
                otel_endpoint = os.getenv("COLLECTOR_SERVICE_ADDR", "otelcollector:4317")
                trace.get_tracer_provider().add_span_processor(
                    BatchSpanProcessor(
                        OTLPSpanExporter(
                            endpoint=otel_endpoint,
                            insecure=True
                        )
                    )
                )
            self.tracer = trace.get_tracer("Recommendation")
        except Exception as e:
            self.logger.warning(f"Exception during tracing initialization: {e}")
            self.tracer = trace.get_tracer("Recommendation")

    def updateMetrics(self):
        drift_score = self.signal.get_current_value()
        self.logger.debug(f"Drift score before update: {drift_score}")
        self.drift_score_gauge.labels(attribute=self.signal.attribute).set(
            drift_score)
        self.logger.info(f"Updated drift score in Prometheus: {drift_score}")

    def startServer(self):
        port = os.getenv('PORT', "8080")
        catalog_addr = os.getenv('PRODUCT_CATALOG_SERVICE_ADDR', '')
        if not catalog_addr:
            raise Exception('PRODUCT_CATALOG_SERVICE_ADDR environment variable not set')

        self.logger.info(f"Product catalog address: {catalog_addr}")
        channel = grpc.insecure_channel(catalog_addr)
        product_catalog_stub = demo_pb2_grpc.ProductCatalogServiceStub(channel)

        self.logger.info(f"Listening on port: {port}")
        self.server.add_insecure_port(f'[::]:{port}')
        self.server.start()

        try:
            while True:
                self.updateMetrics()
                time.sleep(10)
        except KeyboardInterrupt:
            self.server.stop(0)


class HealthService(health_pb2_grpc.HealthServicer):
    """Implements gRPC health check."""
    def __init__(self):
        super().__init__()

    def Check(self, request, context):
        return health_pb2.HealthCheckResponse(
            status=health_pb2.HealthCheckResponse.SERVING
        )

    def Watch(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        return None


if __name__ == "__main__":
    server = RecommendationServer()
