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
    def __init__(self, model, signal, tracer):
        self.model = model
        self.signal = signal
        self.tracer = tracer

    def ListRecommendations(self, request, context):
        max_responses = 5
        self.tracer.start_span("ml-model")

        # Logging connection check
        print("Received request from frontend:", request)

        response = demo_pb2.ListRecommendationsResponse()
        with self.tracer.start_span("ml-model"):
            current_span = trace.get_current_span()
            if not list(request.product_ids):
                current_span.add_event(
                    "No context for ml-model provided, returning no recommendations")
                print(
                    "No product IDs received, returning empty recommendations.")
            else:
                ids = self.model.find_similar_products(
                    list(request.product_ids)[0], max_responses)
                current_span.set_attribute("explanation",
                                           "these products have a similar review-profile compared to the original product")
                current_span.set_attribute("input",
                                           list(request.product_ids)[0])
                current_span.set_attribute("output", str(ids))
                print(
                    f"Generated recommendations: {ids} for product ID: {list(request.product_ids)[0]}")
                response.product_ids.extend(ids)

            current_span.set_attribute("model-metric",
                                       self.model.kNN.effective_metric_)
            current_span.set_attribute("number of features",
                                       self.model.kNN.n_features_in_)

        self.signal.set_current_value(
            hellinger_stat_test(self.model.ratings["rating"],
                                self.model.ratings_test["rating"],
                                ColumnType.Numerical, 0.1).drift_score)
        print(f"Updated drift score: {self.signal.get_current_value()}")
        return response

    def Check(self, request, context):
        print("Health check request received.")
        return health_pb2.HealthCheckResponse(
            status=health_pb2.HealthCheckResponse.SERVING)

    def Watch(self, request, context):
        print("Watch request received.")
        return health_pb2.HealthCheckResponse(
            status=health_pb2.HealthCheckResponse.UNIMPLEMENTED)


class RecommendationServer:
    def __init__(self):
        # Initialize logger
        self.logger = getJSONLogger('recommendationservice-server')
        self.logger.info("initializing recommendationservice")

        # Initialize Stackdriver Profiler
        self.initStackdriverProfiling()

        # Initialize resource
        self.resource = Resource(attributes={
            "service.name": "recommendationservice"
        })

        # Initialize Signal
        self.signal = Signal("drift_score_attribute")

        # Initialize tracing
        self.initTracing()

        # Initialize metrics
        self.initMetrics()

        # Initialize ML Model
        self.model = RecommendationModel()

        # Initialize gRPC server
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        # Create RecommendationService
        self.service = RecommendationService(self.model, self.signal, self.tracer)
        demo_pb2_grpc.add_RecommendationServiceServicer_to_server(self.service, self.server)
        health_pb2_grpc.add_HealthServicer_to_server(self.service, self.server)

        # Start server
        self.startServer()

    def initStackdriverProfiling(self):
        project_id = None
        try:
            project_id = os.environ["GCP_PROJECT_ID"]
        except KeyError:
            # Environment variable not set
            pass

        for retry in range(1, 4):
            try:
                if project_id:
                    googlecloudprofiler.start(service='recommendation_server',
                                              service_version='1.0.0', verbose=0,
                                              project_id=project_id)
                else:
                    googlecloudprofiler.start(service='recommendation_server',
                                              service_version='1.0.0', verbose=0)
                self.logger.info("Successfully started Stackdriver Profiler.")
                return
            except (BaseException) as exc:
                self.logger.info(
                    "Unable to start Stackdriver Profiler Python agent. " + str(
                        exc))
                if (retry < 4):
                    self.logger.info(
                        "Sleeping %d seconds to retry Stackdriver Profiler agent initialization" % (
                            retry * 10))
                    time.sleep(1)
                else:
                    self.logger.warning(
                        "Could not initialize Stackdriver Profiler after retrying, giving up")
        return

    def initTracing(self):
        try:
            grpc_client_instrumentor = GrpcInstrumentorClient()
            grpc_client_instrumentor.instrument()
            grpc_server_instrumentor = GrpcInstrumentorServer()
            grpc_server_instrumentor.instrument()
            if os.environ["ENABLE_TRACING"] == "1":
                trace.set_tracer_provider(TracerProvider(resource=self.resource))
                otel_endpoint = os.getenv("COLLECTOR_SERVICE_ADDR",
                                          "otelcollector:4317")
                trace.get_tracer_provider().add_span_processor(
                    BatchSpanProcessor(
                        OTLPSpanExporter(
                            endpoint=otel_endpoint,
                            insecure=True
                        )
                    )
                )
                # additional
                self.tracer = trace.get_tracer("Recommendation")
            else:
                self.tracer = trace.get_tracer("Recommendation")
        except (KeyError, DefaultCredentialsError):
            self.logger.info("Tracing disabled.")
            self.tracer = trace.get_tracer("Recommendation")
        except Exception as e:
            self.logger.warn(
                f"Exception on Cloud Trace setup: {traceback.format_exc()}, tracing disabled.")
            self.tracer = trace.get_tracer("Recommendation")

    def initMetrics(self):
        # Prometheus Metrics
        self.drift_score_gauge = Gauge(
            'drift_score', 'Current drift score', ['attribute']
        )

        # Start Prometheus HTTP server on a given port
        prometheus_port = int(os.getenv('PROMETHEUS_PORT', '9464'))
        start_http_server(prometheus_port)
        self.logger.info(
            f"Prometheus exporter running on port {prometheus_port}")

    def updateMetrics(self):
        # Update drift score in Prometheus gauge
        drift_score = self.signal.get_current_value()
        self.drift_score_gauge.labels(attribute=self.signal.attribute).set(drift_score)
        self.logger.info(f"Updated drift score in Prometheus: {drift_score}")

    def startServer(self):
        port = os.environ.get('PORT', "8080")
        catalog_addr = os.environ.get('PRODUCT_CATALOG_SERVICE_ADDR', '')
        if catalog_addr == "":
            raise Exception(
                'PRODUCT_CATALOG_SERVICE_ADDR environment variable not set')
        self.logger.info("product catalog address: " + catalog_addr)
        channel = grpc.insecure_channel(catalog_addr)
        product_catalog_stub = demo_pb2_grpc.ProductCatalogServiceStub(channel)

        self.logger.info("listening on port: " + port)
        self.server.add_insecure_port('[::]:' + port)
        self.server.start()

        # Keep alive
        try:
            while True:
                # Periodically update metrics
                self.updateMetrics()
                time.sleep(10)
        except KeyboardInterrupt:
            self.server.stop(0)


if __name__ == "__main__":
    server = RecommendationServer()
