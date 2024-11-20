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

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import \
    OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.metrics import CallbackOptions, Observation

from evidently.calculations.stattests import hellinger_stat_test
from evidently.metric_preset import DataDriftPreset
from evidently.options.data_drift import DataDriftOptions
from evidently.core import ColumnType

from prometheus_client import start_http_server


import numpy as np

np.float = float
import pandas as pd
import sklearn
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

from logger import getJSONLogger

logger = getJSONLogger('recommendationservice-server')


class Signal:
    current_value = 0

    def __init__(self, attribute):
        self.attribute = attribute

    def set_current_value(self, i):
        self.current_value = i

    def get_current_value(self):
        return self.current_value


class MLModel:
    def __init__(self):
        self.ratings = pd.read_json('ratings.json', orient='records')
        self.ratings_test = pd.read_json('ratings_small.json',
                                         orient='records')
        self.X, self.user_mapper, self.product_mapper, self.user_inv_mapper, self.product_inv_mapper = self.create_matrix(
            self.ratings)
        self.kNN = NearestNeighbors(n_neighbors=5, algorithm="brute",
                                    metric="cosine")
        self.kNN.fit(self.X)

    def create_matrix(self, df):
        N = len(df['user_id'].unique())
        M = len(df['product_id'].unique())

        # Map Ids to indices
        user_mapper = dict(zip(np.unique(df["user_id"]), list(range(N))))
        product_mapper = dict(zip(np.unique(df["product_id"]), list(range(M))))

        # Map indices to IDs
        user_inv_mapper = dict(zip(list(range(N)), np.unique(df["user_id"])))
        product_inv_mapper = dict(
            zip(list(range(M)), np.unique(df["product_id"])))

        user_index = [user_mapper[i] for i in df['user_id']]
        product_index = [product_mapper[i] for i in df['product_id']]

        X = csr_matrix((df["rating"], (product_index, user_index)),
                       shape=(M, N))

        return X, user_mapper, product_mapper, user_inv_mapper, product_inv_mapper

    def find_similar_products(self, product_id, k, show_distance=False):
        neighbour_ids = []

        product_ind = self.product_mapper[product_id]
        product_vec = self.X[product_ind]
        product_vec = product_vec.reshape(1, -1)
        neighbour = self.kNN.kneighbors(product_vec,
                                        return_distance=show_distance)

        for i in range(0, k):
            n = neighbour.item(i)
            neighbour_ids.append(self.product_inv_mapper[n])

        neighbour_ids.pop(0)

        return neighbour_ids


class Telemetry:
    def __init__(self, signal):
        self.signal = signal
        self.init_profiling()
        self.init_tracing()
        self.init_metrics()

    def init_profiling(self):
        try:
            if "DISABLE_PROFILER" in os.environ:
                raise KeyError()
            else:
                logger.info("Profiler enabled.")
                self.initStackdriverProfiling()
        except KeyError:
            logger.info("Profiler disabled.")

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
                                              service_version='1.0.0',
                                              verbose=0, project_id=project_id)
                else:
                    googlecloudprofiler.start(service='recommendation_server',
                                              service_version='1.0.0',
                                              verbose=0)
                logger.info("Successfully started Stackdriver Profiler.")
                return
            except (BaseException) as exc:
                logger.info(
                    "Unable to start Stackdriver Profiler Python agent. " + str(
                        exc))
                if (retry < 4):
                    logger.info(
                        "Sleeping %d seconds to retry Stackdriver Profiler agent initialization" % (
                                    retry * 10))
                    time.sleep(1)
                else:
                    logger.warning(
                        "Could not initialize Stackdriver Profiler after retrying, giving up")
        return

    def init_tracing(self):
        try:
            grpc_client_instrumentor = GrpcInstrumentorClient()
            grpc_client_instrumentor.instrument()
            grpc_server_instrumentor = GrpcInstrumentorServer()
            grpc_server_instrumentor.instrument()
            if os.environ["ENABLE_TRACING"] == "1":
                self.resource = Resource(attributes={
                    "service_name": "recommendationservice"
                })
                trace.set_tracer_provider(
                    TracerProvider(resource=self.resource))
                otel_endpoint = os.getenv("COLLECTOR_SERVICE_ADDR",
                                          "localhost:4317")
                trace.get_tracer_provider().add_span_processor(
                    BatchSpanProcessor(
                        OTLPSpanExporter(
                            endpoint=otel_endpoint,
                            insecure=True
                        )
                    )
                )
                # additional
                global tracer
                tracer = trace.get_tracer("Recommendation")
        except (KeyError, DefaultCredentialsError):
            logger.info("Tracing disabled.")
        except Exception as e:
            logger.warn(
                f"Exception on Cloud Trace setup: {traceback.format_exc()}, tracing disabled.")

    def init_metrics(self):
        try:
            if not hasattr(self, 'resource'):
                self.resource = Resource(attributes={
                    "service.name": "recommendationservice"
                })
            reader = PrometheusMetricReader()
            meterProvider = MeterProvider(resource=self.resource,
                                          metric_readers=[reader])
            metrics.set_meter_provider(meterProvider)
            meter = metrics.get_meter("recommendation_meter")

            # Create an Observable Gauge for drift_score with updated description
            def drift_score_callback(options: CallbackOptions):
                return [Observation(value=self.signal.get_current_value(),
                                    attributes={})]

            meter.create_observable_gauge(
                name="drift_score",
                description="Drift score calculated using the Hellinger distance statistical test",
                callbacks=[drift_score_callback],
            )

            # Start the Prometheus HTTP server
            start_http_server(port=9464)
            logger.info("Prometheus metrics server started on port 9464")
        except Exception as e:
            logger.error(
                f"Failed to start Prometheus metrics server: {traceback.format_exc()}"
            )


class RecommendationService(demo_pb2_grpc.RecommendationServiceServicer):
    def __init__(self, ml_model, signal):
        self.ml_model = ml_model
        self.signal = signal

    def ListRecommendations(self, request, context):
        max_responses = 5

        response = demo_pb2.ListRecommendationsResponse()
        with tracer.start_span("ml-model"):
            current_span = trace.get_current_span()
            if not list(request.product_ids):
                current_span.add_event(
                    "No context for ml-model provided, returning no recommendations")
            else:
                ids = self.ml_model.find_similar_products(
                    list(request.product_ids)[0], max_responses)
                current_span.set_attribute("explanation",
                                           "these products have a similar review-profile compared to the original product")
                current_span.set_attribute("input",
                                           list(request.product_ids)[0])
                current_span.set_attribute("output", str(ids))
                response.product_ids.extend(ids)

            current_span.set_attribute("model-metric",
                                       self.ml_model.kNN.effective_metric_)
            current_span.set_attribute("number of features",
                                       self.ml_model.kNN.n_features_in_)

        self.signal.set_current_value(
            hellinger_stat_test(self.ml_model.ratings["rating"],
                                self.ml_model.ratings_test["rating"],
                                ColumnType.Numerical, 0.1).drift_score)
        return response

    # The rest of the class remains the same
    def Check(self, request, context):
        return health_pb2.HealthCheckResponse(
            status=health_pb2.HealthCheckResponse.SERVING)

    def Watch(self, request, context):
        return health_pb2.HealthCheckResponse(
            status=health_pb2.HealthCheckResponse.UNIMPLEMENTED)


if __name__ == "__main__":
    logger.info("initializing recommendationservice")

    # Initialize the Signal instance
    signal = Signal("drift_score_attribute")

    # Initialize telemetry with the signal
    telemetry = Telemetry(signal)

    # Initialize the ML Model
    ml_model = MLModel()



    # configure connections
    port = os.environ.get('PORT', "8080")
    catalog_addr = os.environ.get('PRODUCT_CATALOG_SERVICE_ADDR', '')
    if catalog_addr == "":
        raise Exception(
            'PRODUCT_CATALOG_SERVICE_ADDR environment variable not set')
    logger.info("product catalog address: " + catalog_addr)
    channel = grpc.insecure_channel(catalog_addr)
    product_catalog_stub = demo_pb2_grpc.ProductCatalogServiceStub(channel)

    # create gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # add class to gRPC server
    service = RecommendationService(ml_model, signal)
    demo_pb2_grpc.add_RecommendationServiceServicer_to_server(service, server)
    health_pb2_grpc.add_HealthServicer_to_server(service, server)

    # start server
    logger.info("listening on port: " + port)
    server.add_insecure_port('[::]:' + port)
    server.start()

    # keep alive
    try:
        while True:
            time.sleep(10000)
    except KeyboardInterrupt:
        server.stop(0)
