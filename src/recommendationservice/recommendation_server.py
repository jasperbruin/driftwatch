#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient, GrpcInstrumentorServer
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
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

def initStackdriverProfiling():
  project_id = None
  try:
    project_id = os.environ["GCP_PROJECT_ID"]
  except KeyError:
    # Environment variable not set
    pass

  for retry in range(1,4):
    try:
      if project_id:
        googlecloudprofiler.start(service='recommendation_server', service_version='1.0.0', verbose=0, project_id=project_id)
      else:
        googlecloudprofiler.start(service='recommendation_server', service_version='1.0.0', verbose=0)
      logger.info("Successfully started Stackdriver Profiler.")
      return
    except (BaseException) as exc:
      logger.info("Unable to start Stackdriver Profiler Python agent. " + str(exc))
      if (retry < 4):
        logger.info("Sleeping %d seconds to retry Stackdriver Profiler agent initialization"%(retry*10))
        time.sleep (1)
      else:
        logger.warning("Could not initialize Stackdriver Profiler after retrying, giving up")
  return

class RecommendationService(demo_pb2_grpc.RecommendationServiceServicer):
    def ListRecommendations(self, request, context):
        max_responses = 5
        
        response = demo_pb2.ListRecommendationsResponse()
        with tracer.start_span("ml-model"):
          current_span = trace.get_current_span()
          if not list(request.product_ids):
             current_span.add_event("No context for ml-model provided, returning no recommendations")
          else:
             ids = find_similar_products(list(request.product_ids)[0], X, max_responses)
             current_span.set_attribute("explanation", "these products have a similar review-profile compared to the original product")
             current_span.set_attribute("input", list(request.product_ids)[0])
             current_span.set_attribute("output", str(ids))
             response.product_ids.extend(ids)

          current_span.set_attribute("model-metric", kNN.effective_metric_)
          current_span.set_attribute("number of features", kNN.n_features_in_)

        signal.set_current_value(hellinger_stat_test(ratings["rating"],ratings_test["rating"],ColumnType.Numerical,0.1).drift_score)
        return response
    
    #def transferLabels(self, request, context):
    #   print(request.correct_rec)
    #  return

    def Check(self, request, context):
        return health_pb2.HealthCheckResponse(
            status=health_pb2.HealthCheckResponse.SERVING)

    def Watch(self, request, context):
        return health_pb2.HealthCheckResponse(
            status=health_pb2.HealthCheckResponse.UNIMPLEMENTED)

def create_matrix(df):
	
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
	

def find_similar_products(product_id, X, k, show_distance=False):

  neighbour_ids = []

  product_ind = product_mapper[product_id]
  product_vec = X[product_ind]
  product_vec = product_vec.reshape(1,-1)
  neighbour = kNN.kneighbors(product_vec, return_distance=show_distance)

  for i in range(0,k):
    n = neighbour.item(i)
    neighbour_ids.append(product_inv_mapper[n])

  neighbour_ids.pop(0)

  return neighbour_ids

if __name__ == "__main__":
    logger.info("initializing recommendationservice")

    try:
      if "DISABLE_PROFILER" in os.environ:
        raise KeyError()
      else:
        logger.info("Profiler enabled.")
        initStackdriverProfiling()
    except KeyError:
        logger.info("Profiler disabled.")

    try:
      grpc_client_instrumentor = GrpcInstrumentorClient()
      grpc_client_instrumentor.instrument()
      grpc_server_instrumentor = GrpcInstrumentorServer()
      grpc_server_instrumentor.instrument()
      if os.environ["ENABLE_TRACING"] == "1":
        resource = Resource(attributes={
           "service.name": "recommendationservice"
        })
        trace.set_tracer_provider(TracerProvider(resource=resource))
        otel_endpoint = os.getenv("COLLECTOR_SERVICE_ADDR", "localhost:4317")
        trace.get_tracer_provider().add_span_processor(
          BatchSpanProcessor(
              OTLPSpanExporter(
              endpoint = otel_endpoint,
              insecure = True
            )
          )
        )
        #additional
        tracer = trace.get_tracer("Recommendation")
    except (KeyError, DefaultCredentialsError):
        logger.info("Tracing disabled.")
    except Exception as e:
        logger.warn(f"Exception on Cloud Trace setup: {traceback.format_exc()}, tracing disabled.") 

    #MV: prepare Metric export
    reader = PeriodicExportingMetricReader(
      OTLPMetricExporter(endpoint="otelcollector:4317", insecure=True), 5
    )
    meterProvider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(meterProvider)
    meter = metrics.get_meter("recommendation_meter")

    #MV: observable class for drift score export
    class Signal:
      current_value = 0
      def __init__(self, attribute):
          self.attribute = attribute

      def set_current_value(self, i):
          self.current_value = i
      
      def get_current_value(self):
          return self.current_value
    
    #MV: init class for observation
    signal = Signal("drift_score_attribute")

    #MV: callback function for gauge export
    def read_gauge(options: CallbackOptions):
          yield Observation(signal.get_current_value(), {"attribute": signal.attribute})
    
    ddb_gauge = meter.create_observable_gauge("drift_score",[read_gauge])

    #MV: prepare ML Model
    ratings = pd.read_json('ratings.json', orient='records')
    ratings_test = pd.read_json('ratings_small.json', orient='records')
    X, user_mapper, product_mapper, user_inv_mapper, product_inv_mapper = create_matrix(ratings)

    kNN = NearestNeighbors(n_neighbors=5, algorithm="brute", metric="cosine")
    kNN.fit(X)


    # configure connections
    port = os.environ.get('PORT', "8080")
    catalog_addr = os.environ.get('PRODUCT_CATALOG_SERVICE_ADDR', '')
    if catalog_addr == "":
        raise Exception('PRODUCT_CATALOG_SERVICE_ADDR environment variable not set')
    logger.info("product catalog address: " + catalog_addr)
    channel = grpc.insecure_channel(catalog_addr)
    product_catalog_stub = demo_pb2_grpc.ProductCatalogServiceStub(channel)

    # create gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # add class to gRPC server
    service = RecommendationService()
    demo_pb2_grpc.add_RecommendationServiceServicer_to_server(service, server)
    health_pb2_grpc.add_HealthServicer_to_server(service, server)

    # start server
    logger.info("listening on port: " + port)
    server.add_insecure_port('[::]:'+port)
    server.start()

    # keep alive
    try:
         while True:
            time.sleep(10000)
    except KeyboardInterrupt:
            server.stop(0)
