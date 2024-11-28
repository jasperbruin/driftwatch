#!/bin/bash

# Script to port-forward and curl metrics from otelcollector

# Pod name
POD_NAME=$(kubectl get pods -l app=otelcollector -o jsonpath='{.items[0].metadata.name}')

# Port forward in the background
echo "Setting up port forwarding to pod $POD_NAME..."
kubectl port-forward "$POD_NAME" 8888:8888 > /dev/null 2>&1 &
PORT_FORWARD_PID=$!

# Give port-forward some time to establish
sleep 3

# Check if port forwarding succeeded
if ! nc -z localhost 8888; then
    echo "Port forwarding failed. Please check the pod status and configuration."
    kill $PORT_FORWARD_PID
    exit 1
fi

# Curl the /metrics endpoint
echo "Fetching metrics from http://localhost:8888/metrics..."
curl http://localhost:9090/api/v1/write

# Cleanup: Kill port-forward process
echo "Cleaning up..."
kill $PORT_FORWARD_PID

echo "Done."
