#!/bin/bash

# Variables
NAMESPACE="default" # Change this if your pod is in a different namespace
SERVICE_NAME="recommendationservice"
LOCAL_PORT=9464
POD_PORT=9464

# Step 1: Identify the Pod Name
echo "Fetching the pod name for service: $SERVICE_NAME..."
POD_NAME=$(kubectl get pods -n $NAMESPACE --no-headers -o custom-columns=":metadata.name" | grep $SERVICE_NAME)

if [ -z "$POD_NAME" ]; then
    echo "Error: No pod found for service $SERVICE_NAME in namespace $NAMESPACE."
    exit 1
fi

echo "Found pod: $POD_NAME"

# Step 2: Port Forward
echo "Setting up port forwarding from localhost:$LOCAL_PORT to pod $POD_NAME:$POD_PORT..."
kubectl port-forward -n $NAMESPACE $POD_NAME $LOCAL_PORT:$POD_PORT &

# Save the process ID of the background job
PORT_FORWARD_PID=$!

echo "Port forwarding established. Access the service at http://localhost:$LOCAL_PORT/metrics"
echo "To stop port forwarding, use: kill $PORT_FORWARD_PID"

# Wait for user to terminate
read -p "Press [Ctrl+C] to stop port forwarding or wait for termination..."

# Cleanup on exit
echo "Stopping port forwarding..."
kill $PORT_FORWARD_PID
echo "Port forwarding stopped."
