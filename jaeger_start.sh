#!/bin/bash

# Function to check if kubectl is installed
check_kubectl_installed() {
    if ! command -v kubectl &> /dev/null; then
        echo "kubectl is not installed. Please install it and try again."
        exit 1
    fi
}

# Function to start port forwarding
start_port_forwarding() {
    echo "Starting port forwarding for Jaeger dashboard..."
    kubectl port-forward service/jaeger 16686:16686 &
    PORT_FORWARD_PID=$!

    # Allow port forwarding to start
    sleep 3
}

# Function to open Jaeger dashboard
open_dashboard() {
    echo "Opening Jaeger dashboard in your default browser..."
    if command -v xdg-open &> /dev/null; then
        xdg-open http://localhost:16686
    elif command -v open &> /dev/null; then
        open http://localhost:16686
    else
        echo "Please manually open your browser and navigate to http://localhost:16686"
    fi
}

# Function to clean up port forwarding
cleanup() {
    echo "Stopping port forwarding..."
    kill $PORT_FORWARD_PID
    exit
}

# Check if kubectl is installed
check_kubectl_installed

# Trap Ctrl+C to clean up
trap cleanup SIGINT

# Start port forwarding
start_port_forwarding

# Open the dashboard
open_dashboard

# Wait for user to exit script
wait $PORT_FORWARD_PID
