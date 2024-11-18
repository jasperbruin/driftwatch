#!/bin/bash

# Get the pod name starting with 'loadgenerator'
POD_NAME=$(kubectl get pods --no-headers | grep ^loadgenerator | awk '{print $1}')

# Check if a pod was found
if [ -z "$POD_NAME" ]; then
  echo "No pod found starting with 'loadgenerator'"
  exit 1
fi

# Print the logs of the loadgenerator pod
echo "kubectl logs "$POD_NAME" output:"
kubectl logs "$POD_NAME"
printf "\n\n"
echo "kubectl describe pod "$POD_NAME" | grep USERS output:"
kubectl describe pod "$POD_NAME" | grep USERS
