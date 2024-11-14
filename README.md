# DriftWatch: A Microservice-based Drift Detection Framework

![Continuous Integration](https://github.com/DriftWatch/CI-Badge.svg)

**DriftWatch** is a cloud-native microservices framework designed to explore, implement, and test data and concept drift detection in machine learning systems. It provides a scalable, modular architecture to integrate drift detection techniques into production pipelines, enabling real-time observability and adaptive decision-making.

## About DriftWatch

This framework focuses on:

- **Data Drift Detection**: Monitoring feature distribution changes that could degrade model performance.
- **Concept Drift Detection**: Identifying shifts in the underlying target variable or relationships.
- **Integration with MLOps Pipelines**: Seamlessly integrating drift detection in CI/CD workflows.
- **Adaptability**: Supporting retraining, re-tuning, or model switching strategies.

## Architecture

DriftWatch employs a microservices architecture with multiple modular components for real-time data transformation, drift monitoring, and visualization. Each service is designed to work independently, communicating over gRPC for high performance and scalability.

![Architecture of DriftWatch](/docs/img/architecture-diagram.png)

### Microservices Overview

| Service                                              | Language      | Description                                                                                   |
| ---------------------------------------------------- | ------------- | --------------------------------------------------------------------------------------------- |
| [frontend](/src/frontend)                           | Go            | User interface for drift monitoring dashboards and analytics.                                 |
| [dataprocessor](/src/cartservice)                   | Python        | Handles feature preprocessing and streaming data integration.                                 |
| [driftdetector](/src/erbdetector)                   | Python        | Core service for drift detection algorithms (statistical and machine learning methods).       |
| [alertmanager](/src/emailservice)                   | Python        | Sends notifications when drift thresholds are breached.                                       |
| [recommendationservice](/src/recommendationservice) | Python        | Suggests actions based on detected drifts, such as retraining or hyperparameter optimization. |

## Key Features

1. **Pluggable Drift Detection Algorithms**:
   - Statistical Process Control (SPC)
   - Ensemble-based methods
   - Adaptive sliding window techniques
   - Concept-specific approaches

2. **Visualization and Reporting**:
   - Drift heatmaps
   - Real-time dashboards

3. **Extensible Framework**:
   - Add custom algorithms with minimal configuration.
   - Integrates with CI/CD pipelines via Kubernetes or Terraform.

---

## Deployment Options

DriftWatch supports both **local** and **cloud** deployment to provide flexibility for development and production environments.

### Quickstart with Skaffold for Local Deployment

DriftWatch supports **Skaffold** to simplify the build and deployment process, reducing dependencies like GDrive setup.

#### Prerequisites

- Kubernetes cluster (e.g., Minikube, Kind)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- [Skaffold](https://skaffold.dev/docs/install/) installed locally

#### Build and Deploy Using Skaffold

1. Clone the repository:

   ```bash
   git clone https://github.com/your-org/driftwatch.git
   cd driftwatch
   ```

2. Ensure Skaffold is installed and configured:

   ```bash
   skaffold version
   ```

3. Build and deploy DriftWatch:

   ```bash
   skaffold dev
   ```

   - The `skaffold.yaml` configuration ensures that all services are built and deployed in the correct order.
   - By default, Skaffold uses your local Docker environment to build images and deploys them to your Kubernetes cluster.

4. Access the frontend service:

   Use `kubectl` to fetch the external IP of the frontend:

   ```bash
   kubectl get service frontend-external | awk '{print $4}'
   ```

   Visit `http://<EXTERNAL_IP>` in your browser.

#### Alternative Deployment Profiles

- **Google Cloud Build (GCB)**: Build images using Google Cloud Build:
  ```bash
  skaffold run -p gcb
  ```
- **Debugging**: Enable debugging for cartservice:
  ```bash
  skaffold debug
  ```
- **Network Policies**: Deploy with Kubernetes network policies:
  ```bash
  skaffold run -p network-policies
  ```

#### Cleaning Up

To remove all resources:

```bash
skaffold delete
```

---

### Quickstart with GKE for Cloud Deployment

1. Clone the repository:

   ```bash
   git clone https://github.com/your-org/driftwatch.git
   cd driftwatch
   ```

2. Set up your Google Cloud project:

   ```bash
   export PROJECT_ID=<PROJECT_ID>
   export REGION=<REGION>
   gcloud services enable container.googleapis.com \
       --project=${PROJECT_ID}
   ```

3. Deploy DriftWatch:

   ```bash
   gcloud container clusters create-auto driftwatch-cluster \
       --region=${REGION} --project=${PROJECT_ID}

   kubectl apply -f ./release/kubernetes-manifests.yaml
   ```

4. Access the dashboard:

   ```bash
   kubectl get service frontend-external | awk '{print $4}'
   ```

5. Visit `http://<EXTERNAL_IP>` in your browser.

---

## Documentation

- [Development Guide](/docs/development-guide.md): Run and develop locally.
- [Integration Guide](/docs/integration-guide.md): Add DriftWatch to existing pipelines.

## Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to get involved.

## License

This project is licensed under the terms of the [Apache 2.0 License](LICENSE).