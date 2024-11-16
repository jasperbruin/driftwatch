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

| Service                                              | Language      | Description                                                                                                                       |
| ---------------------------------------------------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| [frontend](/src/frontend)                           | Go            | Exposes an HTTP server to serve the website. Does not require signup/login and generates session IDs for all users automatically. |
| [cartservice](/src/cartservice)                     | C#            | Stores the items in the user's shopping cart in Redis and retrieves it.                                                           |
| [productcatalogservice](/src/productcatalogservice) | Go            | Provides the list of products from a JSON file and ability to search products and get individual products.                        |
| [currencyservice](/src/currencyservice)             | Node.js       | Converts one money amount to another currency. Uses real values fetched from European Central Bank. It's the highest QPS service. |
| [paymentservice](/src/paymentservice)               | Node.js       | Charges the given credit card info (mock) with the given amount and returns a transaction ID.                                     |
| [shippingservice](/src/shippingservice)             | Go            | Gives shipping cost estimates based on the shopping cart. Ships items to the given address (mock)                                 |
| [emailservice](/src/emailservice)                   | Python        | Sends users an order confirmation email (mock).                                                                                   |
| [checkoutservice](/src/checkoutservice)             | Go            | Retrieves user cart, prepares order and orchestrates the payment, shipping and the email notification.                            |
| [recommendationservice](/src/recommendationservice) | Python        | Recommends other products based on what's given in the cart.                                                                      |
| [adservice](/src/adservice)                         | Java          | Provides text ads based on given context words.                                                                                   |
| [loadgenerator](/src/loadgenerator)                 | Python/Locust | Continuously sends requests imitating realistic user shopping flows to the frontend.                                              |

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
   sudo snap install skaffold
   skaffold version
   ```

3. Build and deploy DriftWatch:

```bash
skaffold build
```

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

## Documentation

- [Development Guide](/docs/development-guide.md): Run and develop locally.
- [Integration Guide](/docs/integration-guide.md): Add DriftWatch to existing pipelines.

## Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to get involved.

## Acknowledgment
This project utilizes components and microservices derived from the [GoogleCloudPlatform/microservices-demo](https://github.com/GoogleCloudPlatform/microservices-demo) repository. These resources are used exclusively for academic and experimental purposes to facilitate research and exploration in drift detection methodologies.
