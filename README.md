# Thesis Project

This repository contains the code and experiments for my master's thesis on drift detection in end-to-end machine learning deployment.

## Project Structure

- `data/`: Contains raw and processed data.
- `notebooks/`: Jupyter notebooks for exploration and prototyping.
- `src/`: Source code for the project.
  - `data_processing/`: Data loading and preprocessing modules.
  - `models/`: Model training and evaluation modules.
  - `drift_detection/`: Drift detection algorithms and utilities.
  - `retraining/`: Modules for retraining models upon drift detection.
  - `monitoring/`: Monitoring and observability tools.
  - `utils/`: Helper functions and utilities.
- `tests/`: Unit and integration tests.
- `scripts/`: Helper scripts for running the project.
- `reports/`: Generated reports and dashboards.
- `Dockerfile`: Docker configuration.
- `requirements.txt`: Project dependencies.
- `setup.py`: Setup script for package installation.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/thesis_project.git

# Navigate into the directory
cd thesis_project

# Create a virtual environment
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the pipeline
bash scripts/run_pipeline.sh
```

## License

This project is licensed under the MIT License.
