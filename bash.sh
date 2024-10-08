# Create directories
mkdir -p data/raw
mkdir -p data/processed
mkdir notebooks
mkdir -p src/data_processing
mkdir -p src/models
mkdir -p src/drift_detection
mkdir -p src/retraining
mkdir -p src/monitoring
mkdir -p src/utils
mkdir tests
mkdir scripts

# Create empty __init__.py files in src subdirectories
touch src/data_processing/__init__.py
touch src/models/__init__.py
touch src/drift_detection/__init__.py
touch src/retraining/__init__.py
touch src/monitoring/__init__.py
touch src/utils/__init__.py

# Create Dockerfile with content
cat > Dockerfile <<EOL
# Base image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set the entry point (modify as needed)
CMD ["python", "src/main.py"]
EOL

# Create requirements.txt with content
cat > requirements.txt <<EOL
pandas
numpy
scikit-learn
EOL

# Create README.md with content
cat > README.md <<EOL
# Thesis Project

This repository contains the code and experiments for my master's thesis on drift detection in end-to-end machine learning deployment.

## Project Structure

- \`data/\`: Contains raw and processed data.
- \`notebooks/\`: Jupyter notebooks for exploration and prototyping.
- \`src/\`: Source code for the project.
  - \`data_processing/\`
  - \`models/\`
  - \`drift_detection/\`
  - \`retraining/\`
  - \`monitoring/\`
  - \`utils/\`
- \`tests/\`: Unit and integration tests.
- \`scripts/\`: Helper scripts.
- \`Dockerfile\`: Docker configuration.
- \`requirements.txt\`: Project dependencies.
- \`setup.py\`: Setup script for package installation.

## Installation

Instructions on how to set up the project.

## Usage

Instructions on how to run the project.
EOL

# Create setup.py with content
cat > setup.py <<EOL
from setuptools import setup, find_packages

setup(
    name='thesis_project',
    version='0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
    ],
)
EOL
