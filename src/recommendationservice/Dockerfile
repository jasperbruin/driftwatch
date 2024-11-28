# src/recommendationservice/Dockerfile
# Use Python 3.10-slim as the base image
FROM python:3.10-slim as base

# Builder stage
FROM base as builder

RUN apt-get -qq update \
    && apt-get install -y --no-install-recommends \
        wget g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install numpy from the default PyPI index
RUN pip install --no-cache-dir numpy==1.26.1

# Install PyTorch and related packages from the custom PyTorch index
RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Final stage
FROM base

# Enable unbuffered logging
ENV PYTHONUNBUFFERED=1
ENV ENABLE_TRACING=0
ENV COLLECTOR_SERVICE_ADDR=jaeger:4317

# Set the working directory
WORKDIR /recommendationservice

# Copy installed dependencies from the builder stage
COPY --from=builder /usr/local/lib/python3.10/ /usr/local/lib/python3.10/

# Copy the application code
COPY . .

# Set environment variables and expose ports
ENV PORT="8080"
EXPOSE 8080 9090 9464

# Set the entry point for the container
ENTRYPOINT ["python", "recommendation_server.py"]
