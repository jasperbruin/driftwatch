# Base image
FROM python:3.10-slim as base

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Builder stage
FROM base as builder

# Set working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install PyTorch and other dependencies directly into the image
RUN pip install --no-cache-dir \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install additional dependencies
RUN pip install --no-cache-dir markupsafe==3.0.2 Jinja2==3.1.4
RUN pip install --no-cache-dir -r requirements.txt

# Final image
FROM base

# Set working directory
WORKDIR /loadgen

# Copy installed packages from builder
COPY --from=builder /usr/local /usr/local

# Copy application code
COPY locustfile.py .
COPY products.json .
COPY product_id_mapper.json .
COPY model_formatted_fashion_cpu_v2 .

# Enable gevent support in debugger
ENV GEVENT_SUPPORT=True

# Set entrypoint
ENTRYPOINT ["locust", "--host=http://${FRONTEND_ADDR}", "--headless", "-u", "${USERS:-10}"]
