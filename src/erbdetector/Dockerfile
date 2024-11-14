FROM python:3.10-slim as base

FROM base as builder

RUN apt-get -qq update \
    && apt-get install -y --no-install-recommends \
        wget g++ \
    && rm -rf /var/lib/apt/lists/*

# get packages
COPY requirements.txt .
RUN pip install numpy==1.26.4
RUN pip install -r requirements.txt

FROM base

WORKDIR /erbdetector

# Grab packages from builder
COPY --from=builder /usr/local/lib/python3.10/ /usr/local/lib/python3.10/

# Add the application
COPY . .

# set listen port
ENV PORT "5000"
EXPOSE 5000
ENTRYPOINT ["python", "detector.py"]