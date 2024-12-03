FROM python:3.10-slim

WORKDIR /driftsimulator

COPY simulator.py .

RUN pip install numpy requests flask prometheus_client scipy


ENTRYPOINT ["python", "simulator.py"]
