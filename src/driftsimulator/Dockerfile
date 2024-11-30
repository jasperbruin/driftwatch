FROM python:3.10-slim

WORKDIR /driftsimulator

COPY simulator.py .

RUN pip install requests

ENTRYPOINT ["python", "simulator.py"]
