# src/prometheus/Dockerfile

FROM prom/prometheus
ADD ./prometheus.yml /etc/prometheus/prometheus.yml
USER root
ENV enable-feature=otlp-write-receiver
ENV PORT "9090"
EXPOSE 9090 9464