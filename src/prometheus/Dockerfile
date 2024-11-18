# TARGET src/prometheus/Dockerfile

FROM prom/prometheus
ADD ./prometheus.yml /etc/prometheus/prometheus.yml
USER root
ENV PORT "9090"
EXPOSE 9090