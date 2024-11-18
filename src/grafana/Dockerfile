# Use the official Grafana image as the base
FROM grafana/grafana:latest

# Add custom plugins (optional)
# Uncomment and specify plugin IDs to install them
# RUN grafana-cli plugins install grafana-piechart-panel
# RUN grafana-cli plugins install grafana-worldmap-panel

# Add custom configuration (optional)
# Copy configuration files into the appropriate locations
# COPY grafana.ini /etc/grafana/grafana.ini
# COPY dashboards /var/lib/grafana/dashboards

# Expose Grafana's default port
EXPOSE 3000

# Set environment variables (optional)
# Default admin user and password (change these for security)
ENV GF_SECURITY_ADMIN_USER=admin
ENV GF_SECURITY_ADMIN_PASSWORD=admin

# Start Grafana
CMD ["grafana-server", "--homepath=/usr/share/grafana", "--config=/etc/grafana/grafana.ini"]
