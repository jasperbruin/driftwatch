import requests
import time
import random
import logging
import argparse
import socket

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Update this endpoint based on where erbdetector is accessible
ERBDETECTOR_ENDPOINT = "http://erbdetector:5000"  # Internal cluster
# ERBDETECTOR_ENDPOINT = "http://<node-ip>:31215"  # NodePort access

def check_connectivity():
    """Check network connectivity to erbdetector."""
    try:
        host = ERBDETECTOR_ENDPOINT.split("//")[1].split(":")[0]
        port = int(ERBDETECTOR_ENDPOINT.split(":")[-1])
        logging.debug(f"Checking connectivity to {host}:{port}...")
        socket.create_connection((host, port), timeout=5)
        logging.info(f"Successfully connected to {host}:{port}")
    except Exception as e:
        logging.error(f"Failed to connect to {host}:{port} - {e}")
        return False
    return True

def simulate_drift(delay):
    """Simulate sending drift data to the server."""
    if not check_connectivity():
        logging.error("Network connectivity check failed. Exiting.")
        return

    logging.info(f"Starting drift simulation with delay {delay} seconds...")
    while True:
        # Generate random data simulating drift (0 or 1)
        value = random.choice([0, 1])
        logging.debug(f"Generated drift value: {value}")
        try:
            logging.debug(f"Sending request to {ERBDETECTOR_ENDPOINT}/ with value={value}...")
            response = requests.get(f"{ERBDETECTOR_ENDPOINT}/", params={"value": value}, timeout=10)
            response.raise_for_status()  # Raise an error for HTTP errors
            logging.info(f"Response received: {response.status_code}, {response.json()}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Error during request: {e}")
        finally:
            logging.debug("Sleeping before next request...")
            time.sleep(delay)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate drift data.")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (default: 1.0 seconds).")
    args = parser.parse_args()
    simulate_drift(args.delay)
