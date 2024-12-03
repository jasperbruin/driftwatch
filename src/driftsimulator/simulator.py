# src/driftsimulator/simulator.py

import requests
import time
import random
import logging
import argparse
import socket
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Update this endpoint based on where the drift detector is accessible
ERBDETECTOR_ENDPOINT = "http://erbdetector:5000"  # Internal cluster

def check_connectivity():
    """Check network connectivity to the drift detector."""
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

def simulate_drift(delay, drift_probability=0.01):
    """Simulate sending drift data to the server."""
    if not check_connectivity():
        logging.error("Network connectivity check failed. Exiting.")
        return

    logging.info(f"Starting drift simulation with delay {delay} seconds...")

    # Initial probability of generating a 1
    p = 0.5

    while True:
        # With drift_probability, change p to a new random value
        if random.random() < drift_probability:
            p = random.uniform(0, 1)
            logging.info(f"Drift occurred! New probability of 1s: {p:.2f}")

        # Generate data based on current probability p
        value = np.random.binomial(1, p)
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
    parser.add_argument("--drift_probability", type=float, default=0.01, help="Probability of drift occurring at each step (default: 0.01).")
    args = parser.parse_args()
    simulate_drift(args.delay, args.drift_probability)
