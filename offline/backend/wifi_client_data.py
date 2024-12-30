import sys
import re
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_wifi_client_data() -> dict:
    """
    Queries multiple Pico W devices and retrieves Wi-Fi scan data.
    
    Args:
        pico_ips (list): List of IP addresses of the Pico W devices.
    
    Returns:
        dict: A dictionary containing Wi-Fi scan data keyed by Pico IP address.
    """


    pico_ips = [
        "192.168.0.18",
    ]

    results = {}

    for pico_ip in pico_ips:
        try:
            logging.info(f"Querying Pico W at {pico_ip}...")
            response = requests.get(f"http://{pico_ip}/scan", timeout=5)  # 5-second timeout
            if response.status_code == 200:
                wifi_data = response.json()
                results[pico_ip] = wifi_data
                logging.info(f"Received data from {pico_ip}")
            else:
                logging.warning(f"Failed to fetch data from {pico_ip}, Status Code: {response.status_code}")
                results[pico_ip] = {"error": f"Status code {response.status_code}"}
        except requests.exceptions.RequestException as e:
            logging.error(f"Error querying {pico_ip}: {e}")
            results[pico_ip] = {"error": str(e)}
    
    return results
