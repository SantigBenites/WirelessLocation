import sys
import re
import logging
import requests
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

pico_ips = [
        "192.168.1.31",
        "192.168.1.32",
        "192.168.1.33",
        "192.168.1.34",
        "192.168.1.35",
        "192.168.1.36",
        "192.168.1.37",
        "192.168.1.38",
        "192.168.1.39",
        "192.168.1.30",
]

pico_names= [
    "Pico1",
    "Pico2",
    "Pico3",
    "Pico4",
    "Pico5",
    "Pico6",
    "Pico7",
    "Pico8",
    "Pico9",
    "Pico10"
]

def get_wifi_client_data() -> dict:

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


def get_status():
    
    results = {}
    for key,ip in zip(pico_names,pico_ips):
        try:
            #ping_result = subprocess.run(["ping", "-c", "1", "-W", "1", ip], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            results[key] = True
        except Exception as e:
            results[key] = False
    return results