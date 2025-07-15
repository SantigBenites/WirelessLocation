import sys
import re
import logging
import requests
import subprocess

# Configure logging
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

import concurrent.futures
import requests
import logging

def fetch_single_pico(pico_ip: str) -> tuple:
    """Helper function to fetch data from a single Pico W"""
    try:
        logging.info(f"Querying Pico W at {pico_ip}...")
        response = requests.get(f"http://{pico_ip}/scan", timeout=1)
        if response.status_code == 200:
            wifi_data = response.json()
            logging.info(f"Received data from {pico_ip}")
            return (pico_ip, wifi_data)
        else:
            logging.warning(f"Failed to fetch data from {pico_ip}, Status Code: {response.status_code}")
            return (pico_ip, {"error": f"Status code {response.status_code}"})
    except requests.exceptions.RequestException as e:
        logging.error(f"Error querying {pico_ip}: {e}")
        return (pico_ip, {"error": str(e)})

def get_wifi_client_data() -> dict:
    """Fetch WiFi client data from all Pico W devices in parallel"""
    results = {}
    
    # Using ThreadPoolExecutor to parallelize the HTTP requests
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Start the load operations and mark each future with its pico_ip
        future_to_ip = {executor.submit(fetch_single_pico, pico_ip): pico_ip for pico_ip in pico_ips}
        
        for future in concurrent.futures.as_completed(future_to_ip):
            pico_ip = future_to_ip[future]
            try:
                ip, data = future.result()
                results[ip] = data
            except Exception as e:
                logging.error(f"Unexpected error processing {pico_ip}: {e}")
                results[pico_ip] = {"error": str(e)}
    
    return results


import subprocess
from concurrent.futures import ThreadPoolExecutor

def ping_ip(ip):
    """Helper function to ping a single IP"""
    try:
        subprocess.run(
            ["ping", "-c", "1", "-W", "1", ip],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        return True
    except subprocess.CalledProcessError:
        return False

def get_status():
    """Check status of all IPs in parallel"""
    # Assuming pico_names and pico_ips are defined elsewhere
    results = {}
    
    # Create a dictionary mapping names to IPs
    ip_mapping = dict(zip(pico_names, pico_ips))
    
    with ThreadPoolExecutor() as executor:
        # Submit all ping tasks at once
        future_to_name = {
            name: executor.submit(ping_ip, ip)
            for name, ip in ip_mapping.items()
        }
        
        # Collect results as they complete
        for name, future in future_to_name.items():
            results[name] = future.result()
    
    return results

