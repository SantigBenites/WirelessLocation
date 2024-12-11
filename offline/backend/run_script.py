import sys
from netmiko import ConnectHandler
import re
import logging

# Validate input
if len(sys.argv) < 2:
    print("Usage: python3 run_script.py <button_id>")
    sys.exit(1)

button_id = sys.argv[1]

print(f"Running script for button ID: {button_id}")

# Configure logging
# logging.basicConfig(level=logging.DEBUG)

# Device configuration
device = {
    "device_type": "cisco_wlc",
    "host": "192.168.1.13",  # Replace with the AP's IP address
    "username": "fun_network",
    "password": "fun_network",
    "secret": "fun_network",  # Optional, if an enable password is needed
}

# Commands
get_macs_command = "show wireless client summary"
rssi_command_template = "show wireless client mac-address {mac} detail | include Radio Signal Strength Indicator"

# Regular expression to extract MAC addresses
mac_regex = re.compile(r'^\s*([0-9a-fA-F]{4}\.[0-9a-fA-F]{4}\.[0-9a-fA-F]{4})\s+APA00F', re.MULTILINE)

try:
    # Establish connection using netmiko
    print(f"Connecting to {device['host']}...")
    connection = ConnectHandler(**device)
    print("Connection established.")

    # Execute command to get MAC addresses
    macs_output = connection.send_command(get_macs_command)

    # Find all MAC addresses in the output
    mac_addresses = mac_regex.findall(macs_output)

    if mac_addresses:
        print("MAC Addresses found:")
        for mac in mac_addresses:
            print(mac)

            # Fetch RSSI for each MAC address
            rssi_command = rssi_command_template.format(mac=mac.upper())
            rssi_output = connection.send_command(rssi_command)
            print(f"RSSI for {mac}:")
            print(rssi_output)
    else:
        print("No MAC addresses found.")

finally:
    # Disconnect from the device
    connection.disconnect()
    print(f"Connection to {device['host']} closed.")
