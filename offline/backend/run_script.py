import paramiko
import re,time
from utils import send_to_console
import logging
logging.basicConfig(level=logging.DEBUG)
paramiko.util.log_to_file("paramiko.log")


# Replace with your SSH credentials
host = "192.168.1.13"  # Replace with the AP's IP address
port = 22
username = "fun_network"
password = "fun_network"
get_macs_command = "show wireless client summary"

def execute_ssh_command(ssh_client, command):
    stdin, stdout, stderr = ssh_client.exec_command(command)
    output = stdout.read().decode()
    return output

ssh = paramiko.SSHClient()
ssh.load_system_host_keys()

try:
    print(f"Connecting to {host}...")
    ssh.connect(hostname=host, port=port, username=username, password=password, look_for_keys=False)
    print("Connection established.")

    # Run the initial command to get MAC addresses
    macs_output = execute_ssh_command(ssh, get_macs_command)

    # Regular expression to match MAC addresses
    mac_regex = re.compile(r'^\s*([0-9a-fA-F]{4}\.[0-9a-fA-F]{4}\.[0-9a-fA-F]{4})\s+APA00F', re.MULTILINE)

    # Find all MAC addresses in the string
    mac_addresses = mac_regex.findall(macs_output)

    for mac in mac_addresses:
        rssi_command = f"show wireless client mac-address {mac.upper()} detail | include Radio Signal Strength Indicator"
        rssi_output = execute_ssh_command(ssh, rssi_command)
        print(rssi_output)

    print(f"Connection to {host} closed.")

finally:
    ssh.close()
