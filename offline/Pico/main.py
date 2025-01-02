import network
import socket
import json
import time
import machine

# Configure Wi-Fi connection
SSID = "fun_network"
PASSWORD = "fun_network"

wlan = network.WLAN(network.STA_IF)
wlan.active(True)

# Connect to the specified Wi-Fi network
if not wlan.isconnected():
    print("Connecting to Wi-Fi...")
    wlan.connect(SSID, PASSWORD)
    while not wlan.isconnected():
        time.sleep(1)
print("Connected to Wi-Fi")
print("IP Address:", wlan.ifconfig()[0])

led = machine.Pin("LED", machine.Pin.OUT)
led.off()
led.on()

# Function to perform Wi-Fi scan and return detailed results
def scan_wifi():
    networks = wlan.scan()  # Perform the scan
    result = []
    for ssid, bssid, channel, rssi, security, hidden in networks:
        result.append({
            "SSID": ssid.decode() if ssid else "(hidden)",
            "BSSID": ":".join(f"{b:02x}" for b in bssid),  # Format BSSID (AP MAC address)
            "Channel": channel,
            "RSSI": rssi,
            "Security": security,
            "Hidden": hidden
        })
    return result

# Start the web server
addr = socket.getaddrinfo("0.0.0.0", 80)[0][-1]
server = socket.socket()
server.bind(addr)
server.listen(1)

print("Web server running on:", addr)

# Serve JSON responses
while True:
    conn, addr = server.accept()
    print("Connection from:", addr)
    request = conn.recv(1024).decode("utf-8")
    print("Request:", request)

    if "GET /scan" in request:
        # Perform Wi-Fi scan and return JSON response
        wifi_results = scan_wifi()
        response = json.dumps(wifi_results)
        conn.send("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n")
        conn.send(response)
    else:
        # Default response for unsupported endpoints
        response = json.dumps({"error": "Invalid endpoint. Use /scan for Wi-Fi results."})
        conn.send("HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\n\r\n")
        conn.send(response)

    conn.close()
