import network
import socket
import json
import time
import machine

# -- Configuration --
SSID = "fun_network"
PASSWORD = "fun_network"
WIFI_CHECK_INTERVAL = 10     # seconds between connectivity checks
MAX_BACKOFF = 32             # max seconds for exponential backoff
LOG = True

led = machine.Pin("LED", machine.Pin.OUT)

# -- Watchdog Timer (auto-reset if stalled) --
wdt = machine.WDT(timeout=8000)  # 8 seconds


led.on()
time.sleep(1)
led.off()

# -- Wi-Fi Setup --
wlan = network.WLAN(network.STA_IF)
wlan.active(True)

# Pre-allocated buffer for scan results
time.sleep(1)
_scan_buffer = []

def ensure_connection():
    
    #Connect to WLAN
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(SSID, PASSWORD)
    while wlan.isconnected() == False:
        if LOG : print('Waiting for connection...')
        time.sleep(1)
    if LOG : print(wlan.ifconfig())

# Initial connect
try:
    if not wlan.isconnected():
        if LOG : print("hee to Wi-Fi...")
        ensure_connection()
    if LOG : print("Connected to Wi-Fi, IP:", wlan.ifconfig()[0])
except Exception as e:
    if LOG : print("Failed to set up Wi-Fi:", e)

led.off()
led.on()

# Reusable HTTP server socket with timeout
try:
    addr = socket.getaddrinfo("0.0.0.0", 80)[0][-1]
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.settimeout(5)
    server.bind(addr)
    server.listen(1)
    if LOG : print("Server listening on", addr)
except Exception as e:
    if LOG : print("Failed to start server:", e)
    raise

# Wi-Fi scan function with reusable buffer
def scan_wifi():
    _scan_buffer.clear()
    try:
        networks = wlan.scan()
        for ssid, bssid, channel, rssi, security, hidden in networks:
            _scan_buffer.append({
                "SSID": ssid.decode() if ssid else "(hidden)",
                "BSSID": ":".join(f"{b:02x}" for b in bssid),
                "Channel": channel,
                "RSSI": rssi,
                "Security": security,
                "Hidden": hidden
            })
    except OSError as e:
        if LOG : print("Scan error:", e)
    return _scan_buffer

last_wifi_check = time.time()

# -- Main loop --
while True:
    # Feed the watchdog to avoid reset
    wdt.feed()

    # Periodic Wi-Fi health check
    now = time.time()
    if now - last_wifi_check > WIFI_CHECK_INTERVAL:
        last_wifi_check = now
        if not wlan.isconnected():
            ensure_connection()

    try:
        conn, addr = server.accept()
    except OSError:
        continue

    try:
        if LOG : print("Connection from", addr)
        conn.settimeout(5)
        request = conn.recv(1024).decode('utf-8')
        if LOG : print("Request:", request)

        if "GET /scan" in request:
            led.off()
            results = scan_wifi()
            payload = json.dumps(results)
            headers = (
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {len(payload)}\r\n"
                "\r\n"
            )
            conn.send(headers)
            conn.send(payload)
            led.on()
        else:
            error_payload = json.dumps({"error": "Invalid endpoint. Use /scan."})
            headers = (
                "HTTP/1.1 404 Not Found\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {len(error_payload)}\r\n"
                "\r\n"
            )
            conn.send(headers)
            conn.send(error_payload)

    except OSError as e:
        if LOG : print("Connection error:", e)
    finally:
        try:
            conn.close()
        except:
            pass