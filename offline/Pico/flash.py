from machine import Pin
from utime import sleep
import network
import socket
import machine

pin = Pin("LED", Pin.OUT)

ssid = 'fun_network'
password = 'fun_network'

def connect():
    #Connect to WLAN
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(ssid, password)
    while wlan.isconnected() == False:
        print('Waiting for connection...')
        sleep(1)
    ip = wlan.ifconfig()[0]
    print(f'Connected on {ip}')
    return ip


try:
    ip = connect()
except KeyboardInterrupt:
    machine.reset()

while True:
    try:
        pin.toggle()
        sleep(1) # sleep 1sec
    except KeyboardInterrupt:
        break
pin.off()
