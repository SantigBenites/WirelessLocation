import serial
import time


def send_to_console(ser: serial.Serial, command: str, wait_time: float = 0.5):
    command_to_send = command + "\r"
    print(f"Sending command: {command_to_send}")
    ser.write(command_to_send.encode('utf-8'))
    time.sleep(wait_time)
    output = ser.read(ser.inWaiting()).decode('utf-8')
    print(output)
    return output
    
