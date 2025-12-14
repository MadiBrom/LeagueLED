import time
import serial

arduino = serial.Serial("COM3", 9600, timeout=1)
time.sleep(2)

arduino.write(b"10,0,255,0.")
time.sleep(2)
arduino.write(b"10,255,0,0.")
time.sleep(2)
arduino.write(b"10,0,0,255.")
time.sleep(2)
arduino.write(b"10,0,0,0.")

# blue, red, green, off