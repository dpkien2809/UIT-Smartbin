from picamera2 import Picamera2, Preview
import numpy as np
from PIL import Image
import tensorflow as tf
import RPi.GPIO as GPIO
import time
from grove.display.base import *
from grove.i2c import Bus
import sys
import cv2
import requests
import time
import jwt  # PyJWT
import json
import os
import psutil
from datetime import datetime, timezone

# ==============================
# LCD CLASS (JHD1802)
# ==============================

class JHD1802(Display):
    def __init__(self, address = 0x3E):
        self._bus = Bus()
        self._addr = address
        if self._bus.write_byte(self._addr, 0):
            print("Check if the LCD {} inserted, then try again".format(self.name))
            sys.exit(1)
        self.textCommand(0x02)
        time.sleep(0.1)
        self.textCommand(0x08 | 0x04)  # display on, no cursor
        self.textCommand(0x28)

    @property
    def name(self):
        return "JHD1802"

    def type(self):
        return TYPE_CHAR

    def size(self):
        return 2, 16

    def clear(self):
        self.textCommand(0x01)

    def draw(self, data, bytes):
        return False

    def home(self):
        self.textCommand(0x02)
        time.sleep(0.2)

    def setCursor(self, row, column):
        self.textCommand((0x40 * row) + (column % 0x10) + 0x80)

    def write(self, msg):
        for c in msg:
            self._bus.write_byte_data(self._addr, 0x40, ord(c))

    def _cursor_on(self, enable):
        self.textCommand(0x0E if enable else 0x0C)

    def textCommand(self, cmd):
        self._bus.write_byte_data(self._addr, 0x80, cmd)

# ==============================
# MAIN MODEL + CAMERA + GPIO
# ==============================

interpreter = tf.lite.Interpreter(model_path="/home/kien/Downloads/EffnetB0.tflite")
interpreter.allocate_tensors()
label_names = ["Organic", "Non_recycle", "Recycle", "Toxic"]

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration(main={"size": (224, 224)}))

# Khởi tạo LCD
lcd = JHD1802()
lcd.clear()
lcd.setCursor(0, 0)
lcd.write("Khoi dong...")

def capture_image():
    image_array = picam2.capture_array()
    image = Image.fromarray(image_array).convert('RGB').resize((224,224))
    width, height = image.size
    image2 = image.crop((0,45,150,height - 10))
    image2 = image2.resize((224, 224))
    
        # Hiển thị ảnh đã crop bằng OpenCV
    image_bgr = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)
    # cv2.imshow("Cropped Image", image_bgr)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()
    
    image = np.array(image2, dtype=np.float32)
    image = np.expand_dims(image, axis=0)
    return image

def predict_image():
    image = capture_image()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data)
    label = label_names[prediction]
    print(f"Dự đoán: {label}")

    # Hiển thị lên LCD
    lcd.clear()
    lcd.setCursor(0, 0)
    lcd.write("Du doan:")
    lcd.setCursor(1, 0)
    lcd.write(label)

    return prediction

# ==============================
# GPIO, Servo, HC-SR04 setup
# ==============================

SIG_PIN = 26
SIG_PIN2 = 22  # Cảm biến sóng âm thứ hai

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(SIG_PIN, GPIO.OUT)
GPIO.setup(SIG_PIN2, GPIO.OUT)
GPIO.setup(18, GPIO.OUT)
GPIO.setup(17, GPIO.OUT)

pwm1 = GPIO.PWM(18, 50)
pwm2 = GPIO.PWM(17, 50)
pwm1.start(0)
pwm2.start(0)

def get_distance():
    GPIO.output(SIG_PIN, False)
    time.sleep(0.0002)
    GPIO.output(SIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(SIG_PIN, False)

    GPIO.setup(SIG_PIN, GPIO.IN)
    while GPIO.input(SIG_PIN) == 0:
        pulse_start = time.time()
    while GPIO.input(SIG_PIN) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    distance = round(distance, 2)
    GPIO.setup(SIG_PIN, GPIO.OUT)
    return distance

# Hàm đo khoảng cách bằng cảm biến thứ 2
def get_distance2():
    GPIO.output(SIG_PIN2, False)
    time.sleep(0.0002)
    GPIO.output(SIG_PIN2, True)
    time.sleep(0.00001)
    GPIO.output(SIG_PIN2, False)

    GPIO.setup(SIG_PIN2, GPIO.IN)
    while GPIO.input(SIG_PIN2) == 0:
        pulse_start = time.time()
    while GPIO.input(SIG_PIN2) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    distance = round(distance, 2)
    GPIO.setup(SIG_PIN2, GPIO.OUT)
    return distance

def rotate_servo1(angle):
    duty_cycle = angle / 18 + 2
    pwm1.ChangeDutyCycle(duty_cycle)
    time.sleep(1.5)
    pwm1.ChangeDutyCycle(0)

def rotate_servo2(angle):
    duty_cycle = angle / 18 + 2
    pwm2.ChangeDutyCycle(duty_cycle)
    time.sleep(1.5)
    pwm2.ChangeDutyCycle(0)

def reset_servo1():
    rotate_servo1(50)

def reset_servo2():
    rotate_servo2(110)

# ==============================
# MAIN LOOP
# ==============================

try:
    while True:
        distance = get_distance()
        print(f"Khoảng cách đo được: {distance:.2f} cm")

        if distance < 27:
            print("Phát hiện vật thể gần!")
            picam2.start()
            time.sleep(1)
            prediction = predict_image()

            if prediction == 0:
                rotate_servo1(15)
                time.sleep(1)
                rotate_servo2(180)
            elif prediction == 1:
                rotate_servo1(80)
                time.sleep(1)
                rotate_servo2(180)
            elif prediction == 2:
                rotate_servo1(135)
                time.sleep(1)
                rotate_servo2(180)
            elif prediction == 3:
                rotate_servo1(190)
                time.sleep(1)
                rotate_servo2(180)

            # Đo khoảng cách bằng cảm biến thứ hai (sau khi quay servo)
            distance2 = get_distance2()
            print(f"Khoảng cách đo bởi cảm biến thứ 2: {distance2:.2f} cm")

            time.sleep(1)
            reset_servo1()
            reset_servo2()
            picam2.stop()
        else:
            print("Không có vật thể gần. Đợi 1 giây...")
        time.sleep(1)

except KeyboardInterrupt:
    print("Dừng chương trình...")
    picam2.stop()
    pwm1.stop()
    pwm2.stop()
    GPIO.cleanup()
