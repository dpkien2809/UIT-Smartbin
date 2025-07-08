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
lcd.write("xKhoi dong...")

def capture_image():
    image_array = picam2.capture_array()
    image = Image.fromarray(image_array).convert('RGB').resize((224,224))
    width, height = image.size
    image2 = image.crop((0,45,150,height - 10))
    image2 = image2.resize((224, 224))
    
        # Hiển thị ảnh đã crop bằng OpenCV
    image_bgr = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)
    cv2.imshow("Cropped Image", image_bgr)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    
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
    print(f"Dự đoán: {label} (chỉ số: {prediction})")
    
    # Hiển thị lên LCD
    lcd.clear()
    lcd.setCursor(0, 0)
    lcd.write("xDu doan:")
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
GPIO.setup(20, GPIO.OUT)
GPIO.setup(17, GPIO.OUT)

pwm1 = GPIO.PWM(20, 50)
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
    rotate_servo2(45)

#Send data to server
# Các biến dữ liệu để gửi
bin_name = "UIT Tòa B"
bin_id = 1
status = "online"
storage1 = 0
storage2 = 0
storage3 = 0
storage4 = 0
ram = 10
temperature = 30
lastupdate = "1746173143"

# Biến toàn cục lưu token và hạn
current_token = None
token_exp = 0  # Epoch time


def get_token():
    global current_token, token_exp

    try:
        login_url = 'http://localhost:8090/api/auth/sign-in'
        login_headers = {
            'accept': '*/*',
            'Content-Type': 'application/json'
        }
        login_data = {
            "email": "bin@gmail.com",
            "password": "12345678"
        }

        response = requests.post(login_url,headers=login_headers,json=login_data)
        response.raise_for_status()

        data = response.json()
        token = data["response"]["token"]
        # Giải mã JWT để lấy thời gian hết hạn
        decoded = jwt.decode(token, options={"verify_signature": False})
        token_exp = decoded.get("exp", 0)
        current_token = token

        print("Lấy token thành công. Hết hạn lúc:", datetime.fromtimestamp(token_exp))

    except Exception as e:
        print("Lỗi khi lấy token:", e)
        current_token = None
        token_exp = 0

def token_is_valid():
    # Trả về True nếu token còn hiệu lực trên 5 phút
    buffer_seconds = 300  # 5 phút
    now = int(datetime.now(timezone.utc).timestamp())
    return current_token is not None and (token_exp - now) > buffer_seconds

def send_databin():
    try:
        databins_url = 'http://35.213.129.74:30600/api/databins'
        databins_headers = {
            'accept': '*/*',
            'Authorization': f'Bearer {current_token}',
            'Content-Type': 'application/json'
        }
        print(databins_headers)
        databins_data = {
            "name": bin_name,
            "idbin": bin_id,
            "status": status,
            "storage1": storage1,
            "storage2": storage2,
            "storage3": storage3,
            "storage4": storage4,
            "ram": ram,
            "temperature": temperature,
            "lastupdate": lastupdate
        }
        print(databins_data)
        response = requests.post(databins_url, headers=databins_headers, json=databins_data)
        response.raise_for_status()
        
        print("Gửi dữ liệu thành công!")
        print(response.json())

    except Exception as e:
        print("Lỗi khi gửi dữ liệu:", e)

# Hàm lấy nhiệt độ và RAM
def get_cpu_temperature():
    with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
        temp = int(f.read()) / 1000.0
    return temp


# ==============================
# MAIN LOOP
# ==============================

try:
    while True:
        distance = get_distance()
        print(f"Khoảng cách đo được: {distance:.2f} cm")

        if distance < 25:
            print("Phát hiện vật thể gần!")
            picam2.start()
            time.sleep(1)
            prediction = predict_image()
            picam2.stop()
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
            
            max_height = 68  # cm
            filled_percent = ((max_height - distance2) / max_height) * 100
            filled_percent = round(filled_percent) 
            #filled_percent = 0
            time.sleep(1)
            reset_servo1()
            time.sleep(1)
            reset_servo2()
            picam2.stop()

            
            # gán giá trị cho các biến gửi lên server
            if prediction == 0: 
                storage1 = filled_percent
            if prediction == 1:    
                storage2 = filled_percent
            if prediction == 2:    
                storage3 = filled_percent
            if prediction == 3:
                storage4 = filled_percent    
            
            mem = psutil.virtual_memory()
            ram = round((mem.used / mem.total)*100)
            temperature = round(get_cpu_temperature())
            lastupdate = str(int(time.time()))

            # Send data to server
            print("\n=== Bắt đầu chu kỳ mới ===")
            if not token_is_valid():
                print("Token hết hạn hoặc sắp hết hạn → lấy mới.")
                get_token()
            else:
                print("Token vẫn còn hiệu lực.")

            if current_token:
                send_databin()
            else:
                print("Không có token hợp lệ để gửi dữ liệu.")

            time.sleep(20)
        else:
            print("Không có vật thể gần. Đợi 1 giây...")
            time.sleep(1)
            
except KeyboardInterrupt:
    print("Dừng chương trình...")
    picam2.stop()
    pwm1.stop()
    pwm2.stop()
    GPIO.cleanup()
