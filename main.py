import speech_recognition as sr
import RPi.GPIO as GPIO
import time
import smbus2
import requests
import json
import pandas as pd
import numpy as np
import pickle
import os
import csv
import adafruit_dht
import board
import threading
from flask import Flask, jsonify, request, send_from_directory
import sounddevice
from flask_cors import CORS
from datetime import datetime
from gevent import pywsgi


# Configuration constants
MODEL_FILE = '/home/pi/Desktop/smart_home/fan_pattern_model.pkl'
HISTORY_FILE = 'history_fans.csv'
LED_PIN = 17  # GPIO 17 for LED
DEHUMIDIFIER_PIN = 27  # GPIO 27 for dehumidifier
FAN_INA_PIN = 18  # GPIO 18 for fan INA
FAN_INB_PIN = 23  # GPIO 23 for fan INB
sensor = adafruit_dht.DHT11(board.D4)  # GPIO 4
PWM_FREQUENCY = 1000
DEFAULT_FAN_SPEED = 50
MIN_DATA_POINTS = 10
BH1750_ADDRESS = 0x23
bus = smbus2.SMBus(1)
LIGHT_THRESHOLD = 50  # Lux threshold for light control
HUMIDITY_THRESHOLD = 80  # Humidity threshold for dehumidifier
HUMIDITY_DURATION = 60  # 60 checks * 5 seconds = 300 seconds (5 minutes)

# ThingSpeak API configuration
THINGSPEAK_API_KEY = "6727KBX271S26B5E"
THINGSPEAK_URL = "https://api.thingspeak.com/update"

# Create Flask app
app = Flask(__name__)
CORS(app)
last_thingspeak_update = 0

class DeviceData:
    def __init__(self):
        self.temperature = None
        self.humidity = None
        self.light_intensity = None

    def update(self, temperature=None, humidity=None, light_intensity=None):
        if temperature is not None:
            self.temperature = temperature
        if humidity is not None:
            self.humidity = humidity
        if light_intensity is not None:
            self.light_intensity = light_intensity

    def get_data(self):
        return {
            "temperature": self.temperature,
            "humidity": self.humidity,
            "light_intensity": self.light_intensity
        }

device_data = DeviceData()

def read_light():
    bus.write_byte(BH1750_ADDRESS, 0x10)
    time.sleep(0.18)
    data = bus.read_i2c_block_data(BH1750_ADDRESS, 0x10, 2)
    lux = (data[0] << 8 | data[1]) / 1.2
    return lux

def read_temperature_and_humidity():
    try:
        temperature = sensor.temperature
        humidity = sensor.humidity
        return temperature, humidity
    except RuntimeError as e:
        return None, None

def send_data_to_thingspeak(temperature=None, humidity=None, light_intensity=None):
    global last_thingspeak_update
    global device_data
    current_time = time.time()
    device_data.update(temperature=temperature, humidity=humidity, light_intensity=light_intensity)
    """
    current_data = device_data.get_data()
    if all(value is None for value in current_data.values()):
        print("Some data not available to send to ThingSpeak.")
        return False
    
    temperature = current_data['temperature']
    humidity = current_data['humidity']
    light_intensity = current_data['light_intensity']

    if current_time - last_thingspeak_update < 15:
        print("Skipping ThingSpeak update: Too soon since last update.")
        return False
    """
    try:
        payload = {'api_key': THINGSPEAK_API_KEY}
        if temperature is not None:
            payload['field1'] = temperature
        if humidity is not None:
            payload['field2'] = humidity
        if light_intensity is not None:
            payload['field3'] = light_intensity
        if len(payload) > 1:
            response = requests.get(THINGSPEAK_URL, params=payload, timeout=10)
            if response.status_code == 200:
                last_thingspeak_update = current_time
                print(f"Data sent to ThingSpeak: Temp={temperature}, Humidity={humidity}, Light={light_intensity}")
                return True
            else:
                print(f"Failed to send to ThingSpeak. Status code: {response.status_code}")
                return False
        else:
            print("No data to send to ThingSpeak.")
            return False
    except Exception as e:
        print(f"Error sending to ThingSpeak: {e}")
        return False

class FeatureExtractor:
    @staticmethod
    def extract_features(df):
        features = {
            'avg_temp': df['temperature'].mean(),
            'min_temp': df['temperature'].min(),
            'max_temp': df['temperature'].max()
        }
        on_temps = []
        off_temps = []
        durations = []
        fan_status = df['fan_status'].values
        temps = df['temperature'].values
        current_duration = 0
        is_on = False
        for i in range(1, len(df)):
            if fan_status[i-1] == 0 and fan_status[i] == 1:
                on_temps.append(temps[i])
                is_on = True
            elif fan_status[i-1] == 1 and fan_status[i] == 0:
                off_temps.append(temps[i])
                if current_duration > 0:
                    durations.append(current_duration * 5)
                is_on = False
                current_duration = 0
            if is_on:
                current_duration += 1
        if is_on and current_duration > 0:
            durations.append(current_duration * 5)
        features['num_on'] = len(on_temps)
        features['avg_on_temp'] = np.mean(on_temps) if on_temps else np.nan
        features['avg_off_temp'] = np.mean(off_temps) if off_temps else np.nan
        features['on_time_ratio'] = df['fan_status'].mean()
        return features

class FanPatternPredictor:
    @staticmethod
    def predict():
        try:
            df = pd.read_csv(HISTORY_FILE)
            if len(df) < MIN_DATA_POINTS:
                return {
                    'temp_threshold_on': None,
                    'temp_threshold_off': None,
                    'avg_duration': None,
                    'error': f'At least {MIN_DATA_POINTS} data points are needed for prediction'
                }
            with open(MODEL_FILE, 'rb') as f:
                model = pickle.load(f)
            features = FeatureExtractor.extract_features(df)
            feature_values = np.array(list(features.values()))
            feature_values = np.nan_to_num(feature_values, nan=np.nanmean(feature_values))
            prediction = model.predict([feature_values])[0]
            return {
                'temp_threshold_on': round(prediction[0], 1),
                'temp_threshold_off': round(prediction[1], 1),
                'avg_duration': round(prediction[2], 1)
            }
        except Exception as e:
            return {
                'error': f'Prediction error: {str(e)}'
            }

import mysql.connector

class DeviceLog:
    @staticmethod
    def logActivity(device_name, status, time=None):
        try:
            # Establish connection to the MySQL database
            connection = mysql.connector.connect(
                host="192.168.1.81",
                user="root",
                password="your_password",
                database="aiot"
            )
            cursor = connection.cursor()

            # Ensure the table exists, create it if not
            create_table_query = """
                CREATE TABLE IF NOT EXISTS aiot_device_control_logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    device_name VARCHAR(255) NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    time DATETIME NOT NULL
                )
            """
            cursor.execute(create_table_query)

            # Ensure time is set to the current time if not provided
            if time is None:
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Insert log into the database
            query = """
                INSERT INTO aiot_device_control_logs (device_name, status, time)
                VALUES (%s, %s, %s)
            """
            cursor.execute(query, (device_name, status, time))
            connection.commit()
            print(f"[LOG] Activity logged: Device={device_name}, Status={status}, Time={time}")
        except mysql.connector.Error as err:
            print(f"[ERROR] MySQL Error: {err}")
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    @staticmethod
    def getLogs(limit=20):
        try:
            # Establish connection to the MySQL database
            connection = mysql.connector.connect(
                host="192.168.1.81",
                user="root",
                password="your_password",
                database="aiot"
            )
            cursor = connection.cursor(dictionary=True)

            # Query to fetch logs with a limit
            query = "SELECT * FROM aiot_device_control_logs ORDER BY time DESC LIMIT %s"
            cursor.execute(query, (limit,))
            logs = cursor.fetchall()
            print("[LOG] Retrieved logs successfully")
            return logs
        except mysql.connector.Error as err:
            print(f"[ERROR] MySQL Error: {err}")
            return []
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

class VoiceCommandProcessor:
    def __init__(self, device_controller):
        self.device_controller = device_controller
        self.recognizer = sr.Recognizer()
        self.running = True
        self.microphone = None
        try:
            self.microphone = sr.Microphone(device_index=1, sample_rate=44100, chunk_size=1024)
        except Exception as e:
            print(f"Error initializing microphone: {e}")
    
    def generate_content(self, messages):
        prompt_json = {
            "turn_on_light": "The user's voice text includes the intention to turn on the light",
            "turn_off_light": "The user's voice text includes the intention to turn off the light",
            "turn_on_dehumidifier": "The user's voice text includes the intention to turn on the dehumidifier",
            "turn_off_dehumidifier": "The user's voice text includes the intention to turn off the dehumidifier",
            "turn_on_fan": "The user's voice text includes the intention to turn on the fan",
            "turn_off_fan": "The user's voice text includes the intention to turn off the fan",
            "reverse_fan": "The user's voice text includes the intention to reverse the fan"
        }
        default_prompt = "You are a smart home voice assistant that can recognize user's voice text and execute corresponding commands. The text language could be Chinese, English, or other languages, but this doesn't affect your understanding. Please carefully analyze the user's intention and determine "
        for intent, description in prompt_json.items():
            default_prompt += f"if it {description}, if yes, please only reply \"{intent}\"; "
        default_prompt += "if none of the above, please reply \"unrecognized\". Please don't explain your judgment process, just provide the final result. Here is the user's voice text:"
        messages = default_prompt + "\"\"\"" + messages + "\"\"\""
        LLM_API_URL = "https://api-proxy.me/gemini/v1beta/models/gemini-2.0-flash:generateContent?key=AIzaSyB0ovgUeYF2R4GlT0G9bXGniA9u4iik22U"
        if not LLM_API_URL:
            return None
        try:
            gemini_request = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": messages
                            }
                        ]
                    }
                ]
            }
            response = requests.post(
                LLM_API_URL, 
                headers={"Content-Type": "application/json"}, 
                json=gemini_request,
                timeout=30
            )
            if response.status_code != 200:
                return None
            result = response.json()
            if result and "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if parts and len(parts) > 0 and "text" in parts[0]:
                        return parts[0]["text"]
            if "text" in result:
                return result["text"]
            return None
        except (json.JSONDecodeError, requests.exceptions.Timeout, 
                requests.exceptions.RequestException, Exception):
            return None
    
    def process_voice_command(self, audio_text):
        text = self.generate_content(audio_text)
        print(f"Received voice command: {text}")
        if text and "turn_on_light" in text:
            self.device_controller.control_led("on")
            return {"status": "success", "message": "Light turned on"}
        elif text and "turn_off_light" in text:
            self.device_controller.control_led("off")
            return {"status": "success", "message": "Light turned off"}
        elif text and "turn_on_dehumidifier" in text:
            self.device_controller.control_dehumidifier("on")
            return {"status": "success", "message": "Dehumidifier turned on"}
        elif text and "turn_off_dehumidifier" in text:
            self.device_controller.control_dehumidifier("off")
            return {"status": "success", "message": "Dehumidifier turned off"}
        elif text and "turn_on_fan" in text:
            self.device_controller.control_fan("forward", DEFAULT_FAN_SPEED)
            return {"status": "success", "message": "Fan turned on"}
        elif text and "turn_off_fan" in text:
            self.device_controller.control_fan("stop")
            return {"status": "success", "message": "Fan turned off"}
        elif text and "reverse_fan" in text:
            self.device_controller.control_fan("reverse", DEFAULT_FAN_SPEED)
            return {"status": "success", "message": "Fan reversed"}
    
    def listen_for_commands(self):
        if self.microphone is None:
            print("Microphone not initialized properly. Cannot listen for commands.")
            return
        while self.running:
            try:
                with self.microphone as source:
                    print("Say a command (e.g., 'turn on LED', 'turn off fan'):")
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                    try:
                        audio = self.recognizer.listen(source, timeout=5)
                        print("Recording done, processing...")
                        text = self.recognizer.recognize_google(audio, language="en-US")
                        print(f"You said: {text}")
                        self.process_voice_command(text)
                    except sr.WaitTimeoutError:
                        print("No speech detected within 5 seconds.")
                    except sr.UnknownValueError:
                        print("Could not understand the audio.")
                    except sr.RequestError as e:
                        print(f"Error with the recognition service: {e}")
            except Exception as e:
                print(f"Error with microphone: {e}")
                time.sleep(5)
    
    def stop(self):
        self.running = False

class TemperatureAndHumidityController:
    def __init__(self, device_controller):
        self.device_controller = device_controller
        self.running = True
        self.temp_threshold_on = None
        self.temp_threshold_off = None
        self.consecutive_high_humidity = 0  # Counter for high humidity
        self.consecutive_low_humidity = 0   # Counter for low humidity
        self.dehumidifier_status = "off"    # Track dehumidifier status
        self.consecutive_high_temp = 0
        self.consecutive_low_temp = 0
    
    @staticmethod
    def log_fan_data(timestamp, temperature, fan_status):
        file_exists = os.path.isfile(HISTORY_FILE)
        with open(HISTORY_FILE, 'a', newline='') as csvfile:
            fieldnames = ['time', 'temperature', 'fan_status']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'time': timestamp,
                'temperature': temperature,
                'fan_status': fan_status
            })
        print(f"Data logged: Time={timestamp}, Temperature={temperature}, Fan Status={fan_status}")
    
    def auto_control_dehumidifier(self, humidity):
        """Automatically control dehumidifier based on humidity level over 20 minutes"""
        if humidity is None:
            return
        if humidity > HUMIDITY_THRESHOLD:
            self.consecutive_high_humidity += 1
            self.consecutive_low_humidity = 0
        else:
            self.consecutive_low_humidity += 1
            self.consecutive_high_humidity = 0
        if self.consecutive_high_humidity > HUMIDITY_DURATION and self.dehumidifier_status == "off":
            self.device_controller.control_dehumidifier("on")
            self.dehumidifier_status = "on"
            print(f"[AUTO DEHUMIDIFIER] Dehumidifier ON after 20 minutes above {HUMIDITY_THRESHOLD}% (humidity={humidity}%)")
        elif self.consecutive_low_humidity > HUMIDITY_DURATION and self.dehumidifier_status == "on":
            self.device_controller.control_dehumidifier("off")
            self.dehumidifier_status = "off"
            print(f"[AUTO DEHUMIDIFIER] Dehumidifier OFF after 20 minutes below {HUMIDITY_THRESHOLD}% (humidity={humidity}%)")
        else:
            print(f"[AUTO DEHUMIDIFIER] Monitoring: Dehumidifier {self.dehumidifier_status} (humidity={humidity}%, consecutive {'high' if humidity > HUMIDITY_THRESHOLD else 'low'}={max(self.consecutive_high_humidity, self.consecutive_low_humidity)} checks)")
    
    def monitor_temperature_and_humidity(self, interval=15):
        while self.running:
            prediction = FanPatternPredictor.predict()
            if 'error' not in prediction and prediction['temp_threshold_on'] is not None:
                self.temp_threshold_on = prediction['temp_threshold_on']
                self.temp_threshold_off = prediction['temp_threshold_off']
                print(f"AI prediction: Turn on at {self.temp_threshold_on}°C, Turn off at {self.temp_threshold_off}°C")
            
            temperature, humidity = read_temperature_and_humidity()
            send_data_to_thingspeak(temperature=temperature, humidity=humidity)
            if temperature is not None:
                timestamp = time.strftime("%H:%M:%S")
                self.log_fan_data(timestamp, temperature, self.device_controller.fan_status)
                print(f"Current temperature: {temperature}°C, Humidity: {humidity}%")
                
                if self.temp_threshold_on is not None and self.temp_threshold_off is not None:
                    if temperature >= self.temp_threshold_on and self.device_controller.fan_status == 0:
                        self.consecutive_low_temp = 0
                        self.consecutive_high_temp += 1
                        if self.consecutive_high_temp > 1:
                            print(f"Auto fan control: Temperature ({temperature}°C) reached threshold ({self.temp_threshold_on}°C), turning fan ON")
                            self.device_controller.control_fan("forward", DEFAULT_FAN_SPEED)
                            self.consecutive_high_temp = 0
                    elif temperature <= self.temp_threshold_off and self.device_controller.fan_status == 1:
                        self.consecutive_high_temp = 0
                        self.consecutive_low_temp += 1
                        if self.consecutive_low_temp > 1:
                            print(f"Auto fan control: Temperature ({temperature}°C) reached threshold ({self.temp_threshold_off}°C), turning fan OFF")
                            self.device_controller.control_fan("stop")
                            self.consecutive_low_temp = 0
                    else:
                        self.consecutive_low_temp = 0
                        self.consecutive_high_temp = 0

            
            self.auto_control_dehumidifier(humidity)
            time.sleep(interval)
    
    def stop(self):
        self.running = False

class LightController:
    def __init__(self, device_controller):
        self.device_controller = device_controller
        self.running = True
        self.light_seconds = 0
        self.consecutive_high = 0
        self.consecutive_low = 0
        self.light_status = "off"
    
    def auto_control_light(self, lux):
        if lux > LIGHT_THRESHOLD:
            self.consecutive_high += 1
            self.consecutive_low = 0
        else:
            self.consecutive_low += 1
            self.consecutive_high = 0
        if self.consecutive_high > 60 and self.light_status == "on":
            self.device_controller.control_led("off")
            self.light_status = "off"
            print(f"[AUTO LIGHT] Light OFF after 15 minutes above threshold (lux={lux:.2f})")
        elif self.consecutive_low > 20 and self.light_status == "off":
            self.device_controller.control_led("on")
            self.light_status = "on"
            print(f"[AUTO LIGHT] Light ON after 5 minutes below threshold (lux={lux:.2f})")
        else:
            print(f"[AUTO LIGHT] Monitoring: Light {self.light_status} (lux={lux:.2f}, consecutive {'high' if lux > LIGHT_THRESHOLD else 'low'}={max(self.consecutive_high, self.consecutive_low)} checks)")
    
    def monitor_light_continuously(self, interval=15):
        while self.running:
            lux = read_light()
            send_data_to_thingspeak(light_intensity=lux)
            self.auto_control_light(lux)
            time.sleep(interval)
    
    def stop(self):
        self.running = False

class DeviceController:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(FAN_INA_PIN, GPIO.OUT)
        GPIO.setup(FAN_INB_PIN, GPIO.OUT)
        GPIO.setup(LED_PIN, GPIO.OUT)
        GPIO.setup(DEHUMIDIFIER_PIN, GPIO.OUT)
        self.pwm_ina = GPIO.PWM(FAN_INA_PIN, PWM_FREQUENCY)
        self.pwm_inb = GPIO.PWM(FAN_INB_PIN, PWM_FREQUENCY)
        self.pwm_ina.start(0)
        self.pwm_inb.start(0)
        self.fan_status = 0
        self.led_status = "off"
        self.dehumidifier_status = "off"
    
    def control_fan(self, direction, speed=DEFAULT_FAN_SPEED):
        if direction == "forward":
            DeviceLog.logActivity("fan", "on")
            self.pwm_ina.ChangeDutyCycle(speed)
            self.pwm_inb.ChangeDutyCycle(0)
            print(f"[CONTROL] Fan FORWARD at speed {speed}%")
            self.fan_status = 1
        elif direction == "reverse":
            DeviceLog.logActivity("fan", "on")
            self.pwm_ina.ChangeDutyCycle(0)
            self.pwm_inb.ChangeDutyCycle(speed)
            print(f"[CONTROL] Fan REVERSE at speed {speed}%")
            self.fan_status = 1
        elif direction == "stop":
            DeviceLog.logActivity("fan", "off")
            self.pwm_ina.ChangeDutyCycle(0)
            self.pwm_inb.ChangeDutyCycle(0)
            print("[CONTROL] Fan STOP")
            self.fan_status = 0
    
    def control_led(self, state):
        if state == "on":
            DeviceLog.logActivity("light", "on")
            GPIO.output(LED_PIN, GPIO.LOW)
            print("[CONTROL] LED ON")
            self.led_status = "on"
        elif state == "off":
            DeviceLog.logActivity("light", "off")
            GPIO.output(LED_PIN, GPIO.HIGH)
            print("[CONTROL] LED OFF")
            self.led_status = "off"

    def control_dehumidifier(self, state):
        if state == "on":
            DeviceLog.logActivity("dehumidifier", "on")
            GPIO.output(DEHUMIDIFIER_PIN, GPIO.LOW)
            print("[CONTROL] Dehumidifier ON")
            self.dehumidifier_status = "on"
        elif state == "off":
            DeviceLog.logActivity("dehumidifier", "off")
            GPIO.output(DEHUMIDIFIER_PIN, GPIO.HIGH)
            print("[CONTROL] Dehumidifier OFF")
            self.dehumidifier_status = "off"
    
    def get_status(self):
        return {
            "fan": "on" if self.fan_status == 1 else "off",
            "light": self.led_status,
            "dehumidifier": self.dehumidifier_status
        }
    
    def cleanup(self):
        self.control_fan("stop")
        self.control_led("off")
        self.control_dehumidifier("off")
        self.pwm_ina.stop()
        self.pwm_inb.stop()
        GPIO.cleanup()
        print("[SYSTEM] All devices turned off and GPIO cleaned up successfully")

@app.route('/')
def serve_html():
    return send_from_directory(os.path.dirname(__file__), 'dashboard.html')

@app.route('/mystyle.css')
def serve_css():
    return send_from_directory(os.path.dirname(__file__), 'mystyle.css')

@app.route('/api/command/<command>', methods=['GET'])
def execute_command(command):
    global voice_processor
    return jsonify(voice_processor.process_voice_command(command))

@app.route('/api/predict', methods=['GET'])
def predict_fan_pattern():
    prediction = FanPatternPredictor.predict()
    if 'error' in prediction:
        return jsonify({"status": "error", "message": prediction['error']})
    return jsonify(prediction)

@app.route('/api/data', methods=['GET'])
def get_datas():
    global device_data
    data = device_data.get_data()
    if not data:
        return jsonify({"status": "error", "message": "No data available"})
    return jsonify(data)

@app.route('/api/logs', methods=['GET'])
def get_logs():
    logs = DeviceLog.getLogs()
    if not logs:
        return jsonify({"status": "error", "message": "No logs found"})
    return jsonify(logs)

@app.route('/api/status', methods=['GET'])
def get_status():
    global device_controller
    return jsonify(device_controller.get_status())

@app.route('/api/fan/<status>', methods=['GET'])
def control_fan(status):
    global device_controller
    if status == 'on':
        device_controller.control_fan("forward", DEFAULT_FAN_SPEED)
        return jsonify({"status": "success", "message": "Fan turned on"})
    elif status == 'off':
        device_controller.control_fan("stop")
        return jsonify({"status": "success", "message": "Fan turned off"})
    return jsonify({"status": "error", "message": "Invalid request"})

@app.route('/api/light/<status>', methods=['GET'])
def control_light(status):
    global device_controller
    if status == 'on':
        device_controller.control_led("on")
        return jsonify({"status": "success", "message": "Light turned on"})
    elif status == 'off':
        device_controller.control_led("off")
        return jsonify({"status": "success", "message": "Light turned off"})
    return jsonify({"status": "error", "message": "Invalid request"})

@app.route('/api/dehumidifier/<status>', methods=['GET'])
def control_dehumidifier(status):
    global device_controller
    if status == 'on':
        device_controller.control_dehumidifier("on")
        return jsonify({"status": "success", "message": "Dehumidifier turned on"})
    elif status == 'off':
        device_controller.control_dehumidifier("off")
        return jsonify({"status": "success", "message": "Dehumidifier turned off"})
    return jsonify({"status": "error", "message": "Invalid request"})

def reset_all_devices():
    global device_controller
    device_controller = DeviceController()

if __name__ == "__main__":
    device_controller = DeviceController()
    voice_processor = VoiceCommandProcessor(device_controller)
    temperature_and_humidity_controller = TemperatureAndHumidityController(device_controller)
    light_controller = LightController(device_controller)
    
    voice_thread = threading.Thread(target=voice_processor.listen_for_commands)
    temperature_and_humidity_controller_thread = threading.Thread(target=temperature_and_humidity_controller.monitor_temperature_and_humidity, args=(5,))
    light_controller_thread = threading.Thread(target=light_controller.monitor_light_continuously, args=(15,))
    
    voice_thread.start()
    temperature_and_humidity_controller_thread.start()
    light_controller_thread.start()
    
    # api_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False))
    api_thread = threading.Thread(target=lambda: pywsgi.WSGIServer(('0.0.0.0', 5000), app).serve_forever())
    api_thread.daemon = True
    api_thread.start()


    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Program interrupted")
        voice_processor.stop()
        temperature_and_humidity_controller.stop()
        light_controller.stop()
        voice_thread.join()
        temperature_and_humidity_controller_thread.join()
        light_controller_thread.join()
    finally:
        device_controller.cleanup()