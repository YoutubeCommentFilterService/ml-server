import threading
import redis
import time
import subprocess
import os
import re
import json
from schemes.config import REDIS_PUBSUB_TEGRA_KEY, REDIS_PUBSUB_TEGRA_MAX_VALUE, REDIS_REQUEST_TIME_KEY

redis_client = redis.Redis()
redis_pubsub = redis_client.pubsub()
IDLE_THRESHOLD_SEC = 5

power_mode = {"max": None, "min": None}
is_on_tegra = os.path.exists('/etc/nvpmodel.conf')

tegra_status_high = False

def monitor_gpu_idle():
    while True:
        time.sleep(IDLE_THRESHOLD_SEC)
        try:
            last_request = float(redis_client.get(REDIS_REQUEST_TIME_KEY) or 0)
            if time.time() - last_request > IDLE_THRESHOLD_SEC:
                set_jetson_idle()
        except Exception as e:
            print(f'Error in monitor: {e}')

def set_jetson_high():
    global tegra_status_high
    if not is_on_tegra:
        return
    try:
        subprocess.run(['sudo', 'nvpmodel', '-m', power_mode['max']])
        tegra_status_high = True
    except Exception as e:
        print(f"Error setting GPU to max : {e}")

def set_jetson_idle():
    global tegra_status_high
    if not is_on_tegra:
        return
    try:
        subprocess.run(['sudo', 'nvpmodel', '-m', power_mode['min']])
        tegra_status_high = False
    except Exception as e:
        print(f"Error setting GPU to idle: {e}")

def init_power_mode():
    global power_mode
    try:
        result = subprocess.run('jetson_release | grep Module', shell=True, capture_output=True, text=True)
        output = result.stdout.strip()

        clean_output = re.sub(r'\x1b\[[0-9;]*m', '', output)
        
        module_info = clean_output.split(": ")[-1]

        with open("./tegra_powers.json", "r") as f:
            power_modes = json.load(f)
        
        power_mode["max"] = str(power_modes[module_info]["max"])
        power_mode["min"] = str(power_modes[module_info]["min"])
    except Exception as e:
        print(f"Error loading power modes: {e}")

def subscribe_gpu_status():
    redis_pubsub.subscribe(REDIS_PUBSUB_TEGRA_KEY)

    for message in redis_pubsub.listen():
        if message['type'] == 'message':
            command = message['data'].decode('utf-8')
            if command == REDIS_PUBSUB_TEGRA_MAX_VALUE and not tegra_status_high:
                set_jetson_high()

if __name__ == "__main__":
    if is_on_tegra:
        init_power_mode()
        set_jetson_idle()
        
        threading.Thread(target=monitor_gpu_idle, daemon=True).start()
        subscribe_gpu_status()