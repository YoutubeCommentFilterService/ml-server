import redis
import time
import subprocess
import os
import re
import json
from schemes.config import REDIS_REQUEST_TIME_KEY

redis_client = redis.Redis()

IDLE_THRESHOLD_SEC = 5
is_jseton_idle = None

power_mode = {"max": None, "min": None}

def monitor_gpu_idle():
    global is_jetson_idle
    while True:
        try:
            last_request_time = float(redis_client.get(REDIS_REQUEST_TIME_KEY) or 0)
            is_over_threashold = time.time() - last_request_time > IDLE_THRESHOLD_SEC
            if is_over_threashold and not is_jetson_idle:
                set_jetson_nvp_model('min')
                is_jetson_idle = True
            elif not is_over_threashold and is_jetson_idle:
                set_jetson_nvp_model('max')
                is_jetson_idle = False
        except Exception as e:
            print(f'Error in monitor: {e}', flush=True)
        time.sleep(1)

def set_jetson_nvp_model(state: str):
    try:
        subprocess.run(['sudo', 'nvpmodel', '-m', power_mode[state]])
        print(f'switch to nvp model {state} succeed', flush = True)
    except Exception as e:
        print(f"Error setting GPU to max : {e}", flush=True)

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

        if is_jetson_idle:
            set_jetson_nvp_model('min')
        else:
            set_jetson_nvp_model('max')
    except Exception as e:
        print(f"Error loading power modes: {e}")

if __name__ == "__main__":
    if os.path.exists('/etc/nvpmodel.conf'):
        global is_jetson_idle
        is_jetson_idle = time.time() - float(redis_client.get(REDIS_REQUEST_TIME_KEY) or 0)
        init_power_mode()
        monitor_gpu_idle()