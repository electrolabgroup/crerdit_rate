import subprocess
import time

script_path = r"C:\Users\Sayadf\Desktop\Python_EE\30-7-2024\Credit_Rate_Calulator_UPDATED_21stMay.py"

while True:
    process = subprocess.Popen(["python", script_path])

    time.sleep(30)

    process.terminate()
    # process.kill
    process.wait()
