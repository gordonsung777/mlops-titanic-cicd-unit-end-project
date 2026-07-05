import subprocess
import sys 

def run_step(command):
    print("Running:", command)
    subprocess.run(command, check=True)

run_step([sys.executable, "src/train.py"])
run_step([sys.executable, "src/deploy.py"])
print("Pipeline complete.")