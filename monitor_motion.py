import RPi.GPIO as GPIO
import time
import subprocess
import sys
import os

SENSOR_PIN = 17 

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ACTION_SCRIPT_PATH = os.path.join(SCRIPT_DIR, "main_pipeline.py")
except NameError:
    raise SystemExit("Error file main_pipeline.py not found, exiting...") 

print(f"--- Motion Sensor Monitor ---")
print(f"Monitoring PIR sensor on GPIO pin: {SENSOR_PIN}")
print(f"Will call script: {ACTION_SCRIPT_PATH}")
print("Press Ctrl+C to exit.")

# This function is executed in a separate thread when the interrupt is detected
def motion_detected(channel):
    """
    Called when the PIR sensor pin goes HIGH (motion detected).
    """
    print(f"\nMotion detected at {time.ctime()}!")
    
    try:
        print(f"Running action script: {ACTION_SCRIPT_PATH}")
        subprocess.Popen([SCRIPT_DIR+"/.venv/bin/python3", ACTION_SCRIPT_PATH]) # Open without blocking
    except FileNotFoundError:
        print(f"Error: Could not find the script at {ACTION_SCRIPT_PATH}.")
        print("Please check the ACTION_SCRIPT_PATH variable.")
    except Exception as e:
        print(f"Error running action script: {e}")

try:
    # Set up GPIO mode
    GPIO.setmode(GPIO.BCM)
    
    # Set up the sensor pin as an input
    GPIO.setup(SENSOR_PIN, GPIO.IN) # PIR is supposed to input HIGH on trigger

    # Look for a rising edge for interrupt with debounce time 300ms probably too much?
    GPIO.add_event_detect(
        SENSOR_PIN, 
        GPIO.RISING, 
        callback=motion_detected, 
        bouncetime=300
    )

    # Keep the script running to listen for events
    while True:
        # Do nothing, just wait for the interrupt
        time.sleep(1)

except KeyboardInterrupt:
    print("\nStopping monitor.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Clean up GPIO settings on exit
    GPIO.cleanup()
    print("GPIO cleanup complete. Exiting.")

