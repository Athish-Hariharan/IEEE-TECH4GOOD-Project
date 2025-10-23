### Installation

The setup.sh takes care of the full setup on the raspberrypi

It clones the full scripts repo, installs the dependencies and sets up the process in systemctl

### Scripts

monitor_motion.py: registers an interrupt and awaits a trigger from the PIR sensor, once triggered calls main_pipeline.py

main_pipeline.py: run the camera record a live feed and forward it to the yolo model after processing. Once the model returns the text and confirmation. If the probability is greater than 50% call the lora_send.py to send the text to the lorawan gateway

lora_send.py: sends the AT command to the Ra-08H lorawan transmitter along with the text to be send

### PIR Connection

* VCC (Red wire): Connect to Pin 4 (5V).

* GND (Black wire): Connect to Pin 6 (GND).

* SIG (Yellow wire): Connect to Pin 11 (GPIO 17).

* NC (White wire): Ignore

### TO DO

#### * Write the lora_send.py

#### * Write the logic for raspberry pi camera to take burst images

#### * Write the logic to call the model in main_pipeline.py
