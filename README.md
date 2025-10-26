### Installation

The setup.sh takes care of the full setup on the raspberrypi

It clones the full scripts repo, installs the dependencies and sets up the process in systemctl

### Scripts

main_pipeline.py: setup the PIR sensor, load the model to memory and setup the camera then wait for a motion trigger from the PIR sensor and once a motion occurs take 5 frames in one second pass these frames to the model and get the label and the confidence score then send the data through the LoraWAN mdoule

lora_send.py: sends the AT command to the Ra-08H lorawan transmitter along with the text to be send

### PIR Connection

* VCC (Red wire): Connect to Pin 4 (5V).

* GND (Black wire): Connect to Pin 6 (GND).

* SIG (Yellow wire): Connect to Pin 11 (GPIO 17).

* NC (White wire): Ignore

### TO DO

#### Test and verify the setup.sh

#### Add a dialogue for the user to add the current username to the dialout group
