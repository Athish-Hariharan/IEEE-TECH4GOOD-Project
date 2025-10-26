#!/usr/bin/env bash

# We need root check for it first
check_sudo(){
    if (( EUID != 0 )); then
        echo "ERROR: This script must be run with sudo or as the root user. Exiting." >&2
        exit 1
    fi
}
check_sudo

USER_NAME="tech4gd" # Hardcoded user
PROJECT_DIR="/home/$USER_NAME/Project"
VENV_DIRECTORY="$PROJECT_DIR/.venv"
SERVICE_FILE="/etc/systemd/system/main_pipeline.service"

# to clone the scripts repo
clone_setup(){
    # Run all git commands as the $USER_NAME
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project already cloned '$PROJECT_DIR' Updating....."
        # We cd into the dir, then run git pull as the user
        (cd "$PROJECT_DIR" && sudo -u "$USER_NAME" git pull)
        sudo -u "$USER_NAME" git checkout setup
    else
        echo "Project Directory '$PROJECT_DIR' does not exist, Cloning......"
        # Clone as the user
        sudo -u "$USER_NAME" git clone https://github.com/Athish-Hariharan/IEEE-TECH4GOOD-Project.git "$PROJECT_DIR"
        # cd into the new dir and checkout as the user
        (cd "$PROJECT_DIR" && sudo -u "$USER_NAME" git checkout setup)
    fi
    
    # Check if clone failed (e.g., dir exists but isn't a git repo)
    if [ ! -d "$PROJECT_DIR/.git" ]; then
        echo "ERROR: Failed to clone git repo into $PROJECT_DIR"
        exit 1
    fi
}

# Check if the virtual enve exists
venv_check(){
    local req_file="$PROJECT_DIR/requirements.txt"
    local pip_exec="$VENV_DIRECTORY/bin/pip"
    local python_exec="$VENV_DIRECTORY/bin/python3" # Use the venv python

    if [ ! -f "$req_file" ]; then
        echo "ERROR: requirements.txt not found at $req_file"
        exit 1
    fi

    if [ ! -d "$VENV_DIRECTORY" ]; then
        echo "Virtual env '$VENV_DIRECTORY' does not exist, Creating now....."
        # Create venv as the user
        sudo -u "$USER_NAME" python3 -m venv "$VENV_DIRECTORY"
    else
        echo "Virtual env '$VENV_DIRECTORY' exists. Updating packages..."
    fi

    # Install/update packages as the user
    echo "Installing requirements from $req_file..."
    # Use the venv's pip to install packages
    sudo -u "$USER_NAME" "$pip_exec" install -r "$req_file"
}

# Run as root to setup the .service file
create_service() {
    echo "Creating systemd service file at $SERVICE_FILE..."
    
    # This part runs as root
    cat > "$SERVICE_FILE" << EOL
[Unit]
Description=PIR Motion Monitor Service
After=multi-user.target

[Service]
# Set the working directory so relative paths in Python work
WorkingDirectory=$PROJECT_DIR

# This is the command to run your script (using the venv)
ExecStart=$VENV_DIRECTORY/bin/python3 $PROJECT_DIR/monitor_motion.py

# Run as the specified user
User=$USER_NAME
# Automatically restart the service if it fails
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOL

    echo "Reloading systemd, enabling and starting service..."
    # No sudo needed here, we are already root
    systemctl daemon-reload
    systemctl enable motion_monitor.service
    systemctl start motion_monitor.service
    
    echo "Done. Service created and started."
    echo "Check status with: systemctl status motion_monitor.service"
}


echo "Starting setup for user $USER_NAME..."
clone_setup
venv_check
create_service
echo "Setup complete."
