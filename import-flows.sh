#!/bin/bash

# ================= CONFIGURATION =================
# 1. The name or ID of your Node-RED container
#    (Run 'docker ps' to find this if you aren't sure)
CONTAINER_NAME="rtos-nodered" 

# 2. The path inside the container where the flow file resides.
#    Standard Node-RED Docker images use /data
SOURCE_PATH="/data/flows.json"

# 3. The path on your HOST machine where you want to save the file
#    (Using $(pwd) saves it to the current directory)
DEST_PATH="$(pwd)/node-red_flows"

# 4. Optional: Add a timestamp to the filename to prevent overwriting
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILENAME="flows.json"
# =================================================

# Step 1: Check if the destination directory exists, create if not
if [ ! -d "$DEST_PATH" ]; then
  echo "Creating destination directory: $DEST_PATH"
  mkdir -p "$DEST_PATH"
fi

# Step 2: Check if the container is running
if [ ! "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Error: Container '$CONTAINER_NAME' is not running."
    exit 1
fi

# Step 3: Copy the file
echo "Copying flows from '$CONTAINER_NAME'..."
docker cp "$CONTAINER_NAME:$SOURCE_PATH" "$DEST_PATH/$OUTPUT_FILENAME"

# Step 4: Verify success
if [ $? -eq 0 ]; then
  echo "Success! Flows saved to:"
  echo "$DEST_PATH/$OUTPUT_FILENAME"
else
  echo "Error: Failed to copy file. Please check if '$SOURCE_PATH' exists inside the container."
fi