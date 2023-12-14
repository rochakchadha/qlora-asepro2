# Script to terminate the pod

#!/bin/bash

# Check if the time duration parameter is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <time_in_seconds>"
    exit 1
fi

# Assign the first argument to a variable
time_duration=$1


# Sleep for the user-specified duration
sleep $time_duration
# Run azcopy to push data

runpodctl stop pod $RUNPOD_POD_ID
echo "Terminating the container..."