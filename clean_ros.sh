#!/bin/bash

# Navigate to the target directory
cd Examples_old/ROS/ORB_SLAM3

# List of files to remove
files_to_remove=("Mono" "Mono_Inertial" "Stereo" "Stereo_Inertial" "MonoAR" "RGBD")

# Remove the build directory
if [ -d "build" ]; then
    rm -rf "build"
    echo "Removed build directory"
else
    echo "Directory build does not exist"
fi

# Loop through the list and remove each file
for file in "${files_to_remove[@]}"; do
    if [ -f "$file" ]; then
        rm "$file"
        echo "Removed $file"
    else
        echo "File $file does not exist"
    fi
done

echo "Clean-up completed."