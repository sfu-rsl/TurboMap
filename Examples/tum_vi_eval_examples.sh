#!/bin/bash
pathDatasetTUM_VI=$HOME/ORB_SLAM3_datasets/tumvi/corridor #Example, it is necesary to change it by the dataset path

# Single Session Example

echo "Launching Corridor 2 with Stereo-Inertial sensor"
./Stereo-Inertial/stereo_inertial_tum_vi ../Vocabulary/ORBvoc.txt Stereo-Inertial/TUM-VI.yaml "$pathDatasetTUM_VI"/dataset-corridor2_512_16/mav0/cam0/data "$pathDatasetTUM_VI"/dataset-corridor2_512_16/mav0/cam1/data Stereo-Inertial/TUM_TimeStamps/dataset-corridor2_512.txt Stereo-Inertial/TUM_IMU/dataset-corridor2_512.txt dataset-corridor2_512_stereoi
echo "------------------------------------"
echo "Evaluation of Corridor 2 trajectory with Stereo-Inertial sensor"
python3 ../evaluation/evaluate3.py "$pathDatasetTUM_VI"/dataset-corridor2_512_16/mav0/mocap0/data.csv f_dataset-corridor2_512_stereoi.txt --plot corridor2_512_stereoi.pdf
