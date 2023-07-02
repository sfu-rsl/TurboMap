#!/bin/bash
pathDatasetTUM_VI=$HOME/ORB_SLAM3_Datasets/tumvi #Example, it is necesary to change it by the dataset path

# Single Session Example

# echo "Launching magistrale with Stereo-Inertial sensor"
./Stereo-Inertial/stereo_inertial_tum_vi ../Vocabulary/ORBvoc.txt Stereo-Inertial/TUM-VI.yaml "$pathDatasetTUM_VI"/magistrale/dataset-magistrale1_512_16/mav0/cam0/data "$pathDatasetTUM_VI"/magistrale/dataset-magistrale1_512_16/mav0/cam1/data Stereo-Inertial/TUM_TimeStamps/dataset-magistrale1_512.txt Stereo-Inertial/TUM_IMU/dataset-magistrale1_512.txt  dataset-magistrale1_512_stereoi

# echo "------------------------------------"
# echo "Evaluation of Corridor 2 trajectory with Stereo-Inertial sensor"
# python3 ../evaluation/evaluate3.py "$pathDatasetTUM_VI"/dataset-magistrale1_512_16/mav0/mocap0/data.csv f_dataset-magistrale1_512_stereoi.txt --plot magistrale1_512_stereoi.pdf


#Multi Session Example
# ./Stereo-Inertial/stereo_inertial_tum_vi ../Vocabulary/ORBvoc.txt Stereo-Inertial/TUM-VI.yaml "$pathDatasetTUM_VI"/room/dataset-room1_512_16/mav0/cam0/data "$pathDatasetTUM_VI"/room/dataset-room1_512_16/mav0/cam1/data Stereo-Inertial/TUM_TimeStamps/dataset-room1_512.txt Stereo-Inertial/TUM_IMU/dataset-room1_512.txt "$pathDatasetTUM_VI"/magistrale/dataset-magistrale1_512_16/mav0/cam0/data "$pathDatasetTUM_VI"/magistrale/dataset-magistrale1_512_16/mav0/cam1/data Stereo-Inertial/TUM_TimeStamps/dataset-magistrale1_512.txt Stereo-Inertial/TUM_IMU/dataset-magistrale1_512.txt dataset-room1_magistrale1_512_stereoi
