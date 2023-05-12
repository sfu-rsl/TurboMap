#!/bin/bash
pathDatasetTUM_VI=$HOME/ORB_SLAM3_Datasets/tumvi/magistrale #Example, it is necesary to change it by the dataset path

# Single Session Example

echo "Launching Corridor 2 with Stereo-Inertial sensor"
./Stereo-Inertial/stereo_inertial_tum_vi ../Vocabulary/ORBvoc.txt Stereo-Inertial/TUM-VI.yaml "$pathDatasetTUM_VI"/dataset-magistrale1_512_16/mav0/cam0/data "$pathDatasetTUM_VI"/dataset-magistrale1_512_16/mav0/cam1/data Stereo-Inertial/TUM_TimeStamps/dataset-magistrale1_512.txt Stereo-Inertial/TUM_IMU/dataset-magistrale1_512.txt dataset-magistrale1_512_stereoi
# ./Stereo-Inertial/stereo_inertial_tum_vi ../Vocabulary/ORBvoc.txt Stereo-Inertial/TUM-VI.yaml $HOME/ORB_SLAM3_Datasets/tumvi/room/dataset-magistrale1_512_16/mav0/cam0/data $HOME/ORB_SLAM3_Datasets/tumvi/room/dataset-magistrale1_512_16/mav0/cam1/data Stereo-Inertial/TUM_TimeStamps/dataset-magistrale1_512.txt Stereo-Inertial/TUM_IMU/dataset-magistrale1_512.txt "$pathDatasetTUM_VI"/dataset-magistrale1_512_16/mav0/cam0/data "$pathDatasetTUM_VI"/dataset-magistrale1_512_16/mav0/cam1/data Stereo-Inertial/TUM_TimeStamps/dataset-magistrale1_512.txt Stereo-Inertial/TUM_IMU/dataset-magistrale1_512.txt dataset-magistrale1_magistrale1_512_stereoi
echo "------------------------------------"
echo "Evaluation of Corridor 2 trajectory with Stereo-Inertial sensor"
python3 ../evaluation/evaluate3.py "$pathDatasetTUM_VI"/dataset-magistrale1_512_16/mav0/mocap0/data.csv f_dataset-magistrale1_512_stereoi.txt --plot magistrale1_512_stereoi.pdf
