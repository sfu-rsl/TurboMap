#!/bin/bash
pathDataset=$HOME/Robot-Localization-on-NCLT-dataset-using-ORB-SLAM-III-and-Graph-Based-Sensor-Fusion
 #Example, it is necesary to change it by the dataset path

# Single Session Example

echo "Launching MH01 with Stereo-Inertial sensor"
# ./Stereo-Inertial/stereo_inertial_euroc ../Vocabulary/ORBvoc.txt ./Stereo-Inertial/NCLT.yaml "$pathDatasetEuroc"/NCLT ./Stereo-Inertial/nclt.txt dataset-NCLT_easy_stereoi

./Monocular-Inertial/mono_inertial_euroc ../Vocabulary/ORBvoc.txt ./Monocular-Inertial/NCLT.yaml "$pathDataset"/Utility/NCLT "$pathDataset"/Utility/time_stamp.txt dataset-NCLT_easy_stereoi

python3 ../evaluation/evaluate3.py /localhome/dka119/ORB_SLAM3_Datasets/michigan/groundtruth_2013-01-10.csv f_dataset-NCLT_easy_stereoi.txt --plot dataset-NCLT_easy_stereoi.pdf
