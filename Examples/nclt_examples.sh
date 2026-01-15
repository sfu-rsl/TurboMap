#!/bin/bash
pathDataset=$HOME/ORB_SLAM3_Datasets/nclt/Robot-Localization-on-NCLT-dataset-using-ORB-SLAM-III-and-Graph-Based-Sensor-Fusion/Utility
 #Example, it is necesary to change it by the dataset path

# Single Session Example

echo "Launching NCLT with Monocular-Inertial sensor"
./Monocular-Inertial/mono_inertial_euroc ../Vocabulary/ORBvoc.txt ./Monocular-Inertial/NCLT.yaml "$pathDataset"/NCLT ./Monocular-Inertial/NCLT_TimeStamps/2013-04-05.txt dataset-NCLT_monoi

echo "------------------------------------"
echo "Evaluation of NCLT trajectory with Monocular-Inertial sensor"
python3 ../evaluation/evaluate3.py "$pathDataset"/groundtruth_2013-04-05.csv f_dataset-NCLT_monoi.txt --plot dataset-NCLT_monoi.pdf --verbose
