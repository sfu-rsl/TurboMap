#!/bin/bash
pathDatasetTUM_VI=$HOME/SLAM/Datasets/tumvi #Example, it is necesary to change it by the dataset path

FastTrack_on=$1
TurboMap_on=$2
JacobiGPU_on=$3
kernel_status_FT=$4
kernel_status_TM=$5
dataset_name=$6
statsDir=$7

file_name="dataset-${dataset_name}_stereoi"

# EXECUTABLE=./Stereo-Inertial/stereo_inertial_tum_vi
# ARGS="../Vocabulary/ORBvoc.txt Stereo-Inertial/TUM-VI.yaml ${pathDatasetTUM_VI}/dataset-${dataset_name}_512_16/mav0/cam0/data ${pathDatasetTUM_VI}/dataset-${dataset_name}_512_16/mav0/cam1/data Stereo-Inertial/TUM_TimeStamps/dataset-${dataset_name}_512.txt Stereo-Inertial/TUM_IMU/dataset-${dataset_name}_512.txt  ${file_name} ${statsDir} ${FastTrack_on} ${TurboMap_on} ${JacobiGPU_on} ${kernel_status_FT} ${kernel_status_TM}"
# gdb -ex "set args $ARGS" -ex "run" ./Stereo-Inertial/stereo_inertial_tum_vi
# compute-sanitizer --tool memcheck --report-api-errors all --show-backtrace no ./Stereo-Inertial/stereo_inertial_tum_vi ../Vocabulary/ORBvoc.txt Stereo-Inertial/TUM-VI.yaml ${pathDatasetTUM_VI}/dataset-${dataset_name}_512_16/mav0/cam0/data ${pathDatasetTUM_VI}/dataset-${dataset_name}_512_16/mav0/cam1/data Stereo-Inertial/TUM_TimeStamps/dataset-${dataset_name}_512.txt Stereo-Inertial/TUM_IMU/dataset-${dataset_name}_512.txt ${statsDir} ${FastTrack_on} ${TurboMap_on} ${JacobiGPU_on} ${kernel_status_FT} ${kernel_status_TM}
./Stereo-Inertial/stereo_inertial_tum_vi ../Vocabulary/ORBvoc.txt Stereo-Inertial/TUM-VI.yaml ${pathDatasetTUM_VI}/dataset-${dataset_name}_512_16/mav0/cam0/data ${pathDatasetTUM_VI}/dataset-${dataset_name}_512_16/mav0/cam1/data Stereo-Inertial/TUM_TimeStamps/dataset-${dataset_name}_512.txt Stereo-Inertial/TUM_IMU/dataset-${dataset_name}_512.txt  ${file_name} ${statsDir} ${FastTrack_on} ${TurboMap_on} ${JacobiGPU_on} ${kernel_status_FT} ${kernel_status_TM}

echo "------------------------------------"

echo "Evaluation of ${dataset_name} trajectory with Stereo-Inertial sensor"
python3 -W ignore ../evaluation/evaluate3.py "$pathDatasetTUM_VI"/dataset-${dataset_name}_512_16//mav0/mocap0/data.csv f_${file_name}.txt --plot ${dataset_name}_512_stereoi.pdf --verbose
echo "Plotting data"
python3 ../plot.py "${statsDir}"

files=("f_dataset-${dataset_name}_stereoi.csv"
"f_dataset-${dataset_name}_stereoi.txt"
"f_dataset-${dataset_name}_stereoi.png"
"kf_dataset-${dataset_name}_stereoi.txt"
)
destination_directory="${statsDir}/trajectory"
mkdir -p $destination_directory
mv "${files[@]}" "$destination_directory"