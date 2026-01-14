#!/bin/bash

if [ $# -ne 3 ]; then
    echo "Usage: $0 <[0] for ORB-SLAM3, [1] for FastTrack, [2] for TurboMap, [3] for FastTrack & TurboMap> <version> <num_iterations>"
    exit 1
fi

mode=$1
version=$2
num_itr=$3

if [ "$mode" -eq 0 ]; then
    kernel_status_FT='00001'
    kernel_status_TM='0000'
fi

if [ "$mode" -eq 1 ]; then
    kernel_status_FT='11110'
    kernel_status_TM='0000'
fi

if [ "$mode" -eq 2 ]; then
    kernel_status_FT='00001'
    kernel_status_TM='1111'
fi

if [ "$mode" -eq 3 ]; then
    kernel_status_FT='11110'
    kernel_status_TM='1111'
fi

datasets=("MH01" "MH02" "MH03" "MH04" "MH05" "V101" "V102" "V103" "room1" "room2" "room3" "room4" "room5" "room6" "corridor1" "corridor2" "corridor3" "magistrale1")

for i in $(seq 0 $(expr $num_itr - 1)); do
    for dataset in "${datasets[@]}"; do
        echo -e "[bash:] -> ./run_script.sh $dataset $mode 1 $version.$i $kernel_status_FT $kernel_status_TM"
        ./run_script.sh $dataset $mode 1 $version.$i
    done
done