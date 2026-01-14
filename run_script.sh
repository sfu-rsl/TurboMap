#!/bin/bash

if [ $# -lt 4 ]; then
    echo "Usage: $0 <dataset_name> <[0] for ORB-SLAM3, [1] for FastTrack, [2] for TurboMap, [3] for FastTrack & TurboMap> <[0] for STDOUT output, [1] for file output> <version> <kernel_status1 [kernel_status2]>"
    exit 1
fi

dataset_name=$1
mode=$2
save_ostream=$3
version=$4

kernel_status_FT='00001'
kernel_status_TM='0000'

if [ "$mode" -eq 0 ]; then
    system_name='ORB-SLAM3'
fi

if [ "$mode" -eq 1 ]; then
    system_name='FastTrack'
    if [ $# -eq 5 ]; then
        kernel_status_FT=$5
    else
        kernel_status_FT='11110'
    fi
fi

if [ "$mode" -eq 2 ]; then
    system_name='TurboMap'
    if [ $# -eq 5 ]; then
        kernel_status_TM=$5
    else
        kernel_status_TM='1111'
    fi
fi

if [ "$mode" -eq 3 ]; then
    system_name='FastTrack&TurboMap'
    if [ $# -eq 6 ]; then
        kernel_status_FT=$5
        kernel_status_TM=$6
    else
        kernel_status_FT='11110'
        kernel_status_TM='1111'
    fi
fi

if [ "$system_name" == 'ORB-SLAM3' ]; then
    statsDir="./Results/${system_name}/${dataset_name}/${version}"
fi
if [ "$system_name" == 'FastTrack' ]; then
    statsDir="./Results/${system_name}/${kernel_status_FT}/${dataset_name}/${version}"
fi
if [ "$system_name" == 'TurboMap' ]; then
    statsDir="./Results/${system_name}/${kernel_status_TM}/${dataset_name}/${version}"
fi
if [ "$system_name" == 'FastTrack&TurboMap' ]; then
    statsDir="./Results/${system_name}/${kernel_status_FT}-${kernel_status_TM}/${dataset_name}/${version}"
fi

if [ ! -d "$statsDir" ]; then
    mkdir -p "$statsDir"
fi

tumvi_datasets=("corridor1" "corridor2" "corridor3" "corridor4" "corridor5" "outdoors1" "outdoors5" "room1" "room2" "room3" "room4" "room5" "room6" "magistrale1")
euroc_datasets=("MH01" "MH02" "MH03" "MH04" "MH05" "V101" "V102" "V103" "V201" "V202" "V203")

found_in_tumvi=false
for dataset in "${tumvi_datasets[@]}"; do
    if [[ "$dataset" == "$dataset_name" ]]; then
        found_in_tumvi=true
        break
    fi
done

found_in_euroc=false
for dataset in "${euroc_datasets[@]}"; do
    if [[ "$dataset" == "$dataset_name" ]]; then
        found_in_euroc=true
        break
    fi
done

if [ "$save_ostream" -eq 0 ]; then
    if $found_in_euroc; then
        cd Examples/
        ./euroc_eval_examples.sh "$mode" "$kernel_status_FT" "$kernel_status_TM" "$dataset_name" "../$statsDir" 
    elif $found_in_tumvi; then
        cd Examples/
        ./tum_vi_eval_examples.sh "$mode" "$kernel_status_FT" "$kernel_status_TM" "$dataset_name" "../$statsDir" 
    else
        echo "Invalid dataset: $dataset_name"
        exit 1
    fi
else
    if $found_in_euroc; then
        cd Examples/
        ./euroc_eval_examples.sh "$mode" "$kernel_status_FT" "$kernel_status_TM" "$dataset_name" "../$statsDir" > "../${statsDir}/ostream.txt" 
    elif $found_in_tumvi; then
        cd Examples/
        ./tum_vi_eval_examples.sh "$mode" "$kernel_status_FT" "$kernel_status_TM" "$dataset_name" "../$statsDir" > "../${statsDir}/ostream.txt" 
    else
        echo "Invalid dataset: $dataset_name"
        exit 1
    fi
fi