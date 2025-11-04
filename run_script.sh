#!/bin/bash

if [ $# -lt 4 ]; then
    echo "Usage: $0 <dataset_name> <[0] for ORB-SLAM3, [1] for FastTrack, [2] for TurboMap> <[0] for STDOUT output, [1] for file output> <version> <TurboMap_mode>"
    exit 1
fi

dataset_name=$1
mode=$2
save_ostream=$3
version=$4

if [ $# -eq 5 ]; then
    TurboMap_mode=$5
else
    TurboMap_mode='1111'
fi

if [ "$mode" -eq 2 ]; then
    system_name='TurboMap'
fi
if [ "$mode" -eq 1 ]; then
    system_name='FastTrack'
fi
if [ "$mode" -eq 0 ]; then
    system_name='ORB-SLAM3'
fi
statsDir="./Results/${system_name}/${dataset_name}/${version}"

if [ "$system_name" == 'TurboMap' ]; then
    statsDir="./Results/${system_name}/${TurboMap_mode}/${dataset_name}/${version}"
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
        ./euroc_eval_examples.sh "$mode" "$TurboMap_mode" "$dataset_name" "../$statsDir" 
    elif $found_in_tumvi; then
        cd Examples/
        ./tum_vi_eval_examples.sh "$mode" "$TurboMap_mode" "$dataset_name" "../$statsDir" 
    else
        echo "Invalid dataset: $dataset_name"
        exit 1
    fi
else
    if $found_in_euroc; then
        cd Examples/
        ./euroc_eval_examples.sh "$mode" "$TurboMap_mode" "$dataset_name" "../$statsDir" > "../${statsDir}/ostream.txt" 
    elif $found_in_tumvi; then
        cd Examples/
        ./tum_vi_eval_examples.sh "$mode" "$TurboMap_mode" "$dataset_name" "../$statsDir" > "../${statsDir}/ostream.txt" 
    else
        echo "Invalid dataset: $dataset_name"
        exit 1
    fi
fi