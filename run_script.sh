#!/bin/bash

if [ $# -lt 2 ]; then
    echo "Usage: $0 <dataset_name> <[0] for ORB-SLAM3, [1] for FastTrack> <save_stream>"
    exit 1
fi

dataset_name=$1

if [ "$2" -eq 1 ]; then
    orbExtractionRunstatus=1
    stereoMatchRunstatus=1
    searchLocalPointsRunstatus=1
    poseEstimationRunstatus=1
    poseOptimizationRunstatus=0
    version='FastTrack'
fi

if [ "$2" -eq 0 ]; then
    orbExtractionRunstatus=0
    stereoMatchRunstatus=0
    searchLocalPointsRunstatus=0
    poseEstimationRunstatus=0
    poseOptimizationRunstatus=1
    version='ORB-SLAM3'
fi

if [ $# -ge 3 ]; then
    save_ostream=$3
else
    save_ostream=0
fi

tumvi_datasets=("corridor1" "corridor2" "corridor3" "corridor4" "corridor5" "outdoors1" "outdoors5" "room1" "room2" "room3" "room4" "room5" "room6" "magistrale2" "magistrale6")
euroc_datasets=("MH01" "MH03" "MH02" "MH04" "MH05" "V101" "V102" "V103" "V201" "V202" "V203")

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
        ./euroc_eval_examples.sh "$orbExtractionRunstatus" "$stereoMatchRunstatus" "$searchLocalPointsRunstatus" "$poseEstimationRunstatus" "$poseOptimizationRunstatus" "$dataset_name" "$version" 
    elif $found_in_tumvi; then
        cd Examples/
        ./tum_vi_eval_examples.sh "$orbExtractionRunstatus" "$stereoMatchRunstatus" "$searchLocalPointsRunstatus" "$poseEstimationRunstatus" "$poseOptimizationRunstatus" "$dataset_name" "$version" 
    else
        echo "Invalid dataset: $dataset_name"
        exit 1
    fi
else
    statsDir="Results/${version}/${dataset_name}"
    if [ ! -d "$statsDir" ]; then
        echo $statsDir
        mkdir -p "$statsDir"
    fi
    if $found_in_euroc; then
        cd Examples/
        ./euroc_eval_examples.sh "$orbExtractionRunstatus" "$stereoMatchRunstatus" "$searchLocalPointsRunstatus" "$poseEstimationRunstatus" "$poseOptimizationRunstatus" "$dataset_name" "$version" > "../${statsDir}/ostream.txt" 
    elif $found_in_tumvi; then
        cd Examples/
        ./tum_vi_eval_examples.sh "$orbExtractionRunstatus" "$stereoMatchRunstatus" "$searchLocalPointsRunstatus" "$poseEstimationRunstatus" "$poseOptimizationRunstatus" "$dataset_name" "$version" > "../${statsDir}/ostream.txt" 
    else
        echo "Invalid dataset: $dataset_name"
        exit 1
    fi
fi