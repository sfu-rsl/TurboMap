#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi

orbExtractionRunstatus=1
stereoMatchRunstatus=1
searchLocalPointsRunstatus=1
poseEstimationRunstatus=1
poseOptimizationRunstatus=0
dataset_name=$1
version='v1'

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

