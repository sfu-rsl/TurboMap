#!/bin/bash

if [ $# -ne 3 ]; then
    echo "Usage: $0 <mode> <version> <num_iterations>"
    exit 1
fi

mode=$1
version=$2
num_itr=$3

# datasets=( "MH01" "MH02" "MH03" "MH04" "MH05" "V101" "V102" "V103" "V201" "V202" "V203" "room1" "room2" "room3" "room4" "room5" "room6" "corridor1" "corridor2" "corridor3")
datasets=("MH01" "MH02" "MH05" "room1" "room2" "room3" "corridor1")

for i in $(seq 0 $(expr $num_itr - 1)); do
    for dataset in "${datasets[@]}"; do
        echo -e "[bash:] -> ./run_script.sh $dataset $mode 1 $version.$i"
        ./run_script.sh $dataset $mode 1 $version.$i
    done
done