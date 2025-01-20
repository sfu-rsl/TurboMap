#!/bin/bash

save_ostream=$1

# datasets=( "MH01" "MH02" "MH03" "MH04" "MH05" "V101" "V102" "V103" "V201" "V202" "V203" "room1" "room2" "room3" "room4" "room5" "room6" "corridor1" "corridor2" "corridor3")
datasets=("MH01" "MH02" "MH05" "V101" "V102" "V203" "room1" "room2" "room5" "room6" "corridor1" "corridor2" "corridor3")

for dataset in "${datasets[@]}"; do
    echo -e "[bash:] -> ./run_script.sh $dataset $1"
    ./run_script.sh $dataset 0 $1
done