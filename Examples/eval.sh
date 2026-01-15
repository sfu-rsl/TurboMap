#!/bin/bash
pathDatasetTUM_VI=$HOME/tumvi
seq=dataset-corridor2_512
echo $pathDatasetTUM_VI
echo $seq
python3 ../evaluation/evaluate3.py "$pathDatasetTUM_VI"/"$seq"_16/mav0/mocap0/data.csv f_"$seq"_stereoi.txt