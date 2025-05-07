#!/bin/bash

# Define arrays for all parameters
methods=("FedAvg" "FedProx" "FedDyn" "FedCM" "FedSAM" "FedGamma" "FedSpeed" "FedSMOO")
models=("swin_base_patch4_window7_224" "deit_base_patch16_224" "mixer_b16_224" "davit_base")

# Run all combinations
for model in "${models[@]}"; do
    for method in "${methods[@]}"; do
        python3 train.py --non-iid --model ${model} --dataset BloodMNIST --num_class 8 --method ${method}
    done
done

# Run all combinations
for model in "${models[@]}"; do
    for method in "${methods[@]}"; do
        python3 train.py --non-iid --model ${model} --dataset DermaMNIST --num_class 7 --method ${method}
    done
done