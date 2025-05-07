#!/bin/bash

# Define arrays for all parameters
methods=("FedAvg" "FedProx" "FedDyn" "FedCM" "FedSAM" "FedGamma" "FedSpeed" "FedSMOO")

# Run all combinations
for method in "${methods[@]}"; do
    python3 train.py --non-iid --model vit_base_patch16_224 --dataset BloodMNIST --num_class 8 --method ${method}
done

# Run all combinations
for method in "${methods[@]}"; do
    python3 train.py --non-iid --model vit_base_patch16_224 --dataset DermaMNIST --num_class 7 --method ${method}
done