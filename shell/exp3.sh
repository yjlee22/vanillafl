#!/bin/bash

python3 train.py --non-iid --model vit_base_patch16_224 --dataset BloodMNIST --num_class 8 --method FedProx --lamb 0.01
python3 train.py --non-iid --model vit_base_patch16_224 --dataset BloodMNIST --num_class 8 --method FedProx --lamb 0.001
python3 train.py --non-iid --model vit_base_patch16_224 --dataset BloodMNIST --num_class 8 --method FedDyn --beta 0.01
python3 train.py --non-iid --model vit_base_patch16_224 --dataset BloodMNIST --num_class 8 --method FedDyn --beta 0.001
python3 train.py --non-iid --model vit_base_patch16_224 --dataset BloodMNIST --num_class 8 --method FedCM --alpha 0.01
python3 train.py --non-iid --model vit_base_patch16_224 --dataset BloodMNIST --num_class 8 --method FedCM --alpha 0.001
python3 train.py --non-iid --model vit_base_patch16_224 --dataset BloodMNIST --num_class 8 --method FedSAM --rho 0.1
python3 train.py --non-iid --model vit_base_patch16_224 --dataset BloodMNIST --num_class 8 --method FedSAM --rho 0.01
python3 train.py --non-iid --model vit_base_patch16_224 --dataset BloodMNIST --num_class 8 --method FedGamma --rho 0.1
python3 train.py --non-iid --model vit_base_patch16_224 --dataset BloodMNIST --num_class 8 --method FedGamma --rho 0.01
python3 train.py --non-iid --model vit_base_patch16_224 --dataset BloodMNIST --num_class 8 --method FedSpeed --rho 0.1
python3 train.py --non-iid --model vit_base_patch16_224 --dataset BloodMNIST --num_class 8 --method FedSpeed --rho 0.01
python3 train.py --non-iid --model vit_base_patch16_224 --dataset BloodMNIST --num_class 8 --method FedSMOO --rho 0.1
python3 train.py --non-iid --model vit_base_patch16_224 --dataset BloodMNIST --num_class 8 --method FedSMOO --rho 0.01

python3 train.py --non-iid --model vit_base_patch16_224 --dataset DermaMNIST --num_class 7 --method FedProx --lamb 0.01
python3 train.py --non-iid --model vit_base_patch16_224 --dataset DermaMNIST --num_class 7 --method FedProx --lamb 0.001
python3 train.py --non-iid --model vit_base_patch16_224 --dataset DermaMNIST --num_class 7 --method FedDyn --beta 0.01
python3 train.py --non-iid --model vit_base_patch16_224 --dataset DermaMNIST --num_class 7 --method FedDyn --beta 0.001
python3 train.py --non-iid --model vit_base_patch16_224 --dataset DermaMNIST --num_class 7 --method FedCM --alpha 0.01
python3 train.py --non-iid --model vit_base_patch16_224 --dataset DermaMNIST --num_class 7 --method FedCM --alpha 0.001
python3 train.py --non-iid --model vit_base_patch16_224 --dataset DermaMNIST --num_class 7 --method FedSAM --rho 0.1
python3 train.py --non-iid --model vit_base_patch16_224 --dataset DermaMNIST --num_class 7 --method FedSAM --rho 0.01
python3 train.py --non-iid --model vit_base_patch16_224 --dataset DermaMNIST --num_class 7 --method FedGamma --rho 0.1
python3 train.py --non-iid --model vit_base_patch16_224 --dataset DermaMNIST --num_class 7 --method FedGamma --rho 0.01
python3 train.py --non-iid --model vit_base_patch16_224 --dataset DermaMNIST --num_class 7 --method FedSpeed --rho 0.1
python3 train.py --non-iid --model vit_base_patch16_224 --dataset DermaMNIST --num_class 7 --method FedSpeed --rho 0.01
python3 train.py --non-iid --model vit_base_patch16_224 --dataset DermaMNIST --num_class 7 --method FedSMOO --rho 0.1
python3 train.py --non-iid --model vit_base_patch16_224 --dataset DermaMNIST --num_class 7 --method FedSMOO --rho 0.01