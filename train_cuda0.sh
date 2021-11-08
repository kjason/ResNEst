#!/bin/bash

python=/usr/bin/python3
device=cuda:0

CIFAR10_model_list=(
CIFAR10_Standard_ResNet_110
CIFAR10_Standard_ResNet_20
CIFAR10_Standard_ResNet_110
CIFAR10_Standard_ResNet_20
CIFAR10_Standard_WRN_16_8
CIFAR10_Standard_WRN_40_4
CIFAR10_BNResNEst_ResNet_110
CIFAR10_BNResNEst_ResNet_20
CIFAR10_BNResNEst_WRN_16_8
CIFAR10_BNResNEst_WRN_40_4
)

CIFAR100_model_list=(
CIFAR100_Standard_ResNet_110
CIFAR100_Standard_ResNet_20
CIFAR100_Standard_WRN_16_8
CIFAR100_Standard_WRN_40_4
CIFAR100_BNResNEst_ResNet_110
CIFAR100_BNResNEst_ResNet_20
CIFAR100_BNResNEst_WRN_16_8
CIFAR100_BNResNEst_WRN_40_4
)

for model in ${CIFAR10_model_list[@]}
do
    $python main.py --data CIFAR10 --model $model --device $device
done

for model in ${CIFAR100_model_list[@]}
do
    $python main.py --data CIFAR100 --model $model --device $device
done
