#!/bin/bash

python=/usr/bin/python3
device=cuda:1

CIFAR10_ResNEst_list=(
CIFAR10_ResNEst_ResNet_110
CIFAR10_ResNEst_ResNet_20
CIFAR10_ResNEst_WRN_16_8
CIFAR10_ResNEst_WRN_40_4
)

CIFAR10_AResNEst_list=(
CIFAR10_AResNEst_ResNet_110
CIFAR10_AResNEst_ResNet_20
CIFAR10_AResNEst_WRN_16_8
CIFAR10_AResNEst_WRN_40_4
)

CIFAR100_ResNEst_list=(
CIFAR100_ResNEst_ResNet_110
CIFAR100_ResNEst_ResNet_20
CIFAR100_ResNEst_WRN_16_8
CIFAR100_ResNEst_WRN_40_4
)

CIFAR100_AResNEst_list=(
CIFAR100_AResNEst_ResNet_110
CIFAR100_AResNEst_ResNet_20
CIFAR100_AResNEst_WRN_16_8
CIFAR100_AResNEst_WRN_40_4
)

for model in ${CIFAR10_ResNEst_list[@]}
do
    $python main.py --data CIFAR10 --model $model --device $device --mu 0.01
done

for model in ${CIFAR10_AResNEst_list[@]}
do
    $python main.py --data CIFAR10 --model $model --device $device
done

for model in ${CIFAR100_ResNEst_list[@]}
do
    $python main.py --data CIFAR100 --model $model --device $device --mu 0.01
done

for model in ${CIFAR100_AResNEst_list[@]}
do
    $python main.py --data CIFAR100 --model $model --device $device
done
