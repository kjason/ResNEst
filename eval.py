"""
Created on Sat Mar 6 2021

@author: Kuan-Lin Chen
"""
import torch
import argparse
from data import data_dict
from models import model_dict
from test import testClassifier
from utils import dir_path,cuda_device

parser = argparse.ArgumentParser(description='Evaluate the performance on a given DNN model and report the test accuracy',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', default='CIFAR10', choices=['CIFAR10','CIFAR100'], help='image classification datasets')
parser.add_argument('--checkpoint_folder',default='./checkpoint/', type=dir_path, help='path to the checkpoint folder')
parser.add_argument('--device', default='cuda:0', type=cuda_device, help='specify a CUDA device')
parser.add_argument('--seed_list', default=[0,1,2,3,4,5,6], nargs='+', type=int, help='train a model with different random seeds')
parser.add_argument('--model', default='CIFAR10_Standard_ResNet_110', choices=list(model_dict.keys()), help='the DNN model')
args = parser.parse_args()

torch.backends.cudnn.deterministic = True
print(torch.cuda.get_device_name(int(args.device[-1]))) # print the GPU

testset = data_dict[args.data]().testset

for seed in args.seed_list:
    name = args.model+'_seed='+str(seed)
    print('start testing the model '+name)
    net = model_dict[args.model]()
    pretrained_model = torch.load(args.checkpoint_folder+name+'/last_model.pt')
    net.load_state_dict(pretrained_model,strict=True)
    net = net.to(args.device)
 
    testClassifier(net,testset,args.device,args.checkpoint_folder+name)
