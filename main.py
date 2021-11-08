"""
Created on Sat Mar 6 2021

@author: Kuan-Lin Chen
"""
import torch
import argparse
from data import data_dict
from models import model_dict
from train import TrainParam,TrainClassifier
from utils import dir_path,cuda_device

parser = argparse.ArgumentParser(description='Train a DNN model',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', default='CIFAR10', choices=['CIFAR10','CIFAR100'], help='image classification datasets')
parser_resume_group = parser.add_mutually_exclusive_group()
parser_resume_group.add_argument('--resume', dest='resume', action='store_true', help='resume from the last checkpoint',default=True)
parser_resume_group.add_argument('--no-resume', dest='noresume', action='store_true', help='start a new training or overwrite the last one',default=False)
parser.add_argument('--checkpoint_folder',default='./checkpoint/', type=dir_path, help='path to the checkpoint folder')
parser.add_argument('--device', default='cuda:0', type=cuda_device, help='specify a CUDA device')
parser.add_argument('--mu', default=0.1, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser_nesterov_group = parser.add_mutually_exclusive_group()
parser_nesterov_group.add_argument('--nesterov', dest='nesterov', action='store_true', help='enable Nesterov momentum',default=True)
parser_nesterov_group.add_argument('--no-nesterov', dest='nonesterov', action='store_true', help='disable Nesterov momentum',default=False)
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--weight_decay', default=0.0005, type=float, help='weight decay')
parser.add_argument('--mu_scale', default=[1.0,0.2,0.04,0.008], nargs='+', type=float, help='learning rate scaling')
parser.add_argument('--mu_epoch', default=[60,120,160,200], nargs='+', type=int, help='learning rate scheduling')
parser.add_argument('--milestone', default=[40,100,150], nargs='+', type=int, help='the model trained after these epochs will be saved')
parser.add_argument('--print_every_n_batch', default=50, type=int, help='print the training status every n batch')
parser.add_argument('--seed_list', default=[0,1,2,3,4,5,6], nargs='+', type=int, help='train a model with different random seeds')
parser.add_argument('--model', default='CIFAR10_Standard_ResNet_110', choices=list(model_dict.keys()), help='the DNN model')
args = parser.parse_args()

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
print(torch.cuda.get_device_name(int(args.device[-1]))) # print the GPU

trainset = data_dict[args.data]().trainset

for seed in args.seed_list:
    name = args.model+'_seed='+str(seed)
    tp = TrainParam(
        mu=args.mu,
        mu_scale=args.mu_scale,
        mu_epoch=args.mu_epoch,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        batch_size = args.batch_size,
        nesterov = args.nesterov and not args.nonesterov
        )
    c = TrainClassifier(
        name=name,
        net=model_dict[args.model],
        tp=tp,
        trainset=trainset,
        validationset=trainset,
        device=args.device,
        seed=seed,
        resume=args.resume and not args.noresume,
        checkpoint_folder = args.checkpoint_folder,
        milestone = args.milestone,
        print_every_n_batch = args.print_every_n_batch
    ).train()
    print('training for the model '+name+' is completed')
