"""
Created on Sat Mar 6 2021

@author: Kuan-Lin Chen
"""
import os
import torch

def dir_path(path):
    if os.path.isdir(path) and path[-1]=='/':
        return path
    else:
        raise NotADirectoryError(path)

def cuda_device(device):
    error_msg = 'not found in the available cuda list'
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        for i in range(count):
            if device == 'cuda:'+str(i):
                return device
        raise ValueError(device+error_msg)
    else:
        raise ValueError(device+error_msg)
