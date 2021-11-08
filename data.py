"""
Created on Sat Mar 6 2021

@author: Kuan-Lin Chen
"""
import torch
import torchvision
import torchvision.transforms as transforms

class Data_CIFAR10:
    def __init__(self):
        torch.manual_seed(0)
        mean = (125.3/255.0, 123.0/255.0, 113.9/255.0)
        std = (63.0/255.0, 62.1/255.0, 66.7/255.0)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32,4,padding_mode='reflect'),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,std)])
        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=train_transform)
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=test_transform)

class Data_CIFAR100:
    def __init__(self):
        torch.manual_seed(0)
        mean = (129.3/255.0, 124.1/255.0, 112.4/255.0)
        std = (68.2/255.0, 65.4/255.0, 70.4/255.0)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32,4,padding_mode='reflect'),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,std)])
        self.trainset = torchvision.datasets.CIFAR100(root='./data', train=True,download=True, transform=train_transform)
        self.testset = torchvision.datasets.CIFAR100(root='./data', train=False,download=True, transform=test_transform)

def visualization(data):
    import matplotlib.pyplot as plt
    import numpy as np   
 
    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    batch_size = 20
    
    trainloader = torch.utils.data.DataLoader(data.trainset, batch_size=batch_size,shuffle=True, num_workers=4,drop_last=False)
    testloader = torch.utils.data.DataLoader(data.testset, batch_size=batch_size,shuffle=False, num_workers=4,drop_last=False)
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    
    print('dimension of a batch of input data:') 
    print(images.shape)
    print('dimension of a batch of output data:')
    print(labels.shape)
    print('input data:')    
    print(images)

    print('output data (lables):')
    print(' '.join('%5d' % labels[j] for j in range(20)))
    # show images
    imshow(torchvision.utils.make_grid(images))

def compute_mean_and_std(trainset):
    import numpy as np
    x = np.concatenate([np.asarray(trainset[i][0]) for i in range(len(trainset))])
    mean = np.mean(x, axis=(0, 1))
    std = np.std(x, axis=(0, 1))
    print('mean computed from the training set:')
    print(np.round(mean,1))
    print('std computed from the training set:')
    print(np.round(std,1))

data_dict = {'CIFAR10':Data_CIFAR10,'CIFAR100':Data_CIFAR100}

if __name__ == '__main__':
    compute_mean_and_std(torchvision.datasets.CIFAR100(root='./data', train=True,download=True))
    visualization(Data_CIFAR100())
