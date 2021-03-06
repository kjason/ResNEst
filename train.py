"""
Created on Fri Mar 5 2021

@author: Kuan-Lin Chen
"""
import os
import time
import torch
import scipy.io
import math

class TrainParam:
    def __init__(self,
                mu,
                mu_scale,
                mu_epoch,
                weight_decay,
                momentum,
                batch_size,
                nesterov
                ):
        assert len(mu_scale)==len(mu_epoch), "the length of mu_scale and mu_epoch should be the same"        
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.batch_size = batch_size
        self.max_epoch = mu_epoch[-1]
        self.mu = mu
        self.mu_scale = mu_scale
        self.mu_epoch = mu_epoch
        self.nesterov = nesterov
                 
class TrainClassifier:
    num_workers = 1
    pin_memory = False
    ckpt_filename = 'train.pt'
    def __init__(self,
                name,
                net,
                tp,
                trainset,
                validationset,
                device,
                seed,
                resume,
                checkpoint_folder,
                milestone = [],
                print_every_n_batch = 1
                ):
        torch.manual_seed(seed)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        net = net()
        self.checkpoint_folder = checkpoint_folder
        self.name = name
        self.seed = seed
        self.milestone = milestone
        self.print_every_n_batch = print_every_n_batch
        self.net = net.to(device)
        self.num_parameters = self.count_parameters()
        print(f"Number of parameters: {self.num_parameters:,}")

        self.mu_lambda = lambda i: next(tp.mu_scale[j] for j in range(len(tp.mu_epoch)) if min(tp.mu_epoch[j]//(i+1),1.0) >= 1.0) if i<tp.max_epoch else 0
        self.rho_lambda = lambda i: next(tp.rho_scale[j] for j in range(len(tp.rho_epoch)) if min(tp.rho_epoch[j]//i,1.0) >= 1.0) if i>0 else 0

        self.trainloader = torch.utils.data.DataLoader(trainset,batch_size=tp.batch_size,shuffle=True,num_workers=self.num_workers,pin_memory=self.pin_memory,drop_last=False)
        self.validationloader = torch.utils.data.DataLoader(validationset,batch_size=tp.batch_size,shuffle=False,num_workers=self.num_workers,pin_memory=self.pin_memory,drop_last=False)
        self.len_validationset = len(validationset)
        self.device = device
        self.optimizer = torch.optim.SGD(self.net.parameters(),lr=tp.mu,momentum=tp.momentum,nesterov=tp.nesterov,weight_decay=tp.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda=self.mu_lambda)
        self.tp = tp
        self.total_train_time = 0
        self.start_epoch = 1
        self.train_loss = []
        self.train_acc = []
        self.validation_loss = []
        self.validation_acc = []
        self.best_validation_acc = 0
        self.ckpt_path = self.checkpoint_folder+self.name+'/'+self.ckpt_filename

        if resume is True and os.path.isfile(self.ckpt_path):
            print('Resuming '+self.name+' from a checkpoint at '+self.ckpt_path)
            self.__load()
        else:
            print('Ready to train '+self.name+' from scratch...')
            _,init_validation_acc = self.validation()
            self.init_validation_acc = init_validation_acc
            self.best_validation_acc = init_validation_acc
            self.__save_net('init_model.pt')
            self.__save(0)
 
    def __get_lr(self):
            for param_group in self.optimizer.param_groups:
                return param_group['lr']

    def __check_folder(self):
        if not os.path.isdir(self.checkpoint_folder):
            os.mkdir(self.checkpoint_folder)
        if not os.path.isdir(self.checkpoint_folder+self.name):
            os.mkdir(self.checkpoint_folder+self.name)

    def __load(self):
        # Load checkpoint.
        checkpoint = torch.load(self.ckpt_path,map_location=self.device)
        self.net.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.best_validation_acc = checkpoint['best_validation_acc']
        self.start_epoch = checkpoint['epoch']+1
        self.train_loss = checkpoint['train_loss']
        self.train_acc = checkpoint['train_acc']
        self.validation_loss = checkpoint['validation_loss']
        self.validation_acc = checkpoint['validation_acc']
        self.total_train_time = checkpoint['total_train_time']
        self.init_validation_acc = checkpoint['init_validation_acc']

    def __save_net(self,filename):
        self.__check_folder()
        net_path = self.checkpoint_folder+self.name+'/'+filename
        torch.save(self.net.state_dict(), net_path)
        print('model saved at '+net_path)

    def __save(self,epoch):
        state = {
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'init_validation_acc': self.init_validation_acc,
            'best_validation_acc': self.best_validation_acc,
            'epoch': epoch,
            'train_loss': self.train_loss,
            'train_acc': self.train_acc,
            'validation_loss': self.validation_loss,
            'validation_acc': self.validation_acc,
            'num_param': self.num_parameters,
            'seed': self.seed,
            'mu': self.tp.mu,
            'mu_scale': self.tp.mu_scale,
            'mu_epoch': self.tp.mu_epoch,
            'weight_decay': self.tp.weight_decay,
            'momentum': self.tp.momentum,
            'batch_size': self.tp.batch_size,
            'total_train_time': self.total_train_time,
            }
        self.__check_folder()
        torch.save(state, self.ckpt_path)
        print('checkpoint saved at '+self.ckpt_path)
        del state['net'], state['optimizer'], state['scheduler']
        state_path = self.checkpoint_folder+self.name+'/train.mat'
        scipy.io.savemat(state_path,state)
        print('state saved at '+state_path)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def train(self):
        print('Start training...')
        for i in range(self.start_epoch,self.tp.max_epoch+1):
            tic = time.time()
            train_loss,train_acc = self.__train(i)
            toc = time.time()
            self.total_train_time += (toc-tic)
            print('training speed: %.3f seconds/epoch'%(self.total_train_time/i))

            validation_loss,validation_acc = self.validation()
            
            self.train_loss.append(train_loss)
            self.train_acc.append(train_acc)            
            self.validation_loss.append(validation_loss)
            self.validation_acc.append(validation_acc)
            
            if validation_acc > self.best_validation_acc:
                self.best_validation_acc = validation_acc
                self.__save_net('best_model.pt')

            for k in self.milestone:
                if k==i:
                    self.__save_net('epoch_'+str(k)+'_model.pt')
                    self.__save(k)

            if math.isnan(train_loss):
                break

        if self.start_epoch<self.tp.max_epoch+1:
            self.__save_net('last_model.pt')
            self.__save(i)
            print('end of training at epoch %d'%(i))
        else:
            print('the model '+self.ckpt_path+' has already been trained for '+str(self.tp.max_epoch)+' epochs')
        return self
    
    def __train(self,epoch_idx):
        tic = time.time()
        self.net.train()
        accumulated_train_loss = 0
        correct = 0
        total = 0
        torch.manual_seed(self.seed+epoch_idx)
        lr = self.__get_lr()
        num_batch = len(self.trainloader)
        for batch_idx, (inputs, targets) in enumerate(self.trainloader,1):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            batch_mean_loss = torch.mean(loss)

            if torch.isnan(batch_mean_loss):
                return float("nan"),float("nan")

            batch_mean_loss.backward()
            self.optimizer.step()

            accumulated_train_loss += torch.sum(loss).item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            train_loss = accumulated_train_loss/total
            train_acc = 100.*correct/total
            toc = time.time()
            if (batch_idx-1)%self.print_every_n_batch == 0 or batch_idx == num_batch:
                print('[%s epoch: %d/%d batch: %d/%d lr: %.1e] Loss: %.4f | Acc: %.3f%% (%d/%d) | elapsed: %.2fs'%(
                    self.name,
                    epoch_idx,
                    self.tp.max_epoch,
                    batch_idx,
                    num_batch,
                    lr,
                    train_loss,
                    train_acc,
                    correct,
                    total,
                    self.total_train_time+toc-tic
                    ))
        self.scheduler.step()
        return train_loss,train_acc
    
    def validation(self):
        self.net.eval()
        accumulated_validation_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.validationloader,1):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                
                accumulated_validation_loss += torch.sum(loss).item()
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        validation_loss = accumulated_validation_loss/self.len_validationset
        
        validation_acc = 100.*correct/total
        
        print('[Validation] loss: %.4f | Acc: %.3f%% (%d/%d) | Best Acc: %.3f%%'%(
            validation_loss,
            validation_acc,
            correct,
            total,
            max(self.best_validation_acc,validation_acc)))
        
        return validation_loss,validation_acc
    
if __name__ == '__main__':
    from data import Data_CIFAR10
    from models import *
    from test import *
   
    # setting
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0)) # print the GPU
    cifar10 = Data_CIFAR10()
    seed = 0

    tp = TrainParam(
        mu=0.05,
        mu_scale=[1.0,0.2],#[1.0,0.2,0.04,0.008],
        mu_epoch=[2,4],#[60,120,160,200],
        weight_decay=5e-4,
        momentum=0.9,
        batch_size = 128,
        nesterov = True
        )
    
    # initiate a model
    c = TrainClassifier(
        name='CIFAR10_Standard_ResNet_110_seed='+str(seed),
        net=CIFAR10_Standard_ResNet_110,
        tp=tp,
        trainset=cifar10.trainset,
        validationset=cifar10.trainset,
        device="cuda:0",
        seed=seed,
        resume=False,
        checkpoint_folder = './checkpoint-dryrun/',
        milestone = [40,100,160],
        print_every_n_batch = 1
        )
    
    # train
    c.train()

    # load the model after training
    net = CIFAR10_Standard_ResNet_110()
    pretrained_model = torch.load(c.checkpoint_folder+c.name+'/last_model.pt')
    net.load_state_dict(pretrained_model,strict=True)
    net = net.to("cuda:0") 

    # test
    testClassifier(net,cifar10.testset,"cuda:0",c.checkpoint_folder+c.name)
