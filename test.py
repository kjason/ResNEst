"""
Created on Fri Mar 5 2021

@author: Kuan-Lin Chen
"""
import torch
import scipy.io

def testClassifier(net,testset,device,model_folder):
    testloader = torch.utils.data.DataLoader(testset,batch_size=100,shuffle=False,num_workers=1,pin_memory=False,drop_last=False)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader,1):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += torch.sum(loss).item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_test_loss = test_loss/len(testset)

    test_acc = 100.*correct/total

    print('[Test] loss: %.4f | Acc: %.3f%% (%d/%d)'%(
        avg_test_loss,
        test_acc,
        correct,
        total))

    test_result = {'avg_test_loss':avg_test_loss,'test_acc':test_acc,'correct':correct,'total':total}
    test_result_path = model_folder+'/test.mat'
    scipy.io.savemat(test_result_path,test_result)
    return avg_test_loss,test_acc
