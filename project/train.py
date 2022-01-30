import numpy as np
import torch
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse
from utils import progress_bar
import wandb

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.fc2 = nn.Linear(1000,2)
    def forward(self,x):
        x = self.model(x)
        return self.fc2(x)
    
net = MyNet()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = net.to(device)
# print(net)

print('==> Preparing data..')
transform_train = transforms.Compose([
                                # transforms.CenterCrop(32),
                                # transforms.RandomCrop(32, padding=4),
                                # transforms.RandomHorizontalFlip(),
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])

transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
])

EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 0.0001

config={"epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate" : LEARNING_RATE}

wandb.init(project="classification_project", entity='hyeyoon')


train_data = torchvision.datasets.ImageFolder(root='/workspace/data/train',transform=transform_train)
test_data = torchvision.datasets.ImageFolder(root='/workspace/data/test',transform=transform_test)

train_set = DataLoader(dataset = train_data, batch_size=16, shuffle = True, num_workers=2)
test_set = DataLoader(dataset = test_data, batch_size=len(test_data))

# optimizer = optim.Adam(net.parameters(),lr=0.0001)
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-6)
criterion = nn.CrossEntropyLoss()

def adjust_learning_rate(optimizer, epoch, LR):

    if epoch <= 10:
        lr = LR 
            
    elif epoch <= 15:
        lr = LR * 0.5
            
    else:
        lr = LR * 0.1   
            
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
      

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    print('LR: {:0.6f}'.format(optimizer.param_groups[0]['lr']))
    for batch_idx, (inputs, targets) in enumerate(train_set):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_set), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # print(total)
        
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # 모델 저장
    if epoch%5==0:
      torch.save({
          'epoch': epoch,
          'model_state_dict': net.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'loss': train_loss/(batch_idx+1),
          "acc": 100*correct/total
          }, f"/workspace/checkpoint/resnet18_e_{epoch}_{loss}.pt")
      
    wandb.log({'train_accuracy': 100*correct/total, 'train_loss': train_loss/(batch_idx+1)})

        
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_set):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_set), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        wandb.log({'test_accuracy': 100*correct/total, 'test_loss': test_loss/(batch_idx+1)})
   
start_epoch = 1
for epoch in range(start_epoch, start_epoch+EPOCHS):
    adjust_learning_rate(optimizer, epoch, LR=LEARNING_RATE)
    
    train(epoch)
    test(epoch)
