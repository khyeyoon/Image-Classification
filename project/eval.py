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
import glob
from PIL import Image

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

class EvalDataset(Dataset): 
    def __init__(self, path, train=True, transform=None): 
        self.path = path + '/eval/'
        self.img_list = glob.glob(self.path + '/*.jpg') 
        self.list = []
        if transform:
            self.transform=transform
            
    def __len__(self): 
        return len(self.img_list) 
    
    def __getitem__(self, idx): 
        img_path = self.img_list[idx]  
        img_origin = Image.open(img_path) 
        self.list.append(img_path)
        
        if self.transform:
            img_trans = self.transform(img_origin)
            
        return img_trans
    
    def image_path_list(self):
        return self.list
        

transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
])

test_data = EvalDataset(path='.',transform=transform_test)
test_set = DataLoader(dataset = test_data, batch_size=len(test_data),shuffle=False)

net = MyNet()
checkpoint = torch.load('./checkpoint/resnet18_e_30_0.0012421748833730817.pt')
net.load_state_dict(checkpoint['model_state_dict'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = net.to(device)
net.eval()

def evaluation():
    with torch.no_grad():
        for batch_idx, (inputs_trans) in enumerate(test_set):
            inputs_trans = inputs_trans.to(device)
            outputs = net(inputs_trans)
            
            # 이미지 분류하여 폴더에 저장
            cnt=1
            for result,path in zip(outputs,test_data.image_path_list()):
                # squirrel
                if result[0]<result[1]:
                    if not os.path.exists("./results/squirrels/"):
                        os.makedirs('./results/squirrels/')
                    image = Image.open(path)
                    name = './results/squirrels/' + str(cnt) +".jpg"
                    image.save(name)
                # quokka
                else:
                    if not os.path.exists("./results/quokkas/"):
                        os.makedirs('./results/quokkas/') 
                    image = Image.open(path)
                    name = './results/quokkas/' + str(cnt) +".jpg"
                    image.save(name)
                print("save " + str(cnt) +  " img")
                cnt+=1
        
evaluation()
