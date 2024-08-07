import cv2
from skimage import img_as_ubyte
import numpy as np
from ColorSpace import colorSpace as cp

import torch
import torch.nn as nn
import torch.nn.functional as F


import os
import natsort
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

import time
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

def Enhancement(imgx, device):
    rows, columns, dimension = imgx.shape

    HSVInput  = cp.RgbToHsv(imgx)

    #HSV Components
    hueComponent = HSVInput[:, :, 0]              
    satComponent = HSVInput[:, :, 1]
    valComponent = HSVInput[:, :, 2]

    valComponentTensor = torch.from_numpy(valComponent).float().to(device).view(1, 1, rows, columns)

    image = transform(Image.fromarray(imgx)).unsqueeze(0)

    modelClass.eval()
    with torch.no_grad():
        output = modelClass(image.to(device))
        _, predicted = torch.max(output.data, 1)
        if predicted == 1:
           valEnhancement  = modelValL(valComponentTensor)
        else:
           valEnhancement  = modelValG(valComponentTensor)

    rows1, columns1 = valEnhancement.shape[2:4]
    valEnhComponent = valEnhancement.detach().cpu().numpy().reshape([rows1, columns1])

    HSV = np.dstack((hueComponent, satComponent, valEnhComponent))
    algorithm = cp.HsvToRgb(HSV)
    return  algorithm

def bgr_rgb(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

class RGBTargetDataset(Dataset):
    def __init__(self, valImgDir):
        self.valImgDir = valImgDir
        self.image_files = natsort.natsorted(os.listdir(valImgDir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        validationImg = cv2.imread(os.path.join(self.valImgDir, image_file))
        validationImg = bgr_rgb(validationImg)
        return validationImg

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(in_features=64*56*56, out_features=128)
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(in_features=128, out_features=2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = x.view(-1, 64*56*56)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        return x
    
class Enhance(nn.Module):

    def __init__(self):
        super(Enhance, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= 1, out_channels=32,   kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels= 32,       out_channels=64,   kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels= 64,       out_channels=128,  kernel_size=3, padding=1)


        self.conv4 = nn.Conv2d(in_channels= 128, out_channels=256, kernel_size=3, padding=1)

        
        self.conv5 = nn.Conv2d(in_channels= 256, out_channels=128, kernel_size=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels= 256, out_channels=64,  kernel_size=1, padding=0)
        self.conv7 = nn.Conv2d(in_channels= 128, out_channels=32,   kernel_size=1, padding=0)
        self.conv8 = nn.Conv2d(in_channels= 64, out_channels=1,    kernel_size=1, padding=0)
        self.conv9 = nn.Conv2d(in_channels= 32, out_channels=1,   kernel_size=1, padding=0)


    def forward(self, x):              
        x1 = F.relu(self.conv1(x))     
        x2 = F.relu(self.conv2(x1))    
        x3 = F.relu(self.conv3(x2))    

        x4 = F.relu(self.conv4(x3))   

        x5 = F.relu(self.conv5(x4))    
        x6 = torch.cat([x3,x5], dim=1) 

        x7 = F.relu(self.conv6(x6))  
        x8 = torch.cat([x2,x7], dim=1) 

        x9 = F.relu(self.conv7(x8))     
        x10 = torch.cat([x1,x9], dim=1) 

        x11 = F.relu(self.conv8(x10))   

        return x11


modelClass = Classification().to(device)
modelClass.load_state_dict(torch.load('./Models/CLASSIFICATION.pt'))

modelValL = Enhance().to(device)
modelValL.load_state_dict(torch.load('./Models/LOCAL.pt'))
modelValL.eval()

modelValG = Enhance().to(device)
modelValG.load_state_dict(torch.load('./Models/GLOBAL.pt'))
modelValG.eval()

enhancementSet   = "./1_Input"

dataset = RGBTargetDataset(enhancementSet)


starBatch = 0
endBatch = len(dataset)
sizeBatch = endBatch - starBatch

progress_bar = tqdm(range(starBatch, endBatch), desc="Enhancing images")

start_time = time.time()
for i in progress_bar:
    orginalImg = dataset[i]

    algorithm = Enhancement(orginalImg, device)

    imgsave = np.dstack((algorithm[:, :, 2], algorithm[:, :, 1], algorithm[:, :, 0]))
    cv2.imwrite(os.path.join('2_Output', dataset.image_files[i]), imgsave * 255)

    # Print the processing time for the current image
    elapsed_time = time.time() - start_time
    progress_bar.set_description("Image {0} enhanced in {1:.2f} seconds".format(i + 1, elapsed_time))
    progress_bar.refresh()
elapsed_time = time.time() - start_time  
print(f"Time: {elapsed_time} seconds.")  