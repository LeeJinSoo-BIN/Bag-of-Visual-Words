import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook, tqdm
import kmc2
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn import svm
import torch
from torch import nn
import torch.nn.functional as F

path = './BoWdata'
codebook_len = 600
img_size = 240
jump = 6


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        self.conv1_1_vis = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True)
        self.conv1_1_bn_vis = nn.BatchNorm2d(64, affine=True)
        self.conv1_2_vis = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.conv1_2_bn_vis = nn.BatchNorm2d(64, affine=True)
        
        self.pool1_vis = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv2_1_vis = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.conv2_1_bn_vis = nn.BatchNorm2d(128, affine=True)
        self.conv2_2_vis = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.conv2_2_bn_vis = nn.BatchNorm2d(128, affine=True)

        self.pool2_vis = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv3_1_vis = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_1_bn_vis = nn.BatchNorm2d(256, affine=True)
        self.conv3_2_vis = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_2_bn_vis = nn.BatchNorm2d(256, affine=True)
        self.conv3_3_vis = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_3_bn_vis = nn.BatchNorm2d(256, affine=True)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) 
        
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        self.conv4_1_bn = nn.BatchNorm2d(512, affine=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv4_2_bn = nn.BatchNorm2d(512, affine=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv4_3_bn = nn.BatchNorm2d(512, affine=True)
        self.pool4 = nn.MaxPool2d(kernel_size=3,padding=1,stride=1, ceil_mode=True)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv5_1_bn = nn.BatchNorm2d(512, affine=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv5_2_bn = nn.BatchNorm2d(512, affine=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv5_3_bn = nn.BatchNorm2d(512, affine=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) 

        self.conv6_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv6_1_bn = nn.BatchNorm2d(512, affine=True)  
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=1)

        self.conv7_1 = nn.Conv2d(512, 256, kernel_size=1, stride=2)        
        self.conv7_1_bn = nn.BatchNorm2d(256, affine=True)       
        self.conv7_2 = nn.Conv2d(256, 1024, kernel_size=3, padding=1, bias=True)

        self.linear_1 = nn.Linear(1024,4096)
        self.linear_2 = nn.Linear(4096,102)
        

        self.load_pretrained_layers()

    def forward(self, image_vis, image_lwir):
        #import pdb; pdb.set_trace()
        out_vis = F.relu(self.conv1_1_bn_vis(self.conv1_1_vis(image_vis))) 
        out_vis = F.relu(self.conv1_2_bn_vis(self.conv1_2_vis(out_vis)))   
        out_vis = self.pool1_vis(out_vis)  

        out_vis = F.relu(self.conv2_1_bn_vis(self.conv2_1_vis(out_vis)))
        out_vis = F.relu(self.conv2_2_bn_vis(self.conv2_2_vis(out_vis))) 
        out_vis = self.pool2_vis(out_vis) 

        out_vis = F.relu(self.conv3_1_bn_vis(self.conv3_1_vis(out_vis)))
        out_vis = F.relu(self.conv3_2_bn_vis(self.conv3_2_vis(out_vis)))
        out_vis = F.relu(self.conv3_3_bn_vis(self.conv3_3_vis(out_vis)))
        
        out_vis = self.pool3(out_vis)

        out_vis = F.relu(self.conv4_1_bn(self.conv4_1(out_vis))) 
        out_vis = F.relu(self.conv4_2_bn(self.conv4_2(out_vis))) 
        out_vis = F.relu(self.conv4_3_bn(self.conv4_3(out_vis))) 
        out_vis = self.pool4(out_vis)

        out_vis = F.relu(self.conv5_1_bn(self.conv5_1(out_vis))) 
        out_vis = F.relu(self.conv5_2_bn(self.conv5_2(out_vis))) 
        out_vis = F.relu(self.conv5_3_bn(self.conv5_3(out_vis))) 
        out_vis = self.pool5(out_vis)
        
        out_vis = F.relu(self.conv6_1_bn(self.conv6_1(out_vis))) 
        out_vis = F.relu(self.conv6_2(out_vis))

        out_vis = F.relu(self.conv7_1_bn(self.conv7_1(out_vis))
        out_vis = self.conv7_2(out_vis)
        out_vis = F.relu(self.linear_1(out_vis))
        predict = F.relu(self.linear_2(out_vis))

        return(predict)

    def load_pretrained_layers(self):

        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        pretrained_state_dict = torchvision.models.vgg16_bn(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())
        
        for i, param in enumerate(param_names[:49]):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        for i, param in enumerate(param_names[49:-22]):    
                state_dict[param] = pretrained_state_dict[pretrained_param_names[i+49]]

        self.load_state_dict(state_dict)


optimizer = torch.optim.Adadelta(net.parameters(),lr = 1e-2)
model = VGG().to(device)



for 
out = model()
loss = nn.CrossEntropyLoss(out,label).to(device)
loss.backward()
optimizer.step()
for 
