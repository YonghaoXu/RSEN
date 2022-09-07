import torch
from torch import nn

class BaseNet(nn.Module):  
    def __init__(self,num_features=103, dropout=0, num_classes=0):
        super(BaseNet, self).__init__()

        self.conv0 = nn.Conv2d(5, 64, kernel_size=1, stride=1,
                               bias=True)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
      
        self.num_features = num_features
        
        self.dropout = dropout
        self.drop = nn.Dropout(self.dropout)
        self.num_classes = num_classes

        # Append new layers
        n_fc1 = 1024
        n_fc2 = 512
        
        self.feat_spe = nn.Linear(self.num_features, n_fc1)
        self.feat_ss = nn.Linear(n_fc1+n_fc1, n_fc2)
        
        self.classifier = nn.Linear(n_fc2, self.num_classes)


    def forward(self, x,y):        
        x = self.conv0(x)
        x_res = x
        x = self.conv1(x)
        x = self.relu(x+x_res)
        x = self.avgpool(x)
        x_res = x
        x = self.conv2(x)
        x = self.relu(x+x_res)
        x = self.avgpool(x)        
        
        x = x.view(x.size(0), -1)

        y = self.feat_spe(y)   
        y = self.relu(y)      

        x = torch.cat([x,y],1)
        x = self.feat_ss(x)       
        x = self.relu(x)
        
        if self.dropout > 0:
            x = self.drop(x)

        x = self.classifier(x)     

        return x

def WeightEMA_BN(Base,Ensemble,alpha):
    one_minus_alpha = 1.0 - alpha
    Base_params = Base.state_dict().copy()
    Ensemble_params = Ensemble.state_dict().copy()
    
    for b,e in zip(Base_params,Ensemble_params):
        Ensemble_params[e] = Base_params[b] * one_minus_alpha + Ensemble_params[e] * alpha
    
    Ensemble.load_state_dict(Ensemble_params)
    return Ensemble

