import numpy as np
from torch.utils import data

class HSIDataSet(data.Dataset):
    def __init__(self, dataID, setindex='label',max_iters=None,num_unlabel=1000):

        self.setindex = setindex
        if dataID==1:            
            self.root = './dataset/PaviaU/'
        elif dataID==2:            
            self.root = './dataset/Salinas/'
        elif dataID==3:            
            self.root = './dataset/Houston/'
        
        XP = np.load(self.root+'XP.npy')
        X = np.load(self.root+'X.npy')
        Y = np.load(self.root+'Y.npy')-1

        if self.setindex=='label':
            train_array = np.load(self.root+'train_array.npy')            
            self.XP = XP[train_array]
            self.X = X[train_array]
            self.Y = Y[train_array]            
            if max_iters != None:
                n_repeat = int(max_iters / len(self.Y))
                part_num = max_iters-n_repeat*len(self.Y)
                self.XP = np.concatenate((np.tile(self.XP,(n_repeat,1,1,1)),self.XP[:part_num]))
                self.X = np.concatenate((np.tile(self.X,(n_repeat,1)),self.X[:part_num]))
                self.Y = np.concatenate((np.tile(self.Y,n_repeat),self.Y[:part_num]))
        elif self.setindex=='unlabel':            
            unlabel_array = np.load(self.root+'unlabel_array.npy')
            self.XP = XP[unlabel_array[0:num_unlabel]]
            self.X = X[unlabel_array[0:num_unlabel]]
            self.Y = Y[unlabel_array[0:num_unlabel]]
        elif self.setindex=='test':
            test_array = np.load(self.root+'test_array.npy')
            self.XP = XP[test_array]
            self.X = X[test_array]
            self.Y = Y[test_array]
        elif self.setindex=='wholeset':            
            self.XP = XP
            self.X = X
            
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if (self.setindex=='label') or (self.setindex=='test') or (self.setindex=='unlabel'):        
            XP = self.XP[index].astype('float32')
            X = self.X[index].astype('float32')
            Y = self.Y[index].astype('int')            
            return XP.copy(),X,Y
        else:            
            XP = self.XP[index].astype('float32')
            X = self.X[index].astype('float32')  
            return XP.copy(),X
