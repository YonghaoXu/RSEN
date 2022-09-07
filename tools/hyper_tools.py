import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio  
import torch

def featureNormalize(X,type):
    # type==1: x = (x-mean)/std(x)
    # type==2: x = (x-max(x))/(max(x)-min(x))
    if type==1:
        mu = np.mean(X,0)
        X_norm = X-mu
        sigma = np.std(X_norm,0)
        X_norm = X_norm/sigma
        return X_norm
    elif type==2:
        minX = np.min(X,0)
        maxX = np.max(X,0)
        X_norm = X-minX
        X_norm = X_norm/(maxX-minX)
        return X_norm    
    
def PCANorm(X,num_PC):
    mu = np.mean(X,0)
    X_norm = X-mu
    
    Sigma = np.cov(X_norm.T)
    [U, _, _] = np.linalg.svd(Sigma)   
    XPCANorm = np.dot(X_norm,U[:,0:num_PC])
    return XPCANorm
    
def MirrowCut(X,hw):
    # X size: row * column * num_feature

    [row,col,n_feature] = X.shape

    X_extension = np.zeros((3*row,3*col,n_feature))
    
    for i in range(0,n_feature):
        lr = np.fliplr(X[:,:,i])
        ud = np.flipud(X[:,:,i])
        lrud = np.fliplr(ud)
        
        l1 = np.concatenate((lrud,ud,lrud),axis=1)
        l2 = np.concatenate((lr,X[:,:,i],lr),axis=1)
        l3 = np.concatenate((lrud,ud,lrud),axis=1)
        
        X_extension[:,:,i] = np.concatenate((l1,l2,l3),axis=0)
    
    X_extension = X_extension[row-hw:2*row+hw,col-hw:2*col+hw,:]
    
    return X_extension
    
    
def DrawResult(labels,imageID):
    # ID=1: Pavia University  
    # ID=2: Salinas    
    # ID=3: Houston
  
    num_class = int(labels.max())
    if imageID == 1:
        row = 610
        col = 340
        palette = np.array([[216,191,216],
                            [0,255,0],
                            [0,255,255],
                            [45,138,86],
                            [255,0,255],
                            [255,165,0],
                            [159,31,239],
                            [255,0,0],
                            [255,255,0]])
        palette = palette*1.0/255
   
    elif imageID ==2:
        row = 512
        col = 217
        palette = np.array([[37, 58, 150],
                            [47, 78, 161],
                            [56, 87, 166],
                            [56, 116, 186],
                            [51, 181, 232],
                            [112, 204, 216],
                            [119, 201, 168],
                            [148, 204, 120],
                            [188, 215, 78],
                            [238, 234, 63],
                            [246, 187, 31],
                            [244, 127, 33],
                            [239, 71, 34],
                            [238, 33, 35],
                            [180, 31, 35],
                            [123, 18, 20]])
        palette = palette*1.0/255  

    elif imageID == 3:
        row = 349
        col = 1905
        palette = np.array([[0, 205, 0],
                            [127, 255, 0],
                            [46, 139, 87],
                            [0, 139, 0],
                            [160, 82, 45],
                            [0, 255, 255],
                            [255, 255, 255],
                            [216, 191, 216],
                            [255, 0, 0],
                            [139, 0, 0],
                            [0, 0, 0],
                            [255, 255, 0],
                            [238, 154, 0],
                            [85, 26, 139],
                            [255, 127, 80]])
        palette = palette*1.0/255
    
    X_result = np.zeros((labels.shape[0],3))
    for i in range(1,num_class+1):
        X_result[np.where(labels==i),0] = palette[i-1,0]
        X_result[np.where(labels==i),1] = palette[i-1,1]
        X_result[np.where(labels==i),2] = palette[i-1,2]
    
    X_result = np.reshape(X_result,(row,col,3))
    plt.axis ( "off" ) 
    plt.imshow(X_result)    
    return X_result
    
def CalAccuracy(predict,label):
    n = label.shape[0]
    OA = np.sum(predict==label)*1.0/n
    correct_sum = np.zeros((max(label)+1))
    reali = np.zeros((max(label)+1))
    predicti = np.zeros((max(label)+1))
    producerA = np.zeros((max(label)+1))
    
    for i in range(0,max(label)+1):
        correct_sum[i] = np.sum(label[np.where(predict==i)]==i)
        reali[i] = np.sum(label==i)
        predicti[i] = np.sum(predict==i)
        producerA[i] = correct_sum[i] / reali[i]
   
    Kappa = (n*np.sum(correct_sum) - np.sum(reali * predicti)) *1.0/ (n*n - np.sum(reali * predicti))
    return OA,Kappa,producerA

def ExtractPatches(X,w):
    
    hw = int(w/2)
    
    [row,col,n_feature] = X.shape
    K = row*col
    X_Mirrow = MirrowCut(X,hw)
    
    XP = np.zeros((K,w,w,n_feature)).astype('float32')
    
    for i in range(1,K+1):
        index_row = int(np.ceil(i*1.0/col))
        index_col = i - (index_row-1)*col + hw -1 
        index_row += hw -1
        patch = X_Mirrow[index_row-hw:index_row+hw,index_col-hw:index_col+hw,:]
        XP[i-1,:,:,:] = patch
    
    XP = np.moveaxis(XP, 3, 1)    
    return XP.astype('float32')  

def SampleGen(dataID=1,w=16,n_PC=3):
    # ID=1: Pavia University
    # ID=2: Salinas
    # ID=3: Houston
    if dataID==1:        
        data = sio.loadmat('./dataset/PaviaU.mat')
        X = data['paviaU']
    
        data = sio.loadmat('./dataset/PaviaU_gt.mat')
        Y = data['paviaU_gt']
        
    elif dataID==2:        
        data = sio.loadmat('./dataset/Salinas_corrected.mat')
        X = data['salinas_corrected']
    
        data = sio.loadmat('./dataset/Salinas_gt.mat')
        Y = data['salinas_gt']        

    elif dataID==3:        
        data = sio.loadmat('./dataset/Houston.mat')
        X = data['Houston']
    
        data = sio.loadmat('./dataset/Houston_GT.mat')
        Y = data['Houston_GT']
        
    [row,col,n_feature] = X.shape
    K = row*col
    X = X.reshape(row*col, n_feature)       
    
    X_PCA = featureNormalize(PCANorm(X,n_PC),1)  
    X_PCA = X_PCA.reshape(row,col, n_PC)  

    X = featureNormalize(X,1)  
    # XP: k*n_PC*w*w
    XP = ExtractPatches(X_PCA,w)
    
    Y = Y.reshape(row*col,)
    return XP,X,Y


def test_acc(model, data_loader, epoch,num_classes, print_per_batches=10):

    model.eval()
    class_name_list = list(range(0,num_classes))
    num_batches = len(data_loader)

    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    total = 0
    correct = 0
    class_acc = np.zeros((num_classes,1))
    for batch_idx, data in enumerate(data_loader):

        XP, X, Y = data
        XP = XP.cuda()
        X = X.cuda()
        Y = Y.cuda()

        batch_size = XP.size(0)

        outputs = model(XP,X)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == Y).squeeze()
        total += batch_size
        correct += (predicted == Y).sum().item()
        for i in range(batch_size):
            label = Y[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
            
        if (batch_idx+1)%print_per_batches == 0:
            print('Epoch[%d]-Validation-[%d/%d] Batch OA: %.2f %%' % (epoch,batch_idx+1,num_batches,100.0 * (predicted == Y).sum().item() / batch_size))
       
    for i in range(num_classes):
        class_acc[i] = 1.0*class_correct[i] / class_total[i]
        print('---------------Accuracy of %5s : %.2f %%---------------' % (
            class_name_list[i], 100 * class_acc[i])) 

    acc = 1.0*correct / total
    print('---------------Epoch[%d]Validation-OA: %.2f %%---------------' % (epoch,100.0 * acc))
    print('---------------Epoch[%d]Validation-AA: %.2f %%---------------' % (epoch,100.0 * np.mean(class_acc)))
    return acc

def test_whole(model, data_loader, print_per_batches=10):

    model.eval()    
    num_batches = len(data_loader)

    for batch_idx, data in enumerate(data_loader):
        XP, X = data
        XP = XP.cuda()
        X = X.cuda()
      
        outputs = model(XP,X)
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.cpu().numpy()
        if batch_idx==0:
            output = predicted
        else:
            output = np.append(output,predicted)

        if (batch_idx+1)%print_per_batches == 0:
            print('---------------------Testing the whole set-[%d/%d]---------------------' % (batch_idx+1,num_batches))

    return output
    