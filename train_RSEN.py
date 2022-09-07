import torch
from tools.hyper_tools import *
import argparse
import time
from tools.models import *
from torch.nn import functional as F
from dataset.hsi_loader import HSIDataSet
from torch.utils import data
import os

DataName = {1:'PaviaU',2:'Salinas',3:'Houston'}

def main(args):
    if args.dataID==1:
        num_classes = 9
        num_features = 103
        save_pre_dir = './dataset/PaviaU/'
    elif args.dataID==2:
        num_classes = 16  
        num_features = 204  
        save_pre_dir = './dataset/Salinas/'
    elif args.dataID==3:
        num_classes = 15
        num_features = 144
        save_pre_dir = './dataset/Houston/'

    Y = np.load(save_pre_dir+'Y.npy')-1
    test_array = np.load(save_pre_dir+'test_array.npy')
    Y = Y[test_array]
    print_per_batches = args.print_per_batches

    save_path_prefix = args.save_path_prefix+'Experiment_'+DataName[args.dataID]+\
                                             '/label_'+repr(args.num_label)+'/'
    
    if os.path.exists(save_path_prefix)==False:
        os.makedirs(save_path_prefix)

    labeled_loader = data.DataLoader(
        HSIDataSet(args.dataID, setindex='label', max_iters=args.num_unlabel),
        batch_size=args.labeled_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    unlabeled_loader = data.DataLoader(
        HSIDataSet(args.dataID, setindex='unlabel', max_iters=None, num_unlabel=args.num_unlabel),
        batch_size=args.unlabeled_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    whole_loader = data.DataLoader(
        HSIDataSet(args.dataID, setindex='wholeset', max_iters=None),
        batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    Ensemble = BaseNet(num_features=num_features,
                        dropout=args.dropout,
                        num_classes=num_classes,
                        )
    Base = BaseNet(num_features=num_features,
                        dropout=args.dropout,
                        num_classes=num_classes,
                        )

    Ensemble = torch.nn.DataParallel(Ensemble).cuda()
    Base = torch.nn.DataParallel(Base).cuda()

    cls_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()

    base_optimizer = torch.optim.Adam(Base.parameters(),lr=args.lr)

    num_batches = min(len(labeled_loader),len(unlabeled_loader))

    # freeze the parameters in the ensemble model
    ensemble_params = list(Ensemble.parameters())
    for param in ensemble_params:
        param.requires_grad = False

    num_steps = args.num_epochs*num_batches
    loss_hist = np.zeros((num_steps,5))
    index_i = -1
    for epoch in range(args.num_epochs):
        decay_adv = (1 - epoch/args.num_epochs)
        num_certainty = int(np.exp(-1*decay_adv**2)*args.unlabeled_batch_size+0.99)
        
        for batch_index, (labeled_data, unlabeled_data) in enumerate(zip(labeled_loader, unlabeled_loader)):
            index_i += 1            
            tem_time = time.time()
            base_optimizer.zero_grad()

            # train with labeled data
            Base.eval()
            XP_train, X_train, Y_train = labeled_data
            XP_train = XP_train.cuda() + torch.randn(XP_train.size()).cuda() * args.noise
            X_train = X_train.cuda() + torch.randn(X_train.size()).cuda() * args.noise
            Y_train = Y_train.cuda()            
            labeled_output = Base(XP_train,X_train)

            # ce loss
            cls_loss_value = cls_loss(labeled_output, Y_train)        
            _, labeled_prd_label = torch.max(labeled_output, 1)
            
            # train with unlabeled data
            XP_un, X_un, _ = unlabeled_data
            XP_un = XP_un.cuda()
            X_un = X_un.cuda()

            # dropout is activated for stochastic augmentation in self-ensembling learning
            Base.train()
            Ensemble.train()
            XP_b_input = XP_un + torch.randn(XP_un.size()).cuda() * args.noise
            X_b_input = X_un + torch.randn(X_un.size()).cuda() * args.noise

            un_b_output = Base(XP_b_input,X_b_input)  
            un_b_output = F.softmax(un_b_output,dim=1)  
        
            # stochastic augmentation with Gaussian noise
            XP_un_input_re = XP_un.repeat([args.m,1,1,1])
            X_un_input_re = X_un.repeat([args.m,1])
            XP_un_input_re += torch.randn(XP_un_input_re.size()).cuda() * args.noise
            X_un_input_re += torch.randn(X_un_input_re.size()).cuda() * args.noise

            un_e_output_re = Ensemble(XP_un_input_re,X_un_input_re)
            un_e_predicts_re = F.softmax(un_e_output_re, dim=1)
            un_e_predicts_re = un_e_predicts_re.reshape([args.m,-1,num_classes])
            un_e_output = torch.mean(un_e_predicts_re,0)

            # consistency filter
            cons = torch.sum(torch.std(un_e_predicts_re,0),1).cpu().numpy()
            filter = np.argsort(cons)[:num_certainty]
                
            # consistency loss
            con_loss_value = mse_loss(un_b_output[filter], un_e_output[filter])
                        
            total_loss = cls_loss_value + con_loss_value         
            total_loss.backward()

            # update base and ensemble networks
            base_optimizer.step()
            Ensemble = WeightEMA_BN(Base,Ensemble,args.teacher_alpha)
   
            # training stat
            loss_hist[index_i,0] = time.time()-tem_time
            loss_hist[index_i,1] = total_loss.item()
            loss_hist[index_i,2] = cls_loss_value.item() 
            loss_hist[index_i,3] = con_loss_value.item() 
            loss_hist[index_i,4] = torch.mean((labeled_prd_label == Y_train).float()).item() #acc      
            tem_time = time.time()

            if (batch_index+1) % print_per_batches == 0:
                print('Epoch %d/%d:  %d/%d Time: %.2f total_loss = %.4f cls_loss = %.4f con_loss = %.4f acc = %.2f\n'\
                %(epoch+1, args.num_epochs,batch_index+1,num_batches,
                np.mean(loss_hist[index_i-print_per_batches+1:index_i+1,0]),
                np.mean(loss_hist[index_i-print_per_batches+1:index_i+1,1]),
                np.mean(loss_hist[index_i-print_per_batches+1:index_i+1,2]),
                np.mean(loss_hist[index_i-print_per_batches+1:index_i+1,3]),
                np.mean(loss_hist[index_i-print_per_batches+1:index_i+1,4])*100))
               
    predict_label = test_whole(Ensemble, whole_loader, print_per_batches=10)                  
   
    predict_test = predict_label[test_array]
    OA,Kappa,producerA = CalAccuracy(predict_test,Y)
    print('Result:\n OA=%.2f,Kappa=%.2f' %(OA*100,Kappa*100))
    print('producerA:',producerA*100)  
    print('AA=%.2f' %(np.mean(producerA)*100))  
    img = DrawResult(predict_label+1,args.dataID)
    plt.imsave(save_path_prefix+'RSEN_'+'OA_'+repr(int(OA*10000))+'.png',img)   
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()   
    parser.add_argument('--dataID', type=int, default=1)
    parser.add_argument('--save_path_prefix', type=str, default='./')
    
    # train
    parser.add_argument('--labeled_batch_size', type=int, default=128)
    parser.add_argument('--unlabeled_batch_size', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=4096)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--print_per_batches', type=int, default=10)
    parser.add_argument('--num_label', type=int, default=30)
    parser.add_argument('--num_unlabel', type=int, default=10000)
    
    # network
    parser.add_argument('--teacher_alpha', type=float, default=0.95)
    parser.add_argument('--dropout', type=float, default=0.9)
    parser.add_argument('--noise', type=float, default=0.5)
    parser.add_argument('--m', type=int, default=5, help='number of stochastic augmentations')

    main(parser.parse_args())
