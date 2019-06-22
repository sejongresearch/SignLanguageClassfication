import os
import torch
import numpy as np
import random
import torch.optim
import torchvision
import torch.backends.cudnn as cudnn
import argparse
import time
import torch.utils.data
import logging
import logging.handlers
import torch.nn.init as init
import torchvision.transforms as transforms

from utils import *
#from torchcv.datasets.transforms import *
from sklearn.model_selection import train_test_split
from model import *
from torch.utils.data import DataLoader
from dcdatasets import *
from torch import nn
from datetime import datetime
from tensorboardX import SummaryWriter



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



learning_rate = 1e-3
frames = 16
checkpoint = None
start_epoch = 0
total_epoch = 100
epochs_when_stop = 0
best_loss = 100
port = 8897 #tensorboard
saveDir = "d" #directory name
parser = argparse.ArgumentParser(description='PyTorch Training')
args = parser.parse_args()

cudnn.benchmark = True



torch.backends.cudnn.deterministic=True


import subprocess, atexit



def run_tensorboard( jobs_dir, port=8811 ): # for tensorboard
    pid = subprocess.Popen( ['tensorboard', '--logdir', jobs_dir, '--host', '0.0.0.0', '--port', str(port)] )    
    
    def cleanup():
    	pid.kill()

    atexit.register( cleanup )


def main():

        global epochs_when_stop, start_epoch, best_loss, epoch, checkpoint


        ########################## WEIGHT INITIALIZE #########################################################
        model_C3D = C3D()
        ####
        model_C3D_pretrained_dict = torch.load('./C3D-ucf101_epoch-1249.pth.tar')
        model_C3D_pretrained_dict = model_C3D_pretrained_dict['state_dict']
        ####
        model_C3D_dict = model_C3D.state_dict()
        #import pdb
        #pdb.set_trace()
        model_C3D_pretrained_dict = {k: v for k, v in model_C3D_pretrained_dict.items() if k in model_C3D_dict}
        model_C3D_dict.update(model_C3D_pretrained_dict)
        model_C3D.load_state_dict(model_C3D_dict)
        model_C3D = model_C3D.to(device)
        for param in model_C3D.parameters():
                param.requires_grad = False
        ####################################################################################################


        ################################# USE CHECKPOINT ######################################################
        if checkpoint==None:

                model_LSTM = LSTM_anno()
                model_LSTM = model_LSTM.to(device)
        
                
        
                optimizer = torch.optim.Adam(model_LSTM.parameters(),lr=learning_rate)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10,gamma=0.1)


        else:
                checkpoint = torch.load(checkpoint)
                start_epoch = checkpoint['epoch'] + 1
                epochs_when_stop = checkpoint['epochs_when_stop']
                best_loss = checkpoint['best_loss']
                print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))
                model_LSTM = checkpoint['model']
                model_LSTM.to(device)
                optimizer = checkpoint['optimizer']
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10,gamma=0.1)

        #######################################################################################################


        criterion = nn.CrossEntropyLoss().to(device)

        

        score = 0


        ##################################### MAKE TRAIN AND TEST TXT  ########################################################
        trainindex=[]
        testindex=[]
        if os.path.exists("./trainindex.txt")and os.path.exists("./testindex.txt"):
                trainindexlist=open("./trainindex.txt")
                for line in trainindexlist:
                        trainindex.append(line[:-1])
                testindexlist=open("./testindex.txt")
                for line in testindexlist:
                        testindex.append(line[:-1])    
        else:
                nptotalindex=np.arange(800)
            
                trin,tein=train_test_split(nptotalindex)
            
            
                trainindexpath="./trainindex.txt"
                texttrain = open(trainindexpath,'wt',newline='\n')
            
                for i in trin:
                        trainindex.append(str(i))
                
                for i in tein:
                        testindex.append(str(i))
                texttrain.write('\n'.join(trainindex))
                texttrain.close()
            
                testindexpath="./testindex.txt"
                texttest = open(testindexpath,'wt',newline='\n')
                texttest.write('\n'.join(testindex))
                texttest.close()
        ##########################################################################################################################

       
        ########################################### Set job directory ############################################################

        

        jobs_dir = os.path.join('jobs',saveDir)

        snapshot_dir    = os.path.join( jobs_dir, 'snapshots' )
        tensorboard_dir    = os.path.join( jobs_dir, 'tensorboardX' )
        if not os.path.exists(snapshot_dir):        os.makedirs(snapshot_dir)
        if not os.path.exists(tensorboard_dir):     os.makedirs(tensorboard_dir)
        run_tensorboard( tensorboard_dir, port )

        ############################################################################################################################3


        writer = SummaryWriter(os.path.join(jobs_dir, 'tensorboardX'))
    
                               

        for epoch in range(start_epoch,total_epoch):
                """

                ################### TRAIN ############################
                
                train_loss = train_phase(trainindex=trainindex,
                                         model_C3D=model_C3D,
                                         model_LSTM=model_LSTM,
                                         criterion=criterion,
                                         optimizer=optimizer,
                                         epoch=epoch,
                                         writer=writer)

                is_best = train_loss < best_loss
                best_loss = min(train_loss,best_loss)

                if not is_best: # check loss convergence
                        epochs_when_stop+=1
                else:
                        epochs_when_stop=0

                
                ########################################################



                #write tensorboard
                writer.add_scalars('train/epoch',{'epoch_best_loss':best_loss},global_step=epoch)

                #save checkpoint
                save_checkpoint(epoch, epochs_when_stop, model_LSTM, optimizer, train_loss, best_loss, is_best, jobs_dir)
                filepath = os.path.join(jobs_dir,'checkpoint.pth.tar{:03d}'.format(epoch))
                filename = 'checkpoint.pth.tar{:03d}'.format(epoch)
                """
                
                ################ TEST #################################
                score,rscore = test_phase(testindex=testindex,
                                   model_C3D=model_C3D,
                                   model_LSTM=model_LSTM,
                                   criterion=criterion
                                   )
                print("score : " +str(score)+"_rscore : " +str(rscore))
                writer.add_scalars('test_score/epoch',{'score':score},global_step=epoch)
                writer.add_scalars('test_rscore/epoch',{'rscore':rscore},global_step=epoch)

                ########################################################
                
                # go down learning rate in each 10*n epoch
                scheduler.step()
                






def train_phase(trainindex, model_C3D,model_LSTM, criterion,optimizer,epoch,writer):
        
        idd=0
        model_C3D.eval()
        model_LSTM.train()
        


        losses = AverageMeter()
  

        transforms2 = transforms.Compose([transforms.ToTensor()])
        

        feats_all = torch.Tensor().to(device)
        
        
        for index in trainindex:
                countimage=0
                train_dataset = imageloader(index,co_transform=transforms2,condition='train')

                while True:     
                        idd+=16
                        images ,left,right,label,flag=train_dataset.loadimage(countimage)
                        
                               
                       
                       
                        images = images.view(1,3,16,128,128).to(device)     
                        labeled = torch.argmax(label[0]).view(1).to(device)

                   
                        feats = model_C3D(images)
                        feats_all = torch.cat((feats_all,feats),0)

                        countimage+=1;
                        if flag[-1]==1:
                        

                                scores = model_LSTM(feats_all.view(-1,1,4096))
                                

       
                                
                        
                                loss = criterion(scores,labeled)
      
                                

                                optimizer.zero_grad()

                                loss.backward()

                                optimizer.step()

                                losses.update(loss.item())
                                
                                
                                writer.add_scalars('train/loss', {'loss': losses.avg}, global_step=idd )
                                print('epoch:'+str(epoch)+'_batch:'+str(idd)+'_loss:'+str(loss.item())+' <-index: '+str(labeled.item()))
                                del images , left, right, label, flag, feats,feats_all, scores, labeled

                                feats_all = torch.Tensor().to(device)
                               
                                
                                break


        

        return losses.avg

def test_phase(testindex,model_C3D,model_LSTM,criterion):
        correct_cnt = 0
        all_cnt = 0
        bat_cnt = 0
        feats_all = torch.Tensor().to(device)
        
        
        transforms2 = transforms.Compose([transforms.ToTensor()])
        

        regscores = AverageMeter()
        regressionScore=torch.nn.Softmax()

        print('Please wait!')
        with torch.no_grad():
                model_C3D.eval()
                model_LSTM.eval()
               
                for index in testindex:
                        countimage=0
                        test_dataset = imageloader(index,co_transform=transforms2,condition='test')

                        while True:
                        
                                images ,left,right,label,flag=test_dataset.loadimage(countimage)

                                bat_cnt+=16
                                
                                images = images.view(1,3,16,128,128).to(device)
                                

                                feats = model_C3D(images)
                                feats_all = torch.cat((feats_all,feats),0)

                                countimage+=1;
                                if flag[-1]==1:
                                
                     
                                        
                                        scores = model_LSTM(feats_all.view(-1,1,4096))
                                
                               
                                
                                
                                        labeled = torch.argmax(label[0]).to(device)
                                        
                                        scoresed = torch.argmax(scores)

                                        
                                        regscore = regressionScore(scores)
                                        


                                        all_cnt+=1
                                        print('label:'+str(labeled))
                                        
                                        
                                        
                                        
                                        if (labeled == scoresed):
                                                print('regScore:_ '+str(regscore[0][labeled].item())+' Percent')
                                                regscores.update(regscore[0][labeled].item())
                                                correct_cnt+=1
                                        else:
                                                print('Wrong!')
                                                regscores.update(0)
                                                       
                                        print(regscores)
                                        del images , left, right, label, flag, feats,feats_all, scores, labeled
                                        feats_all = torch.Tensor().to(device)
                                       
                                        break
        
        return float(correct_cnt)/float(all_cnt)*100, (regscores.avg)*100
         
                



        
       
if __name__ == '__main__':
        
        
        main()
