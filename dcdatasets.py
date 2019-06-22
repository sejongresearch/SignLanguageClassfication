import torch
import torch.utils.data as data
import json
import os
import os.path
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torch.nn.functional as F

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from collections import namedtuple
import pandas as pd
DB_ROOT = './'



class imageloader(data.Dataset):
   

    def __init__(self,videoindex, co_transform=None, condition=None):

        
        
        self.co_transform = co_transform    
        self.mode = condition    
        
        txttotal=open('./txtpath.txt')
        self.txtpath=[]
        for line in txttotal:
            self.txtpath.append(line[:-1])
        csvtotal=open('./csvpath.txt')
        self.csvpath=[]
        for line in csvtotal:
            self.csvpath.append(line[:-1])
        import pdb
        #pdb.set_trace()
        self.nowcsvpath=self.csvpath[int(videoindex)]
        self.nowtxtpath=self.txtpath[int(videoindex)]
        
        
        self._imgpath_lwir = os.path.join('%s','%s','%s','%s','%s') 
        self.handlist =pd.read_csv(self.nowcsvpath) 
        self.ids = list() 
        self.count=0
        
        for line in open(self.nowtxtpath):
            if(len(line)==1):
                continue
            self.ids.append((DB_ROOT, line.strip().split('/')))
        
    def __str__(self):
        return self.__class__.__name__ + '_' + self.image_set

    def __getitem__(self, index): 

       
        lwir,Limg,Rimg,label,flag = self.pull_item(index)
        
        lwir=lwir.resize((128, 128),Image.BILINEAR) #128 128 3
        Limg=Limg.resize((128, 128),Image.BILINEAR)
        Rimg=Rimg.resize((128, 128),Image.BILINEAR)
      
        
        lwir= lwir - np.array([[[   np.array(lwir)[:,:,0].mean(),
                                    np.array(lwir)[:,:,1].mean(),
                                    np.array(lwir)[:,:,2].mean()]]])

        Limg= Limg - np.array([[[   np.array(Limg)[:,:,0].mean(),
                                    np.array(Limg)[:,:,1].mean(),
                                    np.array(Limg)[:,:,2].mean()]]])
        
        Rimg= Rimg - np.array([[[   np.array(Rimg)[:,:,0].mean(),
                                    np.array(Rimg)[:,:,1].mean(),
                                    np.array(Rimg)[:,:,2].mean()]]])
        
        if self.co_transform is not None:
            lwir= self.co_transform(lwir)
            Limg= self.co_transform(Limg)
            Rimg= self.co_transform(Rimg)

        
        
        
        return lwir.float(),Limg.float(),Rimg.float(),label,flag

    def pull_item(self, index):
        import pdb
        #pdb.set_trace()
        self.count+=1
        frame_id = self.ids[index]
        #print(frame_id)
        lwir = Image.open(self._imgpath_lwir%(frame_id[1][0],frame_id[1][1],frame_id[1][2],frame_id[1][3],frame_id[1][4])).convert('RGB')
        image = Image.new("RGB",(500,500),(0,0,0))
        Lx=self.handlist.loc[index][0]
        Ly=self.handlist.loc[index][1]
        Rx=self.handlist.loc[index][2]
        Ry=self.handlist.loc[index][3]
        
        if Lx==0 and Ly==0:
             Limg=Image.new("RGB",(128,128),(0,0,0))
            
        else:
             Limg=lwir.crop((Lx-150,Ly-150,300,300))
      
        if(Rx==0and Ry==0):
             Rimg=Image.new("RGB",(128,128),(0,0,0))
        else:
             Rimg=lwir.crop((Rx-150,Ry-150,300,300))
        
        width, height = lwir.size
        
        label=torch.zeros(20)
        labelnum=frame_id[1][2].split(';')
        label[int(labelnum[0])]=1
        
        flag=0
        if index !=len(self.ids)-1:
            ntframe_id = self.ids[index+1]
            ntvideo=ntframe_id[1][3]
        
            if(ntvideo != frame_id[1][3]):
                   flag=1
        else:
            flag=1
      
        return lwir ,Limg,Rimg,label,flag #, boxes_t, labels


    def __len__(self):
        return len(self.ids)
    
    def loadimage(self,cnt):
      
        index=cnt*16
        
        # vis = list()
        lwir = list()
        import pdb
        
        label=list()
        Limg=list()
        Rimg=list()
        flag=list()
        import pdb
        #pdb.set_trace()
        for i in range(16):
            image,Limage,Rimage,L_b,fl=self.__getitem__(index+i)
            lwir.append(image)
            Limg.append(Limage)
            Rimg.append(Rimage)
            label.append(L_b)
            flag.append(fl)
        lwir = torch.stack(lwir, dim=0)
        label= torch.stack(label, dim=0)
        Limg= torch.stack(Limg, dim=0)
        Rimg= torch.stack(Rimg, dim=0)
            
        # return vis, boxes, labels, index # tensor (N, 3, 300, 300), 3 lists of N tensors each    
        return  lwir ,Limg,Rimg,label,flag

if __name__ == '__main__':

    #from datasets import *
    from torchcv.datasets.transforms import *
    from torchcv.datasets.functional import *
    import torch.utils.data
    transforms2 = Compose([ToTensor()])
   
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
  
    for index in trainindex:
        countimage=0
        train_dataset = imageloader(index,co_transform=transforms2,condition='train')
        
        while True:
            lwir ,Limg,Rimg,label,flag=train_dataset.loadimage(countimage)
            import pdb
            # pdb.set_trace()
            print(label)
            countimage+=1;
            if flag[-1]==1:
                
                ######################
                break
