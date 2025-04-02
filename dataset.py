from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import torch
import pandas as pd
from tqdm import tqdm
import cv2 as cv
import sys
import os

class Mini_ImageNet(Dataset):
    def __init__(self, mode='train'):
        super(Mini_ImageNet,self).__init__()
        # ------------------- Basic parameters ------------------- #
        self.datasets = []
        # ------------------- Basic parameters ------------------- #
        if mode=='train':
            df=pd.read_csv(filepath_or_buffer='data/mini_imagenet100/train.csv')
        elif mode=='val':
            df=pd.read_csv(filepath_or_buffer='data/mini_imagenet100/val.csv')
        else:
            print("""
"请正确初始化数据集
训练集（默认）: mode='train'
验证集       : mode='val'
""")
            sys.exit(1)
        # -------------------- #
        classesdict = dict()
        with open('data/mini_imagenet100/classname.txt','r',encoding='utf-8')as f:
            for index,line in enumerate(f.readlines()):
                _list=[]
                for name in line[1:].split(', '):
                    _list.append(name.strip('\n'))
                classesdict[index]=_list
        self.id2name={i:','.join(names) for i,names in classesdict.items()}
        self.name2id={c:i for i,name in classesdict.items() for c in name}
        # -------------------- #
        img_path=df['image:FILE']
        img_classid=df['category']
        for i in tqdm(range(len(img_path)), desc='数据集处理中'):
            x_path=os.path.join('data/mini_imagenet100',img_path[i])
            x_id=img_classid[i]
            self.datasets.append((x_path,x_id))
    
    def __getitem__(self,index):
        img_path,id=self.datasets[index]
        img=cv.imread(img_path)
        img=cv.resize(img,(224,224))
        x=ToTensor()(img)

        y=torch.zeros(100)
        y[id]=1
        return x,y
    
    def __len__(self):
        return len(self.datasets)