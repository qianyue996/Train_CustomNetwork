from torch.utils.data.dataloader import DataLoader
from torch import optim
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm

from dataset import Mini_ImageNet
from network import Darknet19

class Trainer():
    def __init__(self):
        super().__init__()
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.lr=5e-4
        self.batch_size=2
        self.start_epoch=0
        self.epochs=300

        self.losses=[]
        self.loss_count=0
        self.accuracy_count=0
        self.checkpoint=None

    def setup(self):
        # 加载数据集
        train_ds=Mini_ImageNet()
        val_ds=Mini_ImageNet(mode='val')
        # ---
        self.train_dataloader=DataLoader(train_ds,batch_size=self.batch_size,shuffle=True)
        self.val_dataloader=DataLoader(val_ds,batch_size=self.batch_size,shuffle=True)
        # 模型初始化
        self.model=Darknet19().to(self.device)
        self.optimizer=optim.Adam([param for param in self.model.parameters() if param.requires_grad],lr=self.lr)

        # 尝试从上次训练结束点开始
        try:
            self.checkpoint=torch.load('yolov2.weight')
        except Exception as e:
            pass
        if self.checkpoint:
            self.model.load_state_dict(self.checkpoint['model'])
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            self.start_epoch = self.checkpoint['epoch'] + 1

        # tensorboard
        self.writer=SummaryWriter(f'runs/{time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())}')

    def train(self):
        self.model.train()
        for epoch in range(self.start_epoch,self.epochs):
            epoch_all_loss=0
            with tqdm(self.train_dataloader, disable=False) as bar:
                for item in bar:
                    batch_x,batch_y=item
                    batch_x,batch_y=batch_x.to(self.device),batch_y.to(self.device)
                    batch_output=self.model(batch_x)

                    loss=self.compute_loss(batch_output,batch_y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.loss_count+=1
                    epoch_all_loss+=loss.item()
                    bar.set_postfix({'epoch':epoch,
                                     'loss':epoch_all_loss/self.loss_count})
            epoch_avg_loss=epoch_all_loss/self.loss_count
            tqdm.write(f"本轮epoch平均损失为: {epoch_avg_loss}")
            self.writer.add_scalar('epoch_loss',epoch_avg_loss,epoch)
            self.losses.append(epoch_avg_loss)
            self.save_best_model(epoch)
            # val
            self.val(epoch)

    def val(self,epoch):
        self.model.eval()
        with torch.no_grad():
            epoch_all_acc=0
            bar=tqdm(self.val_dataloader, disable=False)
            for item in bar:
                batch_x,batch_y=item
                batch_x,batch_y=batch_x.to(self.device),batch_y.to(self.device)
                batch_output=self.model(batch_x)

                accuracy=self.compute_accuracy(batch_output,batch_y)
                self.accuracy_count+=1
                epoch_all_acc+=accuracy
                bar.set_postfix({'batch_accuracy':f'{epoch_all_acc/self.accuracy_count:.2%}'})
            epoch_avg_acc=epoch_all_acc/self.accuracy_count
            self.writer.add_scalar('epoch_accuracy',epoch_avg_acc,epoch)

    def compute_loss(self,batch_output,batch_y):
        loss=torch.nn.CrossEntropyLoss(reduction='mean')(batch_output,batch_y)
        self.writer.add_scalar('batch_loss(each batch size)',loss,self.loss_count)
        return loss

    def compute_accuracy(self,batch_output,batch_y):
        pred=torch.argmax(batch_output,dim=1)
        targ=torch.argmax(batch_y,dim=1)
        accuracy=(pred==targ).float().mean()
        self.writer.add_scalar('accuracy(each batch size)',accuracy,self.accuracy_count)
        return accuracy

    def save_best_model(self,epoch):
        if len(self.losses)==1 or self.losses[-1]<self.losses[-2]: # 保存更优的model
            checkpoint={
                'model':self.model.state_dict(),
                'optimizer':self.optimizer.state_dict(),
                'epoch':epoch
            }
            torch.save(checkpoint,'.yolov2.weight')
            os.replace('.yolov2.weight','yolov2.weight')

if __name__ == '__main__':
    trainer=Trainer()
    trainer.setup()
    trainer.train()