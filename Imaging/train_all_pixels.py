import numpy as np
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.optim import Adam
import pandas as pd
from scipy.ndimage import zoom
from math import pi

print("Using torch", torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

norm_factor=65536
y_anchor=[70,150,230,310,390,470,550,630,710,790,870,950,1030]
x_anchor=[150,230,310,390,470,550,630,710,790,870,950,1030,1110]

class Simple_CNN(nn.Module):
    def __init__(self,p):
        super().__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(1,16,5,1,2),                              
            nn.ReLU(),            
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16,32,3,1,1),     
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                
        )
        self.conv3 = nn.Sequential(         
            nn.Conv2d(32,64,3,1,1),     
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                
        )
        self.mlp = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(5*5*64,1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,4),
        )
        
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = y.view(y.size(0), -1)       
        y = self.mlp(y)
        return y
    
def train(model,X,y,optimizer,loss_fn,batchsize=5001):
    idx=np.arange(X.shape[0])
    np.random.shuffle(idx)
    
    epoch_loss = 0
    model.train()
    for i in range(0,X.shape[0],batchsize):
        if i+batchsize>=X.shape[0]:
            cur_idx=idx[i:]
        else:
            cur_idx=idx[i:i+batchsize]
            
        optimizer.zero_grad()
        y_pred = model(X[cur_idx])
        loss = loss_fn(y_pred, y[cur_idx])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()*cur_idx.shape[0]
    return epoch_loss/X.shape[0]
    
def evaluate(model,X,y,loss_fn,batchsize=5001,calc_MAE=False):
    epoch_loss = 0
    separate_loss=np.zeros(4,dtype=np.float32)
    MAEs=np.zeros(4,dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(0,X.shape[0],batchsize):
            i_start=i
            if i+batchsize>=X.shape[0]:
                i_end=X.shape[0]-1
            else:
                i_end=i+batchsize
                
            y_pred=model(X[i_start:i_end])
            loss = loss_fn(y_pred, y[i_start:i_end])
            epoch_loss += loss.item()*(i_end-i_start)
            temp = torch.abs(y_pred-y[i_start:i_end])
            separate_loss += torch.sum(torch.square(temp),dim=0).to("cpu").numpy()
            MAEs += torch.sum(temp,dim=0).to("cpu").numpy()
            
        epoch_loss=epoch_loss/X.shape[0]
        separate_loss=separate_loss/X.shape[0]
        MAEs=MAEs/X.shape[0]
            
    return epoch_loss,separate_loss,MAEs


def run_cnn_exp(task_name,x_train,x_val,x_test,y_train,y_val,y_test,
                p=0.001,seed_list=[324,716,10086],n_epoch=1000,save_dir="./pixels/"):
    score=[]
    print("Running %s"%task_name)
    for seed in seed_list:
        torch.manual_seed(seed)
        model=Simple_CNN(p)
        model = model.to(device)
        loss_fn=MSELoss()
        optimizer=Adam(model.parameters(),lr=0.001)
        result=np.zeros((n_epoch//10,11),dtype=np.float32)
        
        first_stage=round(n_epoch*0.5)
        for i in range(first_stage):
            train_loss=train(model,x_train,y_train,optimizer,loss_fn)
            if i%10==0:
                j=i//10
                result[j,0]=train_loss
                result[j,1],_,_=evaluate(model,x_val,y_val,loss_fn)
                result[j,2],result[j,3:7],result[j,7:]=evaluate(model,x_test,y_test,loss_fn)
            #print("(%d/%d)"%(i,n_epoch),end="\r")
        
        torch.save(model,save_dir+task_name+"_seed%d"%seed+".pt")
        #optimizer.param_groups[0]['lr'] = 0.0005
        for i in range(first_stage,n_epoch):
            train_loss=train(model,x_train,y_train,optimizer,loss_fn)
            if i%10==0:
                j=i//10
                result[j,0]=train_loss
                result[j,1],_,_=evaluate(model,x_val,y_val,loss_fn)
                result[j,2],result[j,3:7],result[j,7:]=evaluate(model,x_test,y_test,loss_fn)
                if np.min(result[:j,1])>result[j,1]:
                    torch.save(model,save_dir+task_name+"_seed%d"%seed+".pt")
            #print("(%d/%d)"%(i,n_epoch),end="\r")
        
        best_idx=np.argmin(result[:,1])
        score.append(result[best_idx,2])
        result=pd.DataFrame(result,columns=["train_loss","validation_loss","test_loss","S0_loss","S1_loss","S2_loss","S3_loss",
                                           "S0_MAE","S1_MAE","S2_MAE","S3_MAE"])
        save_filename=save_dir+task_name+"_seed%d"%seed+".csv"
        result.to_csv(save_filename,index=False)
        #print("\n")
    return score
    

def stokes_from_alphabeta(alpha,beta):
    alpha=alpha/180*pi
    beta=beta/180*pi
    result=np.zeros(3,dtype=np.float32)
    result[0]=0.5*(np.cos(4*alpha-4*beta)+np.cos(4*alpha))
    result[1]=0.5*(np.sin(4*alpha)-np.sin(4*alpha-4*beta))
    result[2]=-np.sin(4*alpha-2*beta)
    return result

def get_dataset(i_y,i_x,pixel_size=80,seed=1212):
    np.random.seed(seed)
    ori_y=np.zeros((1200,3),dtype=np.float32)
    fo=np.load("./dataset/alphabeta.npz")
    cur_alpha=fo["arr_0"]
    cur_beta=fo["arr_1"]
    for i in range(1200):
        ori_y[i]=stokes_from_alphabeta(cur_alpha[i],cur_beta[i])
    ori_x=np.zeros((1200,40,40),dtype=np.float32)
    for i in range(12):
        temp=np.load("./dataset/data%d.npz"%i)["arr_0"][:,y_anchor[i_y]:y_anchor[i_y]+80,x_anchor[i_x]:x_anchor[i_x]+80]
        for j in range(100):
            ori_x[i*100+j]=zoom(temp[j],(40/temp[j].shape[0],40/temp[j].shape[1]))

    shuffled_i=np.arange(1200)
    np.random.shuffle(shuffled_i)
    ori_y=ori_y[shuffled_i]
    ori_x=ori_x[shuffled_i]

    x_train=np.zeros((1000*10,1,40,40),dtype=np.float32)
    y_train=np.ones((1000*10,4),dtype=np.float32)
    y_train[:1000,1:]=ori_y[:1000]
    x_train[:1000,0]=ori_x[:1000]
    for i in range(1,10):
        temp=np.random.rand(1000).astype(np.float32)
        for j in range(1000):
            x_train[i*1000+j,0]=ori_x[j]*temp[j]
        y_train[i*1000:(i+1)*1000,0]=temp
        y_train[i*1000:(i+1)*1000,1:]=ori_y[:1000]
    x_val=np.zeros((100*10,1,40,40),dtype=np.float32)
    y_val=np.ones((100*10,4),dtype=np.float32)
    y_val[:100,1:]=ori_y[1000:1100]
    x_val[:100,0]=ori_x[1000:1100]
    for i in range(1,10):
        temp=np.random.rand(100).astype(np.float32)
        for j in range(100):
            x_val[i*100+j,0]=ori_x[1000+j]*temp[j]
        y_val[i*100:(i+1)*100,0]=temp
        y_val[i*100:(i+1)*100,1:]=ori_y[1000:1100]
    x_test=np.zeros((100*10,1,40,40),dtype=np.float32)
    y_test=np.ones((100*10,4),dtype=np.float32)
    y_test[:100,1:]=ori_y[1100:]
    x_test[:100,0]=ori_x[1100:]
    for i in range(1,10):
        temp=np.random.rand(100).astype(np.float32)
        for j in range(100):
            x_test[i*100+j,0]=ori_x[1100+j]*temp[j]
        y_test[i*100:(i+1)*100,0]=temp
        y_test[i*100:(i+1)*100,1:]=ori_y[1100:]
        
    np.random.seed(seed)
    aug_rand=np.random.rand(120)*1.5+1
    dark_img=np.load("./img/dark.npz")["arr_0"][y_anchor[i_y]:y_anchor[i_y]+80,x_anchor[i_x]:x_anchor[i_x]+80]
    x_aug=np.zeros((120,1,40,40),dtype=np.float32)
    for i in range(120):
        x_aug[i,0]=(zoom(dark_img,(40/80,40/80))*aug_rand[i]).astype(np.float32)
    x_train=np.concatenate((x_train,x_aug[:100]))
    x_val=np.concatenate((x_val,x_aug[100:110]))
    x_test=np.concatenate((x_test,x_aug[110:]))
    y_train=np.concatenate((y_train,np.zeros((100,4),dtype=np.float32)))
    y_val=np.concatenate((y_val,np.zeros((10,4),dtype=np.float32)))
    y_test=np.concatenate((y_test,np.zeros((10,4),dtype=np.float32)))
        
    x_train=torch.from_numpy((x_train/norm_factor).astype(np.float32)).to(device)
    x_val=torch.from_numpy((x_val/norm_factor).astype(np.float32)).to(device)
    x_test=torch.from_numpy((x_test/norm_factor).astype(np.float32)).to(device)
    y_train=torch.from_numpy(y_train).to(device)
    y_val=torch.from_numpy(y_val).to(device)
    y_test=torch.from_numpy(y_test).to(device)
    
    return x_train,x_val,x_test,y_train,y_val,y_test

def train_pixel(i_y,i_x,pixel_size=80,seed=1212):
    x_train,x_val,x_test,y_train,y_val,y_test=get_dataset(i_y,i_x,pixel_size,seed=seed)
    run_cnn_exp("y%d_x%d"%(i_y,i_x),x_train,x_val,x_test,y_train,y_val,y_test,
                seed_list=[seed])
    
for i in range(len(y_anchor)):
    for j in range(len(x_anchor)):
        train_pixel(i,j)
print("All finished...")
