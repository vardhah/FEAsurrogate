import os
import glob
import pathlib
import numpy as np
import torch
import torch.nn as nn
from student_model import SNet
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
from utils import *
#from trainer import model_trainer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from lp import load_N_predict
import shutil

######change for each run_id
#run=[1,2,3,4,5]
run=[1]
####################
rhog_safety=15096.9      # calculted as pho=1027, g=9.8, safety factor=1.5 ( from STR excel sheet)

thickness_l=0.01; thickness_h=0.5;

length_l=1.2; length_h=3.6;
depth_l=200; depth_h=6000;
#need to find it
radius_l=0.3; radius_h=1.2
#crush_pressure= rhog_safety*depth*0.000001
cp_l=rhog_safety*depth_l*0.000001
cp_h=rhog_safety*depth_h*0.000001
print('cp low is:',cp_l,'cp high is:',cp_h)

search_itr=5 
input_size=4                             # input size may change if integer/ordinal type variable and represented by one-hot encoding
num_variable = 4                        # number of variables  both real & int type 
output_size=1                            # number of output 
ranges=[thickness_l,thickness_h,radius_l,radius_h,length_l,length_h,cp_l,cp_h]                # ranges in form of [low1,high1,low2,high2,...]

mask=['real','real','real','real']                     # datatype ['dtype1','dtype2']
categories=[[None],[None],[None],[None]]  


max_epoch = 500
at_least_epoch=25
batch_size = 16
device='cuda'
loss_fn=nn.MSELoss()
num_co=[]
flag_first=0


if __name__ == "__main__":
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]="1"  # specify which GPU(s) to be used
        ###Load evaluation data ,derived ground_truth and created storage for result  
        traini_data= np.loadtxt("./data/dataware/train_data.txt", delimiter=" ",skiprows=0, dtype=np.float32)
        
        train_data,validation_data= data_split(traini_data,proportion=0.1)
        copied_train_data=np.copy(train_data)
        copied_validation_data=np.copy(validation_data)
        
        fitted_train_data= data_preperation(copied_train_data,mask,np.array(ranges),categories)
        fitted_validation_data= data_preperation(copied_validation_data,mask,np.array(ranges),categories)
        train_data = SimDataset(fitted_train_data)
        validate_data = SimDataset(fitted_validation_data)
        print('length of train data:',len(train_data),'validate data:',validate_data)
	 
        path='./models/nnl2.pt'
        name= 'nnl2.pt'
        
        neuralNet= SNet(input_size,output_size)
        model = neuralNet.to(device) 
        try: 
           neuralNet.load_state_dict(torch.load(path))       
           print("Loaded earlier trained model successfully")
        except: 
           print('Couldnot find weights of NN')  
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        epoch=0; loss_train=[];loss_validate=[]     
        while True: 
            print('training epoch:',epoch)   
            if epoch > max_epoch:
                break    
            try:
                dataloader = DataLoader(train_data, batch_size, True)
                correct = 0
                for x, y in dataloader:
                	y=y.view(-1,1)
                	x, y = x.to(device), y.to(device)
                	output = model(x)
                	loss = loss_fn(output, y)
                	optimizer.zero_grad()
                	loss.backward() 
                	optimizer.step()
                	correct+= loss.item()
                train_loss=correct/len(train_data); loss_train.append(train_loss)
                
                with torch.no_grad(): 
                  dataloader = DataLoader(validate_data, batch_size, True)
                  correct = 0
                  for x, y in dataloader:
                    y=y.view(-1,1)
                    x, y = x.to(device), y.to(device)
                    output = model(x)
                    loss = loss_fn(output, y)
                    correct += loss.item()
                validate_loss= correct/len(dataloader); loss_validate.append(validate_loss) 
                
                if epoch <= at_least_epoch:
                  whichmodel=epoch  
                  torch.save(model.state_dict(), path)
                #if epoch%20==0:
                   #print('epoch is:',epoch)
                if epoch> at_least_epoch:
                 diff_loss=np.absolute(train_loss-validate_loss)
                 if flag_first==0: 
                   torch.save(model.state_dict(), path)
                   whichmodel=epoch 
                   flag_first=1
                   last_diff_loss=diff_loss

                 elif flag_first==1:
                  if last_diff_loss>diff_loss:
                   torch.save(model.state_dict(), path); whichmodel=epoch ;
                   last_diff_loss=diff_loss

            except KeyboardInterrupt:
                break
           
            epoch+=1

        fig=plt.figure(figsize=(9,6))
        plt.plot(loss_train,label='training')
        plt.plot(loss_validate,label='validate')
        plt.legend()
        plt.show()
        print('--> Saved model is from', whichmodel , ' epoch')
        print('model is:',model) 

