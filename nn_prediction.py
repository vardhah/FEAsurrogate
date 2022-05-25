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


input_size=4                             # input size may change if integer/ordinal type variable and represented by one-hot encoding
output_size=1                            # number of output 
ranges=[thickness_l,thickness_h,radius_l,radius_h,length_l,length_h,cp_l,cp_h]                # ranges in form of [low1,high1,low2,high2,...]

mask=['real','real','real','real']                     # datatype ['dtype1','dtype2']
categories=[[None],[None],[None],[None]]  



first_run=1
device='cuda'

nnstorage=glob.glob("./models/*.pt")
print('nns are:',nnstorage) 

test_data= np.loadtxt("./data/dataware/test_data.txt", delimiter=" ",skiprows=0, dtype=np.float32)
result_file_name= './data/prediction_result.csv'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"  # specify which GPU(s) to be used

def estimate_accuracy(test_data,prediction):
    print('test_data:',test_data.shape,'prediction:',prediction.shape)
    copied_test_data=np.copy(test_data)
    ground_truth=label_data(copied_test_data,prediction)
    
    index_f = np.where(ground_truth[:,-1]==1)
    index_p = np.where(ground_truth[:,-1]==0)
      
    failed_gt= ground_truth[index_f[0]]
    passed_gt=ground_truth[index_p[0]]

    print('outlier is:',failed_gt.shape[0])
    accuracy=( passed_gt.shape[0]/(passed_gt.shape[0]+failed_gt.shape[0]))
    print('Accuracy is:', accuracy) 
    return accuracy


def calc_residual(truth,pred):
    delz= np.subtract(truth,pred)
    residual=np.divide(np.abs(delz),np.abs(truth))
    sum_residual= np.sum(residual)/truth.shape[0]
    print('residual is:',residual.shape,'sum residual is:',sum_residual)

    
def calc_deviation(truth,pred):
    delz= np.subtract(truth,pred)
    var= np.sum(np.square(delz))/truth.shape[0]
    print('variance is:',var)
    std= np.power(var,0.5)
    print('std is:',std)
    return std

def run(): 
    first_run=1
    copied_test_data=np.copy(test_data)
    fitted_test_data= data_preperation(copied_test_data,mask,np.array(ranges),categories)
        
    testing_data = SimDataset(fitted_test_data)
    fitted_text_X= fitted_test_data[:,:-1]; fitted_test_y=fitted_test_data[:,-1]
    print('fitted X:',fitted_text_X,'fitted test Y:',fitted_test_y)
        
    for nn in nnstorage:
      path=nn
      print('Model is:',path)
      neuralNet= SNet(input_size,output_size)
        
      try: 
        neuralNet.load_state_dict(torch.load(path))       
        print("Loaded earlier trained model successfully")
      except: 
        print('Couldnot find weights of NN')  
           
      with torch.no_grad(): 
            output = neuralNet(torch.from_numpy(fitted_text_X).float())
              
      output=output.cpu().detach().numpy()
      estimate_accuracy(test_data,output)
      if first_run==1:
           print('in 1st runid')
           multi_runresults= np.array(output).reshape(-1,1)
           first_run=0
      else: 
           multi_runresults= np.concatenate((multi_runresults,np.array(output).reshape(-1,1)),axis=1)
      np.savetxt(result_file_name,multi_runresults,  delimiter=',')  


def load_and_average(): 
     predicted_data= np.loadtxt(result_file_name, delimiter=",",skiprows=0, dtype=np.float32)
     print('shape of predicted matrix:',predicted_data.shape) 
     mean_prediction= np.average(predicted_data, axis=1)
     print('test prediction:',mean_prediction)
     estimate_accuracy(test_data,mean_prediction)
     return mean_prediction     

if __name__ == "__main__":
        run()  
        pred=load_and_average()
        test= test_data[:,-1]
        print('pred shape:',pred.shape,'test shape is:', test.shape)
        calc_residual(test,pred)
        calc_deviation(test,pred)
        """
       
        ###Load evaluation data ,derived ground_truth and created storage for result  
        test_data= np.loadtxt("./data/dataware/test_data.txt", delimiter=" ",skiprows=0, dtype=np.float32)
        
        copied_test_data=np.copy(test_data)
        fitted_test_data= data_preperation(copied_test_data,mask,np.array(ranges),categories)
        
        testing_data = SimDataset(fitted_test_data)
        fitted_text_X= fitted_test_data[:,:-1]; fitted_test_y=fitted_test_data[:,-1]
        print('fitted X:',fitted_text_X,'fitted test Y:',fitted_test_y)
        
        for nn in nnstorage:
         path=nn
         neuralNet= SNet(input_size,output_size)
        
         try: 
           neuralNet.load_state_dict(torch.load(path))       
           print("Loaded earlier trained model successfully")
         except: 
           print('Couldnot find weights of NN')  
         
           
         with torch.no_grad(): 
            output = neuralNet(torch.from_numpy(fitted_text_X).float())
              

         copied_test_data=np.copy(test_data)
         output=output.cpu().detach().numpy()
         ground_truth=label_data(copied_test_data,output)
    
         index_f = np.where(ground_truth[:,-1]==1)
         index_p = np.where(ground_truth[:,-1]==0)
      
         failed_gt= ground_truth[index_f[0]]
         passed_gt=ground_truth[index_p[0]]

         result=( passed_gt.shape[0]/(passed_gt.shape[0]+failed_gt.shape[0]))
         print('Accuracy is:', result) 
         if first_run==1:
           print('in 1st runid')
           multi_runresults= np.array(output).reshape(-1,1)
           first_run=0
         else: 
           multi_runresults= np.concatenate((multi_runresults,np.array(output).reshape(-1,1)),axis=1)
         print('result is:',multi_runresults)
         np.savetxt(result_file_name,multi_runresults,  delimiter=',')
         """
      
	



