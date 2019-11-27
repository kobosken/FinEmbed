import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
#from torch.utils.data.sampler import SubsetRandomSampler

#import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


torch.empty(3, dtype=torch.long).random_(5)
torch.randn(3, 5, requires_grad=True)

onehot = 'Onehot_US' #Onehot_Global
output = 'Output_US' #Output_Global // Output = range[0,C-1]
Set = 'US_Set' #Global_Set

target = pd.read_csv(r'C:/Users/fjaeckle/Documents/FIM/Research_Toronto/Data/'+output+'.csv',sep =',')
feature = pd.read_csv(r'C:/Users/fjaeckle/Documents/FIM/Research_Toronto/Data/NN_Input_'+Set+'.csv',sep =',')
feature_names = pd.read_csv(r'C:/Users/fjaeckle/Documents/FIM/Research_Toronto/Data/NN_Input_names_'+Set+'.csv',sep =',')
Sharpe_US = pd.read_csv(r'C:/Users/fjaeckle/Documents/FIM/Research_Toronto/Data/Sharpe_US.csv',sep = ',')


#remove first column which occurs through import of csv-file
target = target.drop('Unnamed: 0',axis = 1)
feature = feature.drop('Unnamed: 0',axis = 1)
feature_names = feature_names.drop('Unnamed: 0',axis = 1)
Sharpe_US = Sharpe_US.drop('Unnamed: 0',axis = 1)


#transforming dataframe into numpy
target = target.to_numpy()
feature = feature.to_numpy()
feature_names = feature_names['0'].tolist()
Sharpe_US = Sharpe_US.to_numpy()

#Defining Data for Test Set and Sharpe Ratios for Test Set
test_len = 40
target_test = target[range(len(target)-test_len,len(target))]
feature_test = feature[range(len(feature)-test_len,len(feature))]
Sharpe_US_test = Sharpe_US[range(len(Sharpe_US)-test_len,len(Sharpe_US))]
 

#Defining Data for Cross Validation
target_cv = np.delete(target,range(len(target)-40-2,len(target)),axis=0)
feature_cv = np.delete(feature,range(len(feature)-40-2,len(feature)),axis=0)



class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H1,H2,D_out):
        
        super(TwoLayerNet, self).__init__() #Class TwoLayerNet is sublacss of Class torch.nn.Module
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2,D_out)
        #self.linear4 = torch.nn.Linear(H3,D_out)
        
            
    def forward(self, x):

        linear1 = self.linear1(x)
        h1 = F.relu(linear1) #Activation function
        linear2 = self.linear2(h1)
        h2 = F.relu(linear2)
        y_pred = self.linear3(h2)
        #h3 = F.relu(linear3)
        #linear4 = self.linear4(h3)
        #y_pred =  F.softmax(linear3,dim = 1) #dim 1 = on row; dim0 = on column
        return y_pred

#Definiton of Model
D_in, H1,H2, D_out = feature.shape[1], 500,100,3

model = TwoLayerNet(D_in, H1,H2, D_out)
#model.apply(weights_init)

lr=0.0005
weight = torch.tensor([0.213,0.525,0.262])
weight = weight.float()
criterion = torch.nn.CrossEntropyLoss(weight=weight,reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=lr) #torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

k = 5 #KFolds
obs_k = int(len(feature_cv)/k) #-2 obs. for the first fold, -4 obs. for the next folds
epoch_size = 200
batch_size_tr = 64
batch_size_val = 35

train_loss_list = []
valid_loss_list = []
for epoch in range(epoch_size): ## run the model for 10 epochs
    
    #Cross Validation
    train_loss, valid_loss = [], []
    z = 0
    for i in range(k):
        if i ==0:
            #Dividing Features into Training and Validation Set
            feature_val = feature_cv[range(0+z,obs_k+z)]
            feature_train = np.delete(feature_cv,range(0+z,obs_k+z+2),axis=0)
            #Dividing Targets into Training and Validation Set
            target_val = target_cv[range(0+z,obs_k+z)]
            target_train = np.delete(target_cv,range(0+z,obs_k+z+2),axis=0) 
            #Numpy in Torch
            target_val,target_train = torch.from_numpy(target_val),torch.from_numpy(target_train)
            feature_val,feature_train = torch.from_numpy(feature_val),torch.from_numpy(feature_train)
            #Torch float64 int Torch float32
            target_val,target_train = target_val.long(),target_train.long()
            target_val,target_train = target_val.view(len(target_val)),target_train.view(len(target_train))
            #Reshape of Tensor
            feature_val,feature_train = feature_val.float(),feature_train.float()
        else:
            #Dividing Features into Training and Validation Set
            feature_val = feature_cv[range(0+z,obs_k+z)]
            feature_train = np.delete(feature_cv,range(0+z-2,obs_k+z+2),axis=0)
            #Dividing Targets into Training and Validation Set
            target_val = target_cv[range(0+z,obs_k+z)]
            target_train = np.delete(target_cv,range(0+z-2,obs_k+z+2),axis=0) 
            #Numpy in Torch
            target_val,target_train = torch.from_numpy(target_val),torch.from_numpy(target_train)
            feature_val,feature_train = torch.from_numpy(feature_val),torch.from_numpy(feature_train)
            #Torch float64 int Torch float32
            target_val,target_train = target_val.long(),target_train.long()
            target_val,target_train = target_val.view(len(target_val)),target_train.view(len(target_train))
            feature_val,feature_train = feature_val.float(),feature_train.float()
        
        
        #Creating TensorDataSet for both Training and Validation
        data_train = torch.utils.data.TensorDataset(feature_train,target_train)
        data_val = torch.utils.data.TensorDataset(feature_val,target_val)
        
        #Applying Dataloader if batch picking is desired   
        trainloader = DataLoader(data_train, batch_size=batch_size_tr,drop_last = True)
        validloader = DataLoader(data_val, batch_size=batch_size_val,drop_last = True) 
        
        z = obs_k + z
        ## training part 
        model.train()
        for x, y in trainloader:
            optimizer.zero_grad()
            
            y_pred = model.forward(x)  #y_pred = model(x)
            
            # Compute and print loss
            loss = criterion(y_pred, y)
            loss.backward() #determines the gradients of the loss function
            optimizer.step() #Weights update after each batch
            train_loss.append(loss.item())
            
        model.eval()
        for x, y in validloader:
            y_pred_val = model(x)
            loss = criterion(y_pred_val, y)
            valid_loss.append(loss.item())
    
    
    train_loss_list.append(np.mean(train_loss))
    valid_loss_list.append(np.mean(valid_loss))
    
############################Plot of Testing Error and Validation Error#################
fig, ax = plt.subplots(1,1,figsize = (10, 7))
ax.set_title('Testing Error'+'  '+'Validation Error')
ax.set_xlabel('Number of Epochs')
ax.set_ylabel('Error')
x = range(epoch_size)
plt.plot(x,train_loss_list,c = 'b',label = 'Test Error')
plt.plot(x, valid_loss_list,c = 'm',label = 'Valid Error')
plt.title(str(feature_names)+' '+'lr:'+str(lr)+' '+'kfold:'+str(k)+' '+'layer:'
          +str(D_in)+','+str(H1)+','+str(H2)+','+str(D_out))
ax.legend()

#############################Testing of Goodness######################################
# Applying ROC-Curve on multiclass classification
##Prediciton of test-set
#Transformation of numpy array into tensor 
 
feature_test_tensor = torch.from_numpy(feature_test) 
feature_test_tensor = feature_test_tensor.float()
#
y_pre = model(feature_test_tensor)
y_pre = y_pre.detach().numpy() #Convert tensor into numpy
y_pre = np.argmax(y_pre,axis = 1) #Converting in one hot vector
#
np.savetxt(r'C:/Users/fjaeckle/Documents/FIM/Research_Toronto/Data/test_pred_'+Set+'.csv',
           y_pre,delimiter=',')

np.savetxt(r'C:/Users/fjaeckle/Documents/FIM/Research_Toronto/Data/Sharpe_'+Set+'_test.csv',
           Sharpe_US_test,delimiter=',')

