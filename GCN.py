
# torch
import torch
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool, global_add_pool, global_mean_pool
from kaolin.models.PointNet2 import PointNet2Classifier as Pointnet2


# Data
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from torch_geometric.data import Batch, Data, Dataset, DataLoader

# Analysis, plotting
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

# Util
import os.path as osp
import h5py
import pickle
import time
import os
import numpy as np
import copy

#-----------------------------------------------------------------------------------------#
#filepaths
h5_filepath = "/home/mattStubbs/watchmal/NeutronGNN/data/h5_files/iwcd_mpmt_shorttank_neutrongnn_6files.h5"
train_indices_file = "/home/mattStubbs/watchmal/NeutronGNN/data/splits/train_indicies_6files.txt"
val_indices_file = "/home/mattStubbs/watchmal/NeutronGNN/data/splits/validation_indicies_6files.txt"
test_indices_file = "/home/mattStubbs/watchmal/NeutronGNN/data/splits/test_indicies_6files.txt"

#-----------------------------------------------------------------------------------------#
#config

class CONFIG:
    pass
config=CONFIG()
config.data_path = h5_filepath
config.train_indices_file = train_indices_file
config.val_indices_file = val_indices_file
config.test_indices_file = test_indices_file

config.batch_size = 64
config.lr = 0.01
config.device = 'cuda'
config.gpu_list = [0]
config.num_data_workers = 0
config.epochs = 5
config.model_name = "GCN"

config.dump_path = "/home/mattStubbs/watchmal/NeutronGNN/data/dump/Pointnet/6/"
#-----------------------------------------------------------------------------------------#
#WCH5 Dataset

class WCH5Dataset(Dataset):

    def __init__(self, h5_path, train_indices_path, test_indicies_path, val_indicies_path, shuffle=True, transform=None, nodes=15808,
                 pre_transform=None, pre_filter=None, use_node_attr=False, use_edge_attr=False, cleaned=False ):

        super(WCH5Dataset, self).__init__("", transform, pre_transform,
                                        pre_filter)
        
        self.f = h5py.File(h5_path,'r')
        h5_event_data = self.f["event_data"]
        h5_labels = self.f["labels"]
        h5_nhits = self.f["nhits"]

        assert h5_event_data.shape[0] == h5_labels.shape[0] == h5_nhits.shape[0]
        
        #set up the memory map
        event_data_shape = h5_event_data.shape
        event_data_offset = h5_event_data.id.get_offset()
        event_data_dtype = h5_event_data.dtype         
        
        self.event_data = np.memmap(h5_filepath, mode='r', 
                            shape = event_data_shape,
                            offset=event_data_offset, 
                            dtype=event_data_dtype)
    
        #.........................................................................#
        
        self.labels = np.array(h5_labels)
        self.nhits = np.array(h5_nhits)
        self.transform=transform
        self.nodes = nodes
        
        #.........................................................................#
        
        self.X = self.event_data
        self.y = self.labels
        
        #.........................................................................#
        
        self.train_indices = self.load_indicies(train_indices_file)
        self.val_indices = self.load_indicies(val_indices_file)
        self.test_indices = self.load_indicies(test_indices_file)
        
        #.........................................................................#
                
        
    def load_indicies(self, indicies_file):
        with open(indicies_file, 'r') as f:
            lines = f.readlines()
        indicies = [int(l.strip()) for l in lines]
        return indicies
    
    def load_edges(self, nhits):
        edge_index = torch.ones([nhits, nhits], dtype=torch.int64)
        self.edge_index=edge_index.to_sparse()._indices()
    
    def get(self, idx):
        #nhits = self.nhits[idx]   #dynamic graph
        nhits = 300                #static graph
        x = torch.from_numpy(self.event_data[idx, :nhits, :])
        y = torch.tensor([self.labels[idx]], dtype=torch.int64)
        self.load_edges(nhits)
        return Data(x=x, y=y, edge_index=self.edge_index)
              
    def __len__(self):
        return len(self.X)

#-----------------------------------------------------------------------------------------#
#Dataloaders

def get_loaders(path, train_indices_file, val_indices_file, test_indices_file, batch_size, workers):
    
    dataset = WCH5Dataset(path, train_indices_file, val_indices_file, test_indices_file)
                          
    train_loader=DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                            pin_memory=True, sampler=SubsetRandomSampler(dataset.train_indices))

    val_loader=DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                            pin_memory=True, sampler=SubsetRandomSampler(dataset.val_indices))

    test_loader=DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                            pin_memory=True, sampler=SubsetRandomSampler(dataset.test_indices))
    
    return train_loader, val_loader, test_loader

out = get_loaders(config.data_path, config.train_indices_file, config.val_indices_file, 
                  config.test_indices_file, config.batch_size, config.num_data_workers)

trainDL, valDL, testDL = out

#-----------------------------------------------------------------------------------------#
#Network

class Net(torch.nn.Module):
    def __init__(self, w1=16, w2=64, w3=64, w4 = 10):
        super(Net, self).__init__()
        self.conv1 = GCNConv(8, w1, cached=False)
        self.conv2 = GCNConv(w1, w2, cached=False)
        self.conv3 = GCNConv(w2, w3, cached=False)
        self.conv4 = GCNConv(w3, w4, cached=False)
        self.linear = Linear(w4, 2)

    def forward(self, batch):
        x, edge_index, batch_index = batch.x, batch.edge_index, batch.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = self.conv4(x, edge_index)
        x = global_max_pool(x, batch_index)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)
    
# model = Net()
model = Pointnet2()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)

#-----------------------------------------------------------------------------------------#
# Logging Data

class CSVData:

    def __init__(self,fout):
        self.name  = fout
        self._fout = None
        self._str  = None
        self._dict = {}

    def record(self, keys, vals):
        for i, key in enumerate(keys):
            self._dict[key] = vals[i]

    def write(self):
        if self._str is None:
            self._fout=open(self.name,'w')
            self._str=''
            for i,key in enumerate(self._dict.keys()):
                if i:
                    self._fout.write(',')
                    self._str += ','
                self._fout.write(key)
                self._str+='{:f}'
            self._fout.write('\n')
            self._str+='\n'

        self._fout.write(self._str.format(*(self._dict.values())))
        self.flush()

    def flush(self):
        if self._fout: self._fout.flush()

    def close(self):
        if self._str is not None:
            self._fout.close()
            
#-----------------------------------------------------------------------------------------#
#Training and Validation

def Train(model, iteration_display, log_number, val_number):

    train_log = CSVData(config.dump_path + "log_train.csv")
    val_log = CSVData(config.dump_path + "log_val.csv")
    best_val_log = CSVData(config.dump_path + "log_best_val.csv")
    keys = ['epoch', 'iteration', 'accuracy', 'loss']
    best_model_wts = copy.deepcopy(model.state_dict())

    epoch = 0.; iteration = 0; correct = 0.; acc = 0.; total = 0.; best_acc = 0.;

    dataloaders = {
        "train": trainDL,
        "validation": valDL
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(config.device)

    for epoch in range(config.epochs):
        # monitor training loss
        train_loss = 0.0
    
        print('\nEpoch {}/{}'.format(epoch, config.epochs - 1))
        print('-' * 10)

        for data in trainDL:
            model.train()

            iteration += 1
            epoch += 1 / len(trainDL)

            #send data to GPU
            data = data.to(config.device)

            #zero parameter gradients
            optimizer.zero_grad()

            #display epoch and iteration at intervals
            if iteration % iteration_display == 0:
                print('epoch: {:.3f}, iteration: {:.2f}'.format(epoch, iteration))

            ##.............................................................##
            output = model.forward(data)      # forward pass
            loss = criterion(output, data.y)  # calculate the loss
            loss.backward()                   # propagate loss backwards in network
            optimizer.step()                  # update the gradient weights

            #evaluation parameters
            correct += output.argmax(1).eq(data.y).sum().item()
            total += data.y.shape[0]
            acc = correct / total             # running accuracy, loss in training stage

            #recording training phase statistics
            if iteration % log_number == 0 or iteration == 1 or iteration == len(trainDL)*config.epochs:
                train_log.record(keys, [epoch, iteration, acc, loss])
                train_log.write()

            #validation stage
            if iteration % val_number == 0 or iteration == 1 or iteration == len(trainDL)*config.epochs:
                #^ validation interval adjustable settings
                model.eval()
                correctV = 0.; accV = 0.; totalV = 0.;    #reset statistics 
                with torch.no_grad():
                    for dataV in valDL:                       #iterate over validation dataset
                        dataV = dataV.to(config.device)       #send to device
                        outputV = model.forward(dataV)        #forward pass
                        lossV = criterion(outputV, dataV.y)   #compute loss

                    #evaluation parameters
                        correctV += outputV.argmax(1).eq(dataV.y).sum().item()
                        totalV += dataV.y.shape[0]

                accV = correctV / totalV        #validation accuracy
                
                if accV > best_acc:
                    best_acc = accV
                    print('{}accV: {:.4f}, bestAcc: {:.4f}{}'.format('-'*10, accV, best_acc, '-'*10))
                    best_val_log.record(keys, [epoch, iteration, accV, lossV])
                    best_val_log.write()
                    #save best model parameters
                    torch.save(model.state_dict(), config.dump_path + 'state_dict')

                val_log.record(keys, [epoch, iteration, accV, lossV])
                val_log.write()

        #print epoch statistics
        print('Epoch: {:.4f} Loss: {:.4f}, accuracy: {:.4f}'.format(epoch, loss, acc))
    
    return model

                              
#----------------------------------------------------------------------------------------#        
#training

model = Train(model, iteration_display=200, log_number=100, val_number=5000)

                              
#-----------------------------------------------------------------------------------------#
#Evaluation on Test set

model = Net()
model.load_state_dict(torch.load(config.dump_path + 'state_dict'))
model.to(config.device)
model.eval() # prep model for *evaluation*
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

epoch=0.
iteration=0

with torch.no_grad():           

    correctT = 0.; totalT = 0.; accT = 0.; lossT = 0.;
    y_pred = []; y_true = []; y_pred_np = []
    y_pred_unrounded = np.zeros([1,2])
    
    #for iteration, data in enumerate(data_iter)
    for dataT in testDL:
        
        dataT = dataT.to(config.device)
        
        iteration += 1
        
        if iteration % 200 == 0:
            print("Iteration: {:.3f}, Progress {:.2f}\n".format(iteration, iteration/len(testDL)))
        
        res = model.forward(dataT)
        correctT += res.argmax(1).eq(dataT.y).sum().item()
        totalT += dataT.y.shape[0]
        lossT = criterion(res, dataT.y)
                       
        y_np = dataT.y.cpu().numpy()
        y_pred_unrounded = np.append(y_pred_unrounded, res.cpu().numpy(), axis = 0)
        y_pred_np = res.argmax(1).cpu().numpy()
        y_pred = np.append(y_pred, y_pred_np)
        y_true = np.append(y_true, y_np)

    print('total: ' + str(totalT))
    print('Number correct: ' + str(correctT))
    print('accuracy= ' + str(correctT/totalT * 100))
    accuracyT = [correctT/totalT * 100]
    print('y_pred: ' + str(y_pred))
    print('y_actual: ' + str(y_true))
    cm = confusion_matrix(y_true, y_pred)
    np.savetxt(config.dump_path + 'testAccuracy.csv', accuracyT, delimiter = ',')
    print(cm)

#----------------------------------------------------------------------------------#
                              #Save Testing Data 
#----------------------------------------------------------------------------------#
    
keys = ['y_true', 'nPred', 'ePred']
y_pred_unrounded_del = np.delete(y_pred_unrounded, 0, 0)
neutron_preds = y_pred_unrounded_del[:, 0] #log softmax probabilities
electron_preds = y_pred_unrounded_del[:, 1] #log softmax probabilities
normalizedN = (neutron_preds-min(neutron_preds))/(max(neutron_preds)-min(neutron_preds))
normalizedE = (electron_preds-min(electron_preds))/(max(electron_preds)-min(electron_preds))

np.savetxt(config.dump_path + 'nPred.csv', normalizedN, delimiter = ',')
np.savetxt(config.dump_path + 'ePred.csv', normalizedE, delimiter = ',')
np.savetxt(config.dump_path + 'y_true.csv', y_true, delimiter = ',')

#-----------------------------------------------------------------------------------------#
                              #Visualizations 
#-----------------------------------------------------------------------------------------#

import plotting_tools as ptool
class_labels = ('neutron', 'electron')

ptool.plot_confusion_matrix(cm=cm, classes=class_labels, location=config.dump_path, title="GCN fully connected")

fig = ptool.disp_learn_hist_smoothed(location=config.dump_path, window_train=500, window_val=10, title="GCN fully connected", show=False)

ptool.ROC(location=config.dump_path, title="GCN fully connected")
