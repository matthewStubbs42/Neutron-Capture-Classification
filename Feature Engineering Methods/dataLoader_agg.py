from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import sklearn.preprocessing as sklp
from sklearn import preprocessing
import pandas as pd
        

##......................................................................................................##
#                               MLP Aggregate Features Dataloader Class                                  #
##......................................................................................................##

# define data loaders for MLP/XGBoost
def get_loaders_features(path, train_indices_file, test_indices_file, val_indices_file, batch_size, workers):
    
    dataset = featureDataset(path, train_indices_file, test_indices_file, val_indices_file)
                          
    train_loader=DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                            pin_memory=True, sampler=SubsetRandomSampler(dataset.train_indices))

    val_loader=DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                            pin_memory=True, sampler=SubsetRandomSampler(dataset.val_indices))

    test_loader=DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                            pin_memory=True, sampler=SubsetRandomSampler(dataset.test_indices))

    return train_loader, val_loader, test_loader


##......................................................................................................##

class featureDataset(Dataset):

    def __init__(self, h5_path, train_indices_path, test_indices_path, val_indices_path, shuffle=False, transform=None):
        
        feature_dict = load_features(h5_path)
        self.transform=transform
        
        self.dset = pd.DataFrame(feature_dict)
        self.X = self.dset.values[:, 0:12]
        self.y = self.dset.values[:, 12]

        self.train_indices = self.load_indicies(train_indices_path)
        self.val_indices = self.load_indicies(val_indices_path)
        self.test_indices = self.load_indicies(test_indices_path)
              
    ##...............................................................##
    def load_indicies(self, indicies_file):
        with open(indicies_file, 'r') as f:
            lines = f.readlines()
        indicies = [int(l.strip()) for l in lines]
        return indicies
    
    def __getitem__(self, index):
        
        if self.transform is None:
            return [self.X[index], self.y[index]] 
        else:
            return self.transform([self.X[index], self.y[index]])
        
    def __len__(self):
        return len(self.X)
      
#     def __del__(self):
#         self.f.close()
        
##............................................................................##

def load_features(h5_filepath):
    with h5py.File(h5_filepath, 'r') as file:
        h5_event_data = file['event_data']
        h5_nhits = file['nhits']
        h5_labels = file['labels']
        labels = np.array(h5_labels)
        event_mem_data = np.memmap(h5_filepath, mode='r', shape=h5_event_data.shape,
                                   offset=h5_event_data.id.get_offset(),
                                   dtype=h5_event_data.dtype)
        nhits = np.array(h5_nhits)
        h5_charges = np.array(event_mem_data[:, :, 0])
        q_sums = h5_charges.sum(axis = 1)
    
    betas = np.genfromtxt('/home/mattStubbs/watchmal/NeutronGNN/Kore/Isotropy/betasALL/ne1ALL/betas_sorted_all.csv', delimiter=',')
    wallDist = np.genfromtxt("/home/mattStubbs/watchmal/NeutronGNN/Kore/feature_engineering/distance_to_wall/wallDist.csv", delimiter=',')
    tRMS = np.genfromtxt("/home/mattStubbs/watchmal/NeutronGNN/Kore/feature_engineering/flight_time/tRMS.csv", delimiter=',')
    tMean = np.genfromtxt("/home/mattStubbs/watchmal/NeutronGNN/Kore/feature_engineering/flight_time/meanTime.csv", delimiter=',')
    moa = np.genfromtxt("/home/mattStubbs/watchmal/NeutronGNN/Kore/feature_engineering/MOA/moa.csv", delimiter=',')
    angRMS = np.genfromtxt("/home/mattStubbs/watchmal/NeutronGNN/Kore/feature_engineering/Consec_Angles/angleRMSConsec.csv", delimiter=',')
    
    betas1, betas2, betas3, betas4, betas5 = betas[:, 1], betas[:, 2], betas[:, 3], betas[:, 4], betas[:, 5] 
    
    features = [nhits, q_sums, betas1, betas2, betas3, betas4, betas5, tRMS, wallDist, moa, tMean, angRMS]  
    features = [feature.reshape(-1, 1) for feature in features]
    min_max_scaler = preprocessing.MinMaxScaler()
    features = [min_max_scaler.fit_transform(feature) for feature in features]
    features = [feature.flatten() for feature in features]
    
    feature_Dict = { "nhits": nhits, "charges": q_sums, "dW":wallDist, "tRMS": tRMS, "tMean":tMean, "moa":moa, "angRMS": angRMS, "b1": betas1, "b2": betas2, "b3": betas3, "b4": betas4, "b5": betas5,  "labels": labels}
    
    return feature_Dict
