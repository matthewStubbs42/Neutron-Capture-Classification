from torch_geometric.data import Batch, Data, Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from torch_cluster import knn_graph, radius_graph
import h5py
import numpy as np
import torch
from scipy import sparse
from scipy.spatial import distance_matrix
from torch_geometric.utils import convert
import sklearn.preprocessing as sklp

        
##......................................................................................................##
#                                      Data Loading Settings                                             #
##......................................................................................................##
        
def get_loaders(path, train_indices_path, test_indices_path, val_indices_path, batch_size, workers,
                       k_neighbours, fully_connected=True, dynamic=False, distance_weighted=True):
    
    dataset = WCH5Dataset(path, train_indices_path, test_indices_path, val_indices_path,
                               fully_connected, dynamic, distance_weighted, k_neighbours,)
    
    train_loader=DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                            pin_memory=True, sampler=SubsetRandomSampler(dataset.train_indices))
    val_loader=DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                            pin_memory=True, sampler=SubsetRandomSampler(dataset.val_indices))
    test_loader=DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                            pin_memory=True, sampler=SubsetRandomSampler(dataset.test_indices))
    
    return train_loader, val_loader, test_loader


##......................................................................................................##
#                                       WCH5 Dataset Class                                               #
##......................................................................................................##

class WCH5Dataset(Dataset):

    def __init__(self, h5_path, train_indices_path, test_indices_path, val_indices_path, 
                        fully_connected, dynamic, distance_weighted, k_neighbours):

        super(WCH5Dataset, self).__init__()
        
        print('fully connected: {}'.format(fully_connected))
        print('dynamic: {}'.format(dynamic))
        print('distance weighted: {}'.format(distance_weighted))
        self.f = h5py.File(h5_path,'r')
        h5_event_data = self.f["event_data"]
        h5_labels = self.f["labels"]
        h5_nhits = self.f["nhits"]

        assert h5_event_data.shape[0] == h5_labels.shape[0] == h5_nhits.shape[0]
        
        #set up the memory map
        event_data_shape = h5_event_data.shape
        event_data_offset = h5_event_data.id.get_offset()
        event_data_dtype = h5_event_data.dtype         
        
        self.event_data = np.memmap(h5_path, mode='r', shape = event_data_shape,
                        offset=event_data_offset,  dtype=event_data_dtype)
    
        #.........................................................................#
        
        self.labels = np.array(h5_labels)
        self.nhits = np.array(h5_nhits)
        self.dynamic = dynamic
        self.fully_connected = fully_connected
        self.k_neighbours = k_neighbours
        self.distance_weighted = distance_weighted
        
        #.........................................................................#
        
        self.X = self.event_data
        self.y = self.labels
        
        #.........................................................................#
        
        self.train_indices = self.load_indicies(train_indices_path)
        self.val_indices = self.load_indicies(val_indices_path)
        self.test_indices = self.load_indicies(test_indices_path)
        
        #.........................................................................#
                
        
    def load_indicies(self, indicies_file):
        with open(indicies_file, 'r') as f:
            lines = f.readlines()
        indicies = [int(l.strip()) for l in lines]
        return indicies
    
    def dist_pos_matrix(self, idx, nhits):
        hitPosMatrix = np.array(self.X[idx, :nhits, 2:5])
        hitDistMatrix = distance_matrix(hitPosMatrix, hitPosMatrix)
        norm_hitDistMatrix = sklp.normalize(hitDistMatrix)
        norm_sparse_hitDistMatrix = sparse.csr_matrix(norm_hitDistMatrix)
        hitEdges = convert.from_scipy_sparse_matrix(norm_sparse_hitDistMatrix)
        return hitEdges
    
    #.........................................................................#
    # load edges
    
    def load_edges(self, idx, nhits):
        if self.distance_weighted:
            distEdgeTensor = self.dist_pos_matrix(idx, nhits)
            self.edge_index = distEdgeTensor[0]
            self.edge_attr = distEdgeTensor[1]
            
        else:
            if self.fully_connected:
                edge_index = torch.ones([nhits, nhits], dtype=torch.int64)
                self.edge_index=edge_index.to_sparse()._indices()
            else:
                pos = torch.as_tensor(self.event_data[idx, :nhits, 2:5], dtype=torch.float)
                self.edge_index = knn_graph(pos, k=self.k_neighbours)
    
    #.........................................................................#
    # get graph at index
    
    def get(self, idx):
        if self.dynamic:
            nhits = self.nhits[idx]   #dynamic graph
        else:
            nhits = 250                #static graph
        x = torch.from_numpy(self.event_data[idx, :nhits, :])
        y = torch.tensor([self.labels[idx]], dtype=torch.int64)
#         print(nhits)
        self.load_edges(idx, nhits)
        
        if self.distance_weighted:
            return Data(x=x, y=y, edge_index=self.edge_index, edge_attr = self.edge_attr)
        else:
            return Data(x=x, y=y, edge_index=self.edge_index)
        
    #.........................................................................#
    # graph length
    def __len__(self):
        return len(self.X)
    
