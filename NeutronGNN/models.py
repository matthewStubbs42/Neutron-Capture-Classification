import torch
from torch_geometric.nn import GCNConv, AGNNConv, GINConv, SGConv,  GATConv
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool, global_add_pool, global_mean_pool

# <......................................................................................> #
# Graph Convolutional Networks: 
# T. N. Kipf and M. Welling, in Proc. of ICLR, 2017.
#..........................................................................................#

class GCN(torch.nn.Module):
    def __init__(self, w1=24, w2=8):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(8, w1, cached=False)
        self.conv2 = GCNConv(w1, w2, cached=False)
        self.linear = Linear(w2, 2)

    def forward(self, batch):
            x, edge_index, batch_index = batch.x, batch.edge_index, batch.batch
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = global_max_pool(x, batch_index)
            x = self.linear(x)
            return F.log_softmax(x, dim=1)
        

# <......................................................................................> #
# Attention-based Graph Neural Network for Semi-supervised Learning
# Thekumparampil, K. K., Wang, C., Oh, S., & Li, L. J. (2018). arXiv:1803.03735.
#..........................................................................................#

class AGNN(torch.nn.Module):
    def __init__(self, w1=16, w2=64, w3=64, w4 = 10):
        super(AGNN, self).__init__()
        self.lin1 = torch.nn.Linear(8, 16)
        self.prop1 = AGNNConv(requires_grad=False)
        self.prop2 = AGNNConv(requires_grad=True)
        self.lin2 = torch.nn.Linear(16, 2)

    def forward(self, batch):
        x, edge_index, batch_index = batch.x, batch.edge_index, batch.batch
        x = F.relu(self.lin1(x))
        x = self.prop1(x, edge_index)
        x = self.prop2(x, edge_index)
        x = global_max_pool(x, batch_index)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

    
# <......................................................................................> #
# Simplifying Graph Convolutional Networks
# Wu, F., Zhang, T., Souza Jr, A. H. D., Fifty, C., Yu, T., & Weinberger, K. Q. (2019).
# arXiv:1902.07153.
#...................................SGConv................................................#



class SG(torch.nn.Module):
    def __init__(self, w1=16, w2=64, w3=64, w4 = 10):
        super(SG, self).__init__()
        self.conv1 = SGConv(8, w1, cached=False)
        self.conv2 = SGConv(w1, w2, cached=False)
        self.conv3 = SGConv(w2, w3, cached=False)
        self.conv4 = SGConv(w3, w4, cached=False)
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

# <......................................................................................> #
# Graph Attention Networks
# Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2017). arXiv preprint arXiv:1710.10903.
#...................................GATConv................................................#

class GAT(torch.nn.Module):
    def __init__(self, num_features=8, w1=16, w2=16, num_heads=5):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, w1, heads=num_heads)
        self.conv2 = GATConv(w1*num_heads, w2, concat=False)
        self.lin2 = torch.nn.Linear(16, 2)

    def forward(self, batch):
        x, edge_index, batch_index = batch.x, batch.edge_index, batch.batch
#         x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_max_pool(x, batch_index)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)