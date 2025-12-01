import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import ImplicitGraph
from torch.nn import Parameter
from utils import get_spectral_rad, SparseDropout
import torch.sparse as sparse
from torch_geometric.nn.norm import PairNorm

class IGNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, num_node, dropout, kappa=0.9, adj_orig=None):
        super(IGNN, self).__init__()

        self.adj = None
        self.adj_rho = None
        self.adj_orig = adj_orig

        #one layer with V
        self.ig1 = ImplicitGraph(nfeat, nhid, num_node, kappa)
        self.ig2 = ImplicitGraph(nhid, nhid, num_node, kappa)
        self.dropout = dropout
        self.X_0 = None
        self.V_0 = nn.Linear(nfeat, nhid)
        self.V_1 = nn.Linear(nhid, nhid)
        #self.map=nn.Linear(nhid,nhid//4)#8
        #self.norm = PairNorm(scale=10)
    def forward(self, features, adj):
        if adj is not self.adj:
            self.adj = adj
            self.adj_rho = get_spectral_rad(adj)

        x = features
        x = F.elu(self.ig1(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig).T + self.V_0(x.T)).T

        # import pdb; pdb.set_trace()

        x = self.ig2(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig).T + self.V_1(x.T)
        #x=self.norm(x)
        x1 = F.dropout(x, self.dropout, training=self.training)
        #x = self.V(x1)
        #x2 = self.map(x1)
        return x1
class final_predictor(nn.Module):
    def __init__(self, nfeat, head, nclass):
        super(final_predictor, self).__init__()
        self.nfeat = nfeat
        self.head = head
        self.nclass = nclass
        self.attn=nn.MultiheadAttention(self.nfeat,self.head)
        self.fc = nn.Linear(self.nfeat, self.nclass)
    def forward(self,node,edge):
        x,_=self.attn(node,edge,edge,need_weights=False)
        x=self.fc(x)
        return x
class IGNN_v2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, num_node, num_edge, typed, dropout, kappa=0.9, adj_orig=None):
        super(IGNN_v2, self).__init__()

        self.adj = None
        self.adj_rho = None
        self.adj_orig = adj_orig
        self.typed=typed
        #one layer with V
        self.ig1 = ImplicitGraph(nfeat, nhid, num_node, kappa)
        self.dropout = dropout
        self.X_0 = Parameter(torch.zeros(nhid, num_node), requires_grad=False)
        self.V = nn.Linear(nhid, nclass, bias=False)
        if self.typed == 'edge':
            self.scaler = nn.Linear(num_node,num_edge)
        if self.typed == 'node':
            self.scaler = nn.Linear(num_edge,num_node)
        #self.map=nn.Linear(nhid,nhid//4)#8
        #self.norm = PairNorm(scale=10)
    def forward(self, features, adj,last=False):
        #if self.type == 'edge':
        adj=self.scaler(adj).to_sparse_coo()
        import pdb;pdb.set_trace()
        if adj is not self.adj:
            self.adj = adj
            self.adj_rho = get_spectral_rad(adj)

        x = features
        x = self.ig1(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig).T
        #x=self.norm(x)
        x = F.dropout(x, self.dropout, training=self.training)
        if last==True:
            x = self.V(x)
        #x = self.V(x)
        #x2 = self.map(x1)
        return x

class IGNN2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, num_node, dropout, kappa=0.9, adj_orig=None):
        super(IGNN2, self).__init__()

        self.adj = None
        self.adj_rho = None
        self.adj_orig = adj_orig

        #one layer with V
        self.ig1 = ImplicitGraph(nfeat, nhid, num_node, kappa)
        self.dropout = dropout
        self.X_0 = Parameter(torch.zeros(nhid, num_node), requires_grad=False)
        self.V = nn.Linear(nhid, nclass, bias=False)

    def forward(self, features, adj):
        if adj is not self.adj:
            self.adj = adj
            self.adj_rho = get_spectral_rad(adj)

        x = features
        x = self.ig1(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig).T
        x = F.dropout(x, self.dropout, training=self.training)
        #x = self.V(x1)
        return x
