import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.utils.data import dataloader
import dgl
import dgl.function as fn
import dgl.nn.pytorch as dgl_nn


import os
from os import listdir
from os.path import isfile, join
import networkx as nx
import numpy as np
import pickle




class NNConvLayer(nn.Module):
    """MPNN Conv Layer
    Paramters
    ---------
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 edge_net,
                 root_weight=True,
                 bias=True):
        super(NNConvLayer, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_net = edge_net

        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        if self.root is not None:
            nn.init.xavier_normal_(self.root.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.zero_()
        for m in self.edge_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1.414)

    def message(self, edges):
        return {'m': torch.matmul(edges.src['h'].unsqueeze(1), edges.data['w']).squeeze(1)}
            
    def apply_node_func(self, nodes):
        aggr_out = nodes.data['aggr_out']
        if self.root is not None:
            aggr_out = torch.mm(nodes.data['h'], self.root) + aggr_out

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return {'h': aggr_out}

    def forward(self, g, h, e):
        h = h.unsqueeze(-1) if h.dim() == 1 else h
        e = e.unsqueeze(-1) if e.dim() == 1 else e

        g.ndata['h'] = h
        g.edata['w'] = self.edge_net(e).view(-1, self.in_channels, self.out_channels)
        g.update_all(self.message, fn.sum("m", "aggr_out"), self.apply_node_func)
        return g.ndata.pop('h')


class MPNNModel(nn.Module):
    """MPNN model"""

    def __init__(self,
                 node_input_dim=22,
                 edge_input_dim=5,
                 output_dim=5,
                 node_hidden_dim=8, # 64
                 edge_hidden_dim=32, #128
                 num_step_message_passing=6,
                 num_step_set2set=6,
                 num_layer_set2set=3,
                 device=torch.device('cuda')
                 ):
        """model parameters setting
        Paramters
        ---------
        """
        super(MPNNModel, self).__init__()
        self.name = "MPNN"
        self.num_step_message_passing = num_step_message_passing
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        self.device = device
        edge_network = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim)
        )

        self.tanh=nn.Tanh()
        self.conv = NNConvLayer(in_channels=node_hidden_dim, out_channels=node_hidden_dim, edge_net=edge_network, root_weight=False)
        self.gru = nn.GRU(node_hidden_dim, node_hidden_dim)
        
        self.bn1=nn.BatchNorm1d(node_hidden_dim)
        self.bn2=nn.BatchNorm1d(output_dim+1)
        #self.bn2=nn.BatchNorm1d(1)
        self.bn3=nn.BatchNorm1d(5)
    

        self.set2set = dgl_nn.glob.Set2Set(node_hidden_dim, num_step_set2set, num_layer_set2set)
        self.lin1 = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        self.lin2 = nn.Linear(node_hidden_dim, output_dim)
        self.lin3 =nn.Linear(output_dim+1,5)
        self.lin4 = nn.Linear(5,1)

    def forward(self, g,basisnums):
        

        h = g.ndata['n_feat']
        h = h.to(self.device)
        out = F.relu(self.lin0(h))
        h = out.unsqueeze(0)

        for i in range(self.num_step_message_passing):
            m = F.relu(self.conv(g, out, g.edata['e_feat']))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
        
        # import pdb
        # pdb.set_trace()
        s2sout = self.set2set(g, out)
        out = self.bn1(self.tanh(self.lin1(s2sout)))
        out = self.lin2(out)
        #basisnum = basisnum.squeeze(1)
        #out = torch.cat((out,basisnum),1)
        out = torch.cat((out,basisnums),1)
        #out = torch.cat((out,basisnum),2)
        out = self.bn2(out)
        #out= self.bn3(self.tanh(self.lin3(out)))
        out= self.lin3(out)
        out=self.lin4(out).squeeze(-1)
        #print(out)
        return out