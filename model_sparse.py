import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
#import torch.nn.parameter as Parameter
import math
from matplotlib import pyplot as plt
import pdb
from torch_geometric.utils import dense_to_sparse, f1_score
from gcn import GCNConv
from torch_scatter import scatter_add
import torch_sparse
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import remove_self_loops, add_self_loops
import utils

class GTN(nn.Module):
    
    def __init__(self, config_param, max_seq_length, item_probs, device, d_type, norm):
        super(GTN, self).__init__()
        self.num_edge = config_param['num_edge']
        self.num_channels = config_param['num_channels']
        self.rnn_units = config_param['rnn_units']
        self.rnn_layers = config_param['rnn_layers']
        self.w_in = config_param['w_in']
        self.w_out = config_param['w_out']
        self.num_class = config_param['num_class']
        self.num_layers = config_param['num_layers']
        self.max_seq_length = max_seq_length
        self.item_probs = item_probs
        self.device = device
        self.dtype = d_type
        self.is_norm = norm
        self.nb_items = len(item_probs)
        layers = []
        for i in range(self.num_layers):
            if i == 0:
                layers.append(GTLayer(self.num_edge, self.num_channels, first=True))
            else:
                layers.append(GTLayer(self.num_edge, self.num_channels, first=False))
        self.layers = nn.ModuleList(layers)
        self.gcn = GCNConv(in_channels=self.w_in, out_channels=self.w_out)
        self.lstm = nn.LSTM(self.w_out, self.rnn_units, self.rnn_layers, bias=True, batch_first=True)
        self.linear1 = nn.Linear(self.w_out * self.num_channels, self.w_out)
        self.h2item_score = nn.Linear(in_features=self.rnn_units, out_features=self.nb_items, bias=False)

    def normalization(self, H):
        norm_H = []
        for i in range(self.num_channels):
            edge, value=H[i]
            edge, value = remove_self_loops(edge, value)
            deg_row, deg_col = self.norm(edge.detach(), self.num_nodes, value.detach())
            value = deg_col * value
            norm_H.append((edge, value))
        return norm_H

    def norm(self, edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        with torch.no_grad(): 
            if edge_weight is None:
                edge_weight = torch.ones((edge_index.size(1), ),
                                        dtype=dtype,
                                        device=edge_index.device)
            edge_weight = edge_weight.view(-1)
            assert edge_weight.size(0) == edge_index.size(1)
            row, col = edge_index
            deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
            deg_inv_sqrt = deg.pow(-1)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row], deg_inv_sqrt[col]

    def forward(self, A, X, seq_len, seqs, hidden):
        batch_size = seqs.shape[0]
        Ws = []
        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A)
            else:
                H = self.normalization(H)
                H, W = self.layers[i](A, H)
            Ws.append(W)
        for i in range(self.num_channels):
            if i==0:
                edge_index, edge_weight = H[i][0], H[i][1]
                X_ = self.gcn(X,edge_index=edge_index.detach(), edge_weight=edge_weight)
                X_ = F.relu(X_)
            else:
                edge_index, edge_weight = H[i][0], H[i][1]
                X_ = torch.cat((X_,F.relu(self.gcn(X,edge_index=edge_index.detach(), edge_weight=edge_weight))), dim=1)

        X_ = self.linear1(X_)

        basket_seqs = torch.zeros(batch_size, self.max_seq_length, self.w_out)
        for seq_id, seq in enumerate(seqs, 0):
            for basket_id, basket in enumerate(seq, 0):
                items_id = torch.nonzero(basket).squeeze()
                if items_id.size()[0] > 1:
                    basket_seqs[seq_id, basket_id] = utils.max_pooling(X_[items_id])
                else:
                    if items_id.size()[0] == 1:
                        basket_seqs[seq_id, basket_id] = X_[items_id]

        lstm_out, (h_n, c_n) = self.lstm(basket_seqs, hidden)
        actual_index = torch.arange(0, batch_size) * self.max_seq_length + (seq_len - 1)
        actual_lstm_out = lstm_out.reshape(-1, self.rnn_units)[actual_index]

        hidden_to_score = self.h2item_score(actual_lstm_out)
        # print(hidden_to_score)

        # predict next items score
        next_item_probs = torch.sigmoid(hidden_to_score)

        # loss = self.loss(next_item_probs, target_basket)
        # return loss, target_basket, Ws
        return next_item_probs

class GTLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_nodes, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        self.num_nodes = num_nodes
        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels, num_nodes)
            self.conv2 = GTConv(in_channels, out_channels, num_nodes)
        else:
            self.conv1 = GTConv(in_channels, out_channels, num_nodes)
    
    def forward(self, A, H_=None):
        if self.first == True:
            result_A = self.conv1(A)
            result_B = self.conv2(A)                
            W = [(F.softmax(self.conv1.weight, dim=1)).detach(),(F.softmax(self.conv2.weight, dim=1)).detach()]
        else:
            result_A = H_
            result_B = self.conv1(A)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach()]
        H = []
        for i in range(len(result_A)):
            a_edge, a_value = result_A[i]
            b_edge, b_value = result_B[i]
            
            edges, values = torch_sparse.spspmm(a_edge, a_value, b_edge, b_value, self.num_nodes, self.num_nodes, self.num_nodes)
            H.append((edges, values))
        return H, W

class GTConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_nodes):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels))
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.num_nodes = num_nodes
        self.reset_parameters()
    def reset_parameters(self):
        n = self.in_channels
        nn.init.constant_(self.weight, 1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        filter = F.softmax(self.weight, dim=1)
        num_channels = filter.shape[0]
        results = []
        for i in range(num_channels):
            for j, (edge_index,edge_value) in enumerate(A):
                if j == 0:
                    total_edge_index = edge_index
                    total_edge_value = edge_value*filter[i][j]
                else:
                    total_edge_index = torch.cat((total_edge_index, edge_index), dim=1)
                    total_edge_value = torch.cat((total_edge_value, edge_value*filter[i][j]))
            index, value = torch_sparse.coalesce(total_edge_index.detach(), total_edge_value, m=self.num_nodes, n=self.num_nodes)
            results.append((index, value))
        return results
