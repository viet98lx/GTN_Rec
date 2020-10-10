import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from matplotlib import pyplot as plt
import pdb
import utils


class GTN(nn.Module):
    
    def __init__(self, config_param, max_seq_length, item_probs, device, d_type, num_nodes, norm):
        super(GTN, self).__init__()
        self.num_edge = config_param['num_edge']
        self.num_channels = config_param['num_channels']
        self.rnn_units = config_param['rnn_units']
        self.rnn_layers = config_param['rnn_layers']
        self.w_in = config_param['w_in']
        self.w_out = config_param['w_out']
        self.num_class = config_param['num_class']
        self.num_layers = config_param['num_layers']
        self.batch_size = config_param['batch_size']
        self.max_seq_length = max_seq_length
        self.item_probs = item_probs
        self.device = device
        self.dtype = d_type
        self.num_nodes = num_nodes
        self.is_norm = norm
        self.nb_items = len(item_probs)
        layers = []
        for i in range(self.num_layers):
            if i == 0:
                layers.append(GTLayer(self.num_edge, self.num_channels, first=True))
            else:
                layers.append(GTLayer(self.num_edge, self.num_channels, first=False))
        self.layers = nn.ModuleList(layers)
        self.weight = nn.Parameter(torch.Tensor(self.w_in, self.w_out))
        self.bias = nn.Parameter(torch.Tensor(self.w_out))
        self.lstm = nn.LSTM(self.w_out, self.rnn_units, self.rnn_layers, bias=True, batch_first=True)
        self.linear1 = nn.Linear(self.w_out*self.num_channels, self.w_out)
        self.h2item_score = nn.Linear(in_features=self.rnn_units, out_features=self.nb_items, bias=False)
        # self.linear2 = nn.Linear(self.w_out, self.num_class)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def gcn_conv(self,X,H):
        X = torch.mm(X, self.weight)
        H = self.norm(H, add=True)
        return torch.mm(H.t(),X)

    def normalization(self, H):
        for i in range(self.num_channels):
            if i==0:
                H_ = self.norm(H[i,:,:]).unsqueeze(0)
            else:
                H_ = torch.cat((H_,self.norm(H[i,:,:]).unsqueeze(0)), dim=0)
        return H_

    def norm(self, H, add=False):
        H = H.t()
        if add == False:
            H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor).to(self.device))
        else:
            H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor).to(self.device)) + torch.eye(H.shape[0]).type(torch.FloatTensor).to(self.device)
        deg = torch.sum(H, dim=1)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        deg_inv = deg_inv*torch.eye(H.shape[0]).type(torch.FloatTensor)
        H = torch.mm(deg_inv,H)
        H = H.t()
        return H

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        weight = next(self.parameters()).data
        return (weight.new(self.rnn_layers, batch_size, self.rnn_units).zero_(),
                weight.new(self.rnn_layers, batch_size, self.rnn_units).zero_())

    def forward(self, A, X, seq_len, seqs, hidden):
        batch_size = seqs.shape[0]
        # Learn new structure graph by combine adjacency matrices
        A = A.unsqueeze(0).permute(0,3,1,2)
        Ws = []
        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A)
            else:
                H = self.normalization(H)
                H, W = self.layers[i](A, H)
            Ws.append(W)
        
        #H,W1 = self.layer1(A)
        #H = self.normalization(H)
        #H,W2 = self.layer2(A, H)
        #H = self.normalization(H)
        #H,W3 = self.layer3(A, H)

        #GCN on graph
        for i in range(self.num_channels):
            if i==0:
                X_ = F.relu(self.gcn_conv(X,H[i]))
            else:
                X_tmp = F.relu(self.gcn_conv(X,H[i]))
                X_ = torch.cat((X_,X_tmp), dim=1)

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
    
    def __init__(self, in_channels, out_channels, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels)
            self.conv2 = GTConv(in_channels, out_channels)
        else:
            self.conv1 = GTConv(in_channels, out_channels)
    
    def forward(self, A, H_=None):
        if self.first == True:
            a = self.conv1(A)
            b = self.conv2(A)
            H = torch.bmm(a,b)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach(),(F.softmax(self.conv2.weight, dim=1)).detach()]
        else:
            a = self.conv1(A)
            H = torch.bmm(H_,a)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach()]
        return H,W

class GTConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels,1,1))
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        nn.init.constant_(self.weight, 0.1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        A = torch.sum(A*F.softmax(self.weight, dim=1), dim=1)
        return A
