import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# import torch.nn.parameter as Parameter
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

    def __init__(self, config_param, max_seq_length, item_probs, device, d_type, num_nodes, norm):
        super(GTN, self).__init__()
        self.num_edge = config_param['num_edge']
        self.num_channels = config_param['num_channels']
        self.basket_embed_dim = config_param['basket_embed_dim']
        self.rnn_units = config_param['rnn_units']
        self.rnn_layers = config_param['rnn_layers']
        self.num_class = config_param['num_class']
        self.num_layers = config_param['num_layers']
        self.batch_size = config_param['batch_size']
        self.alpha = config_param['alpha']
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
        self.list_linear = nn.ModuleList(
            [nn.Linear(self.nb_items, self.basket_embed_dim) for i in range(self.num_channels)])
        self.lstm = nn.LSTM(self.basket_embed_dim, self.rnn_units, self.rnn_layers, bias=True, batch_first=True)
        self.project_embed = nn.Linear(self.basket_embed_dim * self.num_channels, self.basket_embed_dim)
        # self.linear1 = nn.Linear(self.basket_embed_dim * self.num_channels, self.basket_embed_dim)
        self.h2item_score = nn.Linear(in_features=self.rnn_units, out_features=self.nb_items, bias=False)
        # self.linear2 = nn.Linear(self.w_out, self.num_class)
        item_bias = torch.ones(self.nb_items) / self.nb_items
        self.I_B = nn.Parameter(data=item_bias.type(d_type))
        # self.reset_parameters()

    def normalization(self, H):
        norm_H = []
        for i in range(self.num_channels):
            edge, value = H[i]
            edge, value = remove_self_loops(edge, value)
            deg_row, deg_col = self.norm(edge.detach(), self.num_nodes, value.detach())
            value = deg_col * value
            norm_H.append((edge, value))
        return norm_H

    def norm(self, edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        with torch.no_grad():
            if edge_weight is None:
                edge_weight = torch.ones((edge_index.size(1),),
                                         dtype=dtype,
                                         device=edge_index.device)
            edge_weight = edge_weight.view(-1)
            assert edge_weight.size(0) == edge_index.size(1)
            row, col = edge_index
            deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
            deg_inv_sqrt = deg.pow(-1)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row], deg_inv_sqrt[col]

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        weight = next(self.parameters()).data
        return (weight.new(self.rnn_layers, batch_size, self.rnn_units).zero_(),
                weight.new(self.rnn_layers, batch_size, self.rnn_units).zero_())

    def forward(self, A, seq_len, seqs, hidden):
        batch_size = seqs.shape[0]
        Ws = []
        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A)
            else:
                H = self.normalization(H)
                H, W = self.layers[i](A, H)
            Ws.append(W)

        reshape_x = seqs.reshape(-1, self.nb_items)
        item_bias_diag = F.relu(torch.diag(self.I_B))
        for i in range(self.num_channels):
            if i == 0:
                encode_x_graph = F.relu(torch.mm(reshape_x, item_bias_diag)) + F.relu(
                    torch.mm(reshape_x, H[i][:self.nb_items, :self.nb_items]))
                encode_basket = self.list_linear[i](encode_x_graph)
            else:
                encode_x_graph = F.relu(torch.mm(reshape_x, item_bias_diag) + F.relu(
                    torch.mm(reshape_x, H[i][:self.nb_items, :self.nb_items])))
                encode_basket_term = self.list_linear[i](encode_x_graph)
                encode_basket = torch.cat((encode_basket, encode_basket_term), dim=1)

        combine_encode_basket = self.project_embed(encode_basket)
        basket_encoder = combine_encode_basket.reshape(-1, self.max_seq_length, self.basket_embed_dim)
        # basket_encoder = F.dropout(F.relu(self.fc_basket_encoder_1(basket_x)), p=0.2)

        lstm_out, (h_n, c_n) = self.lstm(basket_encoder, hidden)
        actual_index = torch.arange(0, batch_size) * self.max_seq_length + (seq_len - 1)
        actual_lstm_out = lstm_out.reshape(-1, self.rnn_units)[actual_index]

        hidden_to_score = self.h2item_score(actual_lstm_out)
        # print(hidden_to_score)

        # predict next items score
        next_item_probs = torch.sigmoid(hidden_to_score)
        # next_item_probs = hidden_to_score
        predict = (1 - self.alpha) * next_item_probs + self.alpha * torch.mm(next_item_probs, item_bias_diag)
        return predict


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
            W = [(F.softmax(self.conv1.weight, dim=1)).detach(), (F.softmax(self.conv2.weight, dim=1)).detach()]
        else:
            result_A = H_
            result_B = self.conv1(A)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach()]
        H = []
        for i in range(len(result_A)):
            a_edge, a_value = result_A[i]
            b_edge, b_value = result_B[i]

            edges, values = torch_sparse.spspmm(a_edge, a_value, b_edge, b_value, self.num_nodes, self.num_nodes,
                                                self.num_nodes)
            H.append((edges, values))
        return H, W


class GTConv(nn.Module):

    def __init__(self, in_channels, out_channels, num_nodes):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))
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
            for j, (edge_index, edge_value) in enumerate(A):
                if j == 0:
                    total_edge_index = edge_index
                    total_edge_value = edge_value * filter[i][j]
                else:
                    total_edge_index = torch.cat((total_edge_index, edge_index), dim=1)
                    total_edge_value = torch.cat((total_edge_value, edge_value * filter[i][j]))
            index, value = torch_sparse.coalesce(total_edge_index.detach(), total_edge_value, m=self.num_nodes,
                                                 n=self.num_nodes)
            results.append((index, value))
        return results