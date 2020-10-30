import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from matplotlib import pyplot as plt
import pdb
import utils
import GTN


class GTN_Rec(nn.Module):

    def __init__(self, config_param, max_seq_length, item_probs, device, d_type, num_nodes, norm):
        super(GTN_Rec, self).__init__()
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
        # layers = []
        # for i in range(self.num_layers):
        #     if i == 0:
        #         layers.append(GTLayer(self.num_edge, self.num_channels, first=True))
        #     else:
        #         layers.append(GTLayer(self.num_edge, self.num_channels, first=False))
        # self.layers = nn.ModuleList(layers)
        self.gtn = GTN.GTLayer(self.num_edge, self.num_channels, first=True)
        self.list_linear = nn.ModuleList([nn.Linear(self.nb_items, self.basket_embed_dim) for i in range(self.num_channels)])
        self.lstm = nn.LSTM(self.basket_embed_dim, self.rnn_units, self.rnn_layers, bias=True, batch_first=True)
        self.project_embed = nn.Linear(self.basket_embed_dim * self.num_channels, self.basket_embed_dim)
        self.h2item_score = nn.Linear(in_features=self.rnn_units, out_features=self.nb_items, bias=False)
        item_bias = torch.ones(self.nb_items) / self.nb_items
        self.I_B = nn.Parameter(data=item_bias.type(d_type))
        self.threshold = nn.Parameter(data=torch.Tensor([0.001]))
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.weight)
        # nn.init.zeros_(self.bias)
        for i in range(self.num_channels):
            nn.init.kaiming_uniform_(self.list_linear[i].weight)
        nn.init.kaiming_uniform_(self.project_embed.weight)
        nn.init.xavier_uniform_(self.h2item_score.weight)

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        weight = next(self.parameters()).data
        return (weight.new(self.rnn_layers, batch_size, self.rnn_units).zero_(),
                weight.new(self.rnn_layers, batch_size, self.rnn_units).zero_())

    def forward(self, A, seq_len, seqs, hidden):
        batch_size = seqs.shape[0]
        # Learn new structure graph by combine adjacency matrices
        A = A.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        Ws = []
        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A)
            else:
                # H = self.normalization(H)
                H, W = self.layers[i](A, H)
            Ws.append(W)

        # H,W1 = self.layer1(A)
        # H = self.normalization(H)
        # H,W2 = self.layer2(A, H)
        # H = self.normalization(H)
        # H,W3 = self.layer3(A, H)

        reshape_x = seqs.reshape(-1, self.nb_items)
        item_bias_diag = F.relu(torch.diag(self.I_B))
        # for i in range(self.num_channels):
        #     if i == 0:
        #         encode_x_graph = F.relu(torch.mm(reshape_x, item_bias_diag)) + F.relu(
        #             torch.mm(reshape_x, H[i][:self.nb_items, :self.nb_items]))
        #         encode_basket = F.relu(self.list_linear[i](encode_x_graph))
        #     else:
        #         encode_x_graph = F.relu(torch.mm(reshape_x, item_bias_diag) + F.relu(
        #             torch.mm(reshape_x, H[i][:self.nb_items, :self.nb_items])))
        #         encode_basket_term = F.relu(self.list_linear[i](encode_x_graph))
        #         encode_basket = torch.cat((encode_basket, encode_basket_term), dim=1)

        encode_x_graph = torch.mm(reshape_x, item_bias_diag) + F.relu(
            torch.mm(reshape_x, H[0]) - self.threshold)
        encode_basket = F.relu(self.list_linear[0](encode_x_graph))

        # combine_encode_basket = F.relu(self.project_embed(encode_basket))
        basket_encoder = encode_basket.reshape(-1, self.max_seq_length, self.basket_embed_dim)

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

