import model_v2
import model_utils
import utils
import data_utils
import loss
import check_point

import argparse
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import scipy.sparse as sp
import random
import os

torch.set_printoptions(precision=8)
parser = argparse.ArgumentParser(description='Train model')

parser.add_argument('--batch_size', type=int, help='batch size of data set (default:32)', default=32)
parser.add_argument('--rnn_units', type=int, help='number units of hidden size lstm', default=16)
parser.add_argument('--rnn_layers', type=int, help='number layers of RNN', default=1)
parser.add_argument('--num_channels', type=int, default=2, help='number of channels')
# parser.add_argument('--alpha', type=float, help='coefficient of C matrix in predict item score', default=0.4)
parser.add_argument('--lr', type=float, help='learning rate of optimizer', default=0.001)
parser.add_argument('--dropout', type=float, help='drop out after linear model', default= 0.2)
parser.add_argument('--item_embed_dim', type=int, help='dimension of linear layers', default=8)
# parser.add_argument('--embed_dim_transformer', type=int, help='dimension of transformer project layers', default=8)
# parser.add_argument('--transformer_layers', type=int, help='number transformer layers', default=1)
# parser.add_argument('--transformer_head', type=int, help='number heads of transformer layers', default=2)
parser.add_argument('--device', type=str, help='device for train and predict', default='cpu')
parser.add_argument('--top_k', type=int, help='top k predict', default=10)
# parser.add_argument('--nb_hop', type=int, help='level of correlation matrix', default=1)
parser.add_argument('--epoch', type=int, help='epoch to train', default=30)
parser.add_argument('--epsilon', type=float, help='different between loss of two consecutive epoch ', default=0.00000001)
parser.add_argument('--model_name', type=str, help='name of model', required=True)
parser.add_argument('--norm', type=str, default='true', help='normalization')
parser.add_argument('--data_dir', type=str, help='folder contains data', required=True)
parser.add_argument('--output_dir', type=str, help='folder to save model', required=True)
parser.add_argument('--seed', type=int, help='seed for random', default=1)

args = parser.parse_args()
seed = args.seed
torch.manual_seed(seed)

config_param={}
config_param['rnn_units'] = args.rnn_units
config_param['rnn_layers'] = args.rnn_layers
config_param['dropout'] = args.dropout
config_param['w_out'] = args.item_embed_dim # item embedding dim
config_param['batch_size'] = args.batch_size
config_param['num_heads'] = args.transformer_head
config_param['top_k'] = args.top_k
config_param['alpha'] = args.alpha
config_param['num_edge'] = args.num_edge  # num adj matrix edge type
config_param['num_channels'] = args.num_channels # num heads in GTN
data_dir = args.data_dir
output_dir = args.output_dir
nb_hop = args.nb_hop

torch.manual_seed(1)
np.random.seed(2)
random.seed(3)

train_data_path = data_dir + 'train.txt'
train_instances = utils.read_instances_lines_from_file(train_data_path)
nb_train = len(train_instances)
print(nb_train)

validate_data_path = data_dir + 'validate.txt'
valid_instances = utils.read_instances_lines_from_file(validate_data_path)
nb_validate = len(valid_instances)
print(nb_validate)

test_data_path = data_dir + 'test.txt'
test_instances = utils.read_instances_lines_from_file(test_data_path)
nb_test = len(test_instances)
print(nb_test)

### build knowledge ###

print("---------------------@Build knowledge-------------------------------")
MAX_SEQ_LENGTH, item_dict, reversed_item_dict, item_probs, user_consumption_dict = utils.build_knowledge(train_instances, valid_instances, test_instances)

config_param = dict()
config_param['batch_size'] = 16
print('---------------------Create data loader--------------------')
train_loader = data_utils.generate_data_loader(train_instances[:1000], config_param['batch_size'], item_dict, MAX_SEQ_LENGTH, is_bseq=True, is_shuffle=True)
valid_loader = data_utils.generate_data_loader(valid_instances[:500], config_param['batch_size'], item_dict, MAX_SEQ_LENGTH, is_bseq=True, is_shuffle=False)
test_loader = data_utils.generate_data_loader(test_instances, config_param['batch_size'], item_dict, MAX_SEQ_LENGTH, is_bseq=True, is_shuffle=False)

### init model ####
exec_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# exec_device = torch.device('cpu')
data_type = torch.float32
num_nodes = len(item_dict) + len(user_consumption_dict)
config_param['num_class'] = 9984 # number items
config_param['num_layers'] = 2 # len of metapath in GTN
norm = True # normalize adj matrix

rec_sys_model = model_v2.GTN(config_param, MAX_SEQ_LENGTH, item_probs, exec_device, data_type, num_nodes, norm)
rec_sys_model = rec_sys_model.to(exec_device, dtype= data_type)

#### loss and optim ######
loss_func = loss.Weighted_BCE_Loss()
optimizer = torch.optim.Adam(rec_sys_model.parameters(), lr=0.005)
edges = []
i_i_adj = sp.load_npz(data_dir + 'adj_matrix/i_vs_i_sparse.npz')
edges.append(i_i_adj)
u_i_adj = sp.load_npz(data_dir + 'adj_matrix/u_vs_i_sparse.npz')
edges.append(u_i_adj)

############### Dense version ##########################
for i, edge in enumerate(edges):
    if i ==0:
        A = torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)
    else:
        A = torch.cat([A,torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)

# edges.clear()
A = torch.cat([A,torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)

A = A.to(device = exec_device)

print("Device (A, model, X_feature): ")
print(A[0][0].device)
print(rec_sys_model.device)
# print(X_feature.device)

########## train #################
epoch = 2
top_k = 10
train_display_step = 300
train_losses = []
train_recalls = []

for ep in range(epoch):
    rec_sys_model, optimizer, avg_train_loss, avg_train_recall = model_utils.train_model(rec_sys_model, loss_func, optimizer, A, train_loader, ep, top_k, train_display_step)
    train_losses.append(avg_train_loss)
    train_recalls.append(avg_train_recall)

    # avg_val_loss, avg_val_recall = model_utils.validate_model(rec_sys_model, loss_func, valid_loader,
    #                                                           ep, top_k, val_display_step)
    # val_losses.append(avg_val_loss)
    # val_recalls.append(avg_val_recall)

    # avg_test_loss, avg_test_recall = model_utils.test_model(rec_sys_model, loss_func, test_loader,
    #                                                         ep, top_k, test_display_step)
    # test_losses.append(avg_test_loss)
    # test_recalls.append(avg_test_recall)