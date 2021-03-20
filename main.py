import model
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
import time
import os
from torch.utils.tensorboard import SummaryWriter

def train_model(model, device, dtype, batch_size, loss_func, optimizer, A, train_loader, epoch, top_k, train_display_step):
    running_train_loss = 0.0
    running_train_recall = 0.0
    running_train_prec = 0.0
    running_train_f1 = 0.0
    # device = model.device
    # dtype = model.dtype
    nb_train_batch = len(train_loader.dataset) // batch_size
    if len(train_loader.dataset) % batch_size == 0:
        total_train_batch = nb_train_batch
    else:
        total_train_batch = nb_train_batch + 1
    model.train()
    start = time.time()

    for i, data in enumerate(train_loader, 0):

        user_seq, train_seq_len, target_basket = data
        x_train_batch = user_seq.to_dense().to(dtype=dtype, device=device)
        real_batch_size = x_train_batch.size()[0]
        # hidden = model.init_hidden(real_batch_size)
        target_basket_train = target_basket.to(device=device, dtype=dtype)

        optimizer.zero_grad()  # clear gradients for this training step

        predict = model(A, train_seq_len, x_train_batch)  # predicted output
        loss = loss_func(predict, target_basket_train)  # WBCE loss
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # update gradient

        train_loss_item = loss.item()
        running_train_loss += train_loss_item
        avg_train_loss = running_train_loss / (i + 1)

        train_recall_item, train_prec_item, train_f1_item = utils.compute_recall_at_top_k(model, predict.detach(), top_k, target_basket_train.detach(), real_batch_size)
        running_train_recall += train_recall_item
        running_train_prec += train_prec_item
        running_train_f1 += train_f1_item
        avg_train_recall = running_train_recall / (i + 1)
        avg_train_prec = running_train_prec / (i + 1)
        avg_train_f1 = running_train_f1 / (i + 1)

        end = time.time()

        if ((i + 1) % train_display_step == 0 or (i + 1) == total_train_batch):  # print every 50 mini-batches
            top_pred = predict.clone().detach().topk(dim=-1, k=top_k, sorted=True)
            print(
                '[Epoch : % d ,Batch Index : %d / %d] Train Loss : %.8f ----- Train Recall@%d: %.8f / Train Precision: %.8f / Train F1: %.8f ----- Time : %.3f seconds ' %
                (epoch, i + 1, total_train_batch, avg_train_loss, top_k, avg_train_recall, avg_train_prec, avg_train_f1, end - start))
            print("top k indices predict: ")
            print('--------------------------------------------------------------')
            print('*****  indices *****')
            print(top_pred.indices)
            print('*****  values *****')
            print(top_pred.values)
            print('--------------------------------------------------------------')

            start = time.time()
    torch.cuda.empty_cache()
    print('finish a train epoch')
    return avg_train_loss, avg_train_recall


def validate_model(model, device, dtype, batch_size, loss_func, valid_loader, epoch, top_k, val_display_step):
    running_val_loss = 0.0
    running_val_recall = 0.0
    running_val_prec = 0.0
    running_val_f1 = 0.0
    # device = model.device
    nb_val_batch = len(valid_loader.dataset) // batch_size
    if len(valid_loader.dataset) % batch_size == 0:
        total_val_batch = nb_val_batch
    else:
        total_val_batch = nb_val_batch + 1

    model.eval()
    with torch.no_grad():
        for valid_i, valid_data in enumerate(valid_loader, 0):
            valid_in, valid_seq_len, valid_out = valid_data
            x_valid = valid_in.to_dense().to(dtype=dtype, device=device)
            val_batch_size = x_valid.size()[0]
            # hidden = model.init_hidden(val_batch_size)
            y_valid = valid_out.to(device=device, dtype=dtype)

            valid_predict = model(A, x_valid, valid_seq_len)
            val_loss = loss_func(valid_predict, y_valid)

            val_loss_item = val_loss.item()
            running_val_loss += val_loss_item
            avg_val_loss = running_val_loss / (valid_i + 1)

            val_recall_item, val_prec_item, val_f1_item = utils.compute_recall_at_top_k(model, valid_predict, top_k, y_valid, val_batch_size)

            running_val_recall += val_recall_item
            running_val_prec += val_prec_item
            running_val_f1 += val_f1_item

            avg_val_recall = running_val_recall / (valid_i + 1)
            avg_val_prec = running_val_prec / (valid_i + 1)
            avg_val_f1 = running_val_f1 / (valid_i + 1)

            if ((valid_i + 1) % val_display_step == 0 or (
                    valid_i + 1) == total_val_batch):  # print every 50 mini-batches
                print('[Epoch : % d ,Valid batch Index : %d / %d] Valid Loss : %.8f ----- Valid Recall@%d: %.8f / Valid Precision: %.8f / Valid F1: %.8f' %
                      (epoch, valid_i + 1, total_val_batch, avg_val_loss, top_k, avg_val_recall, avg_val_prec, avg_val_f1))

    return avg_val_loss, avg_val_recall


def test_model(model, device, dtype, batch_size, loss_func, test_loader, epoch, top_k, test_display_step):
    running_test_recall = 0.0
    running_test_loss = 0.0
    running_test_prec = 0.0
    running_test_f1 = 0.0
    # device = model.device
    nb_test_batch = len(test_loader.dataset) // batch_size
    if len(test_loader.dataset) % batch_size == 0:
        total_test_batch = nb_test_batch
    else:
        total_test_batch = nb_test_batch + 1

    model.eval()
    with torch.no_grad():
        for test_i, test_data in enumerate(test_loader, 0):
            test_in, test_seq_len, test_out = test_data
            x_test = test_in.to_dense().to(dtype=dtype, device=device)
            real_test_batch_size = x_test.size()[0]
            # hidden = model.init_hidden(real_test_batch_size)
            y_test = test_out.to(device=device, dtype=dtype)

            test_predict = model(A, x_test, test_seq_len)
            test_loss = loss_func(test_predict, y_test)

            test_loss_item = test_loss.item()
            running_test_loss += test_loss_item
            avg_test_loss = running_test_loss / (test_i + 1)

            test_recall_item, test_prec_item, test_f1_item = utils.compute_recall_at_top_k(model, test_predict, top_k, y_test, real_test_batch_size)
            running_test_recall += test_recall_item
            running_test_prec += test_prec_item
            running_test_f1 += test_f1_item
            avg_test_recall = running_test_recall / (test_i + 1)
            avg_test_prec = running_test_prec / (test_i + 1)
            avg_test_f1 = running_test_f1 / (test_i + 1)
            if ((test_i + 1) % test_display_step == 0 or (test_i + 1) == total_test_batch):
                print('[Epoch : % d , Test batch_index : %3d --------- Test loss: %.8f ----- Test Recall@%d : %.8f / Test Prec: %.8f / Test F1: %.8f' %
                      (epoch, test_i + 1, avg_test_loss, top_k, avg_test_recall, avg_test_prec, avg_test_f1))

    return avg_test_loss, avg_test_recall

torch.set_printoptions(precision=8)
parser = argparse.ArgumentParser(description='Train model')

parser.add_argument('--batch_size', type=int, help='batch size of data set (default:16)', default=16)
parser.add_argument('--rnn_units', type=int, help='number units of hidden size lstm', default=16)
parser.add_argument('--rnn_layers', type=int, help='number layers of RNN', default=1)
parser.add_argument('--num_gtn_layers', type=int, help='number layers of GTN', default=1)
parser.add_argument('--num_channels', type=int, default=2, help='number of GTN channels')
parser.add_argument('--alpha', type=float, help='coefficient item bias in predict item score', default=0.4)
parser.add_argument('--lr', type=float, help='learning rate of optimizer', default=0.001)
parser.add_argument('--dropout', type=float, help='drop out after linear model', default= 0.2)
parser.add_argument('--basket_embed_dim', type=int, help='dimension of linear layers', default=8)
parser.add_argument('--device', type=str, help='device for train and predict', default='cpu')
parser.add_argument('--multiple_gpu', type=int, default=0)
parser.add_argument('--topk', type=int, help='top k predict', default=10)
parser.add_argument('--num_edges', type=int, help='number of adj matrix', default=2)
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

data_type = torch.float
# X_feature = torch.rand((num_nodes, 16)).to(exec_device)
# X_feature = torch.eye(num_nodes, dtype=data_type, device=exec_device)



config_param={}
config_param['basket_embed_dim'] = args.basket_embed_dim
config_param['rnn_units'] = args.rnn_units
config_param['rnn_layers'] = args.rnn_layers
config_param['dropout'] = args.dropout
config_param['batch_size'] = args.batch_size
config_param['top_k'] = args.topk
config_param['alpha'] = args.alpha
config_param['num_layers'] = 1 # len of metapath in GTN
config_param['num_channels'] = 1 # num heads in transformer

data_dir = args.data_dir
output_dir = args.output_dir

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

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
MAX_SEQ_LENGTH, item_dict, reversed_item_dict, item_probs = utils.build_knowledge(train_instances, valid_instances)

print('---------------------Create data loader--------------------')
train_loader = data_utils.generate_data_loader(train_instances, config_param['batch_size'], item_dict, MAX_SEQ_LENGTH, is_bseq=True, is_shuffle=True)
valid_loader = data_utils.generate_data_loader(valid_instances, config_param['batch_size'], item_dict, MAX_SEQ_LENGTH, is_bseq=True, is_shuffle=False)
test_loader = data_utils.generate_data_loader(test_instances, config_param['batch_size'], item_dict, MAX_SEQ_LENGTH, is_bseq=True, is_shuffle=False)

### init model ####
exec_device = torch.device('cuda:{}'.format(args.device[-1]) if ('gpu' in args.device and torch.cuda.is_available()) else 'cpu')
data_type = torch.float
# num_nodes = len(item_dict) + len(user_consumption_dict)

norm = True # normalize adj matrix

edges = []

for i in range(args.num_edges):
    adj_matrix = sp.load_npz(data_dir + 'adj_matrix/v2_r_matrix_' + str(i+1) + 'w.npz')
    edges.append(adj_matrix)

############### Dense version ##########################
for i, edge in enumerate(edges):
    if i ==0:
        A = torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)
    else:
        A = torch.cat([A,torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)

# edges.clear()
num_nodes = len(item_dict)
A = torch.cat([A,torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
A = A.to(device = exec_device, dtype = data_type)
config_param['num_edge'] = len(edges)+1
config_param['num_class'] = len(item_dict) # number items

rec_sys_model = model.GTN_Rec(config_param, MAX_SEQ_LENGTH, item_probs, exec_device, data_type, num_nodes, norm)
# multiple_gpu = args.multiple_gpu
# if multiple_gpu and torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     rec_sys_model = nn.DataParallel(rec_sys_model)
rec_sys_model = rec_sys_model.to(exec_device, dtype= data_type)

#### loss and optim ######
loss_func = loss.Weighted_BCE_Loss()
# optimizer = torch.optim.Adam(rec_sys_model.parameters(), lr=0.0001)
optimizer = torch.optim.RMSprop(rec_sys_model.parameters(), lr=args.lr)

print("Device (A, model, X_feature): ")
print(A[0][0].device)
# print(rec_sys_model.device)
# print(X_feature.device)

########## train #################
writer = SummaryWriter()
epoch = args.epoch
top_k = args.topk
train_display_step = 300
val_display_step = 100
test_display_step = 30
train_losses = []
train_recalls = []
val_losses = []
val_recalls = []
test_losses = []
test_recalls = []
recall_max = 0.0
loss_min = 10000

for ep in range(epoch):
    avg_train_loss, avg_train_recall = train_model(rec_sys_model, exec_device, data_type, config_param['batch_size'], loss_func, optimizer, A, train_loader, ep, top_k, train_display_step)
    # train_losses.append(avg_train_loss)
    # train_recalls.append(avg_train_recall)

    writer.add_scalar("Loss/train", avg_train_loss, ep)
    writer.add_scalar("Recall/train", avg_train_recall, ep)
    # writer.add_scalar("Precision/train", avg_train_precision, ep)
    # writer.add_scalar("F1/train", avg_train_f1, ep)

    avg_val_loss, avg_val_recall = validate_model(rec_sys_model, exec_device, data_type, config_param['batch_size'], loss_func, valid_loader,
                                                              ep, top_k, val_display_step)
    writer.add_scalar("Loss/val", avg_val_loss, ep)
    writer.add_scalar("Recall/val", avg_val_recall, ep)
    # writer.add_scalar("Precision/val", avg_val_precision, ep)
    # writer.add_scalar("F1/val", avg_val_f1, ep)
    # val_losses.append(avg_val_loss)
    # val_recalls.append(avg_val_recall)

    avg_test_loss, avg_test_recall = test_model(rec_sys_model, exec_device, data_type, config_param['batch_size'], loss_func, test_loader,
                                                            ep, top_k, test_display_step)
    # test_losses.append(avg_test_loss)
    # test_recalls.append(avg_test_recall)
    writer.add_scalar("Loss/test", avg_test_loss, ep)
    writer.add_scalar("Recall/test", avg_test_recall, ep)
    # writer.add_scalar("Precision/test", avg_test_precision, ep)
    # writer.add_scalar("F1/test", avg_test_f1, ep)
    if (avg_test_recall > recall_max):
        score_matrix = []
        print('Test loss decrease from ({:.6f} --> {:.6f}) '.format(loss_min, avg_test_loss))
        print('F1 increase from {:.6f} --> {:.6f}'.format(recall_max, avg_test_recall))
        # check_point.save_ckpt(checkpoint, True, model_name, checkpoint_dir, best_model_dir, ep)
        check_point.save_config_param(output_dir, args.model_name, config_param)
        loss_min = avg_test_loss
        recall_max = avg_test_recall
        torch.save(rec_sys_model, output_dir+'/best_'+args.model_name+'.pt')
        print('Can save model')

        # avg_R_score, avg_P_score, avg_F1_score = matrix_score_utils.F1_matrix_score_for_data(rec_sys_model, A,
        #                                                                                      test_loader,
        #                                                                                      config_param['batch_size'],
        #                                                                                      top_k)
        # avg_MRR_score = matrix_score_utils.MRR_score_for_data(rec_sys_model, A, test_loader, config_param['batch_size'])
        # avg_HLU_score = matrix_score_utils.HLU_score_for_data(rec_sys_model, A, test_loader, config_param['batch_size'])
        # score_matrix.extend([avg_R_score, avg_P_score, avg_F1_score, avg_MRR_score, avg_HLU_score])
        # check_point.save_score_matrix(best_model_dir, model_name, score_matrix)
        # score_matrix.clear()

    print('-' * 100)

    writer.flush()
    writer.close()