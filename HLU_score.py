import torch
import utils
import argparse
import check_point
import model
import scipy.sparse as sp
import numpy as np
import data_utils

parser = argparse.ArgumentParser(description='Calculate HLU score')
parser.add_argument('--ckpt_dir', type=str, help='folder contains check point', required=True)
parser.add_argument('--model_name', type=str, help='name of model', required=True)
parser.add_argument('--epoch', type=int, help='last epoch before interrupt', required=True)
parser.add_argument('--data_dir', type=str, help='folder contains data', required=True)
parser.add_argument('--nb_hop', type=int, help='level of correlation matrix', default=1)
parser.add_argument('--batch_size', type=int, help='batch size predict', default=8)

args = parser.parse_args()

prefix_model_name = args.model_name
ckpt_dir = args.ckpt_dir
data_dir = args.data_dir

### init model ####
exec_device = torch.device('cuda:{}'.format(args.device[-1]) if ('gpu' in args.device and torch.cuda.is_available()) else 'cpu')
data_type = torch.float

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

# ckpt_path = ckpt_dir + '/' + prefix_model_ckpt + '/' + 'epoch_' + str(args.epoch) + '/' + prefix_model_ckpt + '_checkpoint.pt'
config_param_file = ckpt_dir + '/' + prefix_model_name + '_config.json'
load_param = check_point.load_config_param(config_param_file)

train_data_path = data_dir + 'train.txt'
train_instances = utils.read_instances_lines_from_file(train_data_path)
nb_train = len(train_instances)
print(nb_train)

validate_data_path = data_dir + 'validate.txt'
validate_instances = utils.read_instances_lines_from_file(validate_data_path)
nb_validate = len(validate_instances)
print(nb_validate)

test_data_path = data_dir + 'test.txt'
test_instances = utils.read_instances_lines_from_file(test_data_path)
nb_test = len(test_instances)
print(nb_test)

### build knowledge ###

print("@Build knowledge")
MAX_SEQ_LENGTH, item_dict, reversed_item_dict, item_probs = utils.build_knowledge(train_instances, validate_instances)

num_nodes = len(item_dict)
# A = torch.cat([A,torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
A = A.to(device = exec_device, dtype = data_type)

batch_size = args.batch_size
# train_loader = data_utils.generate_data_loader(train_instances, load_param['batch_size'], item_dict, MAX_SEQ_LENGTH, is_bseq=True, is_shuffle=True)
# valid_loader = data_utils.generate_data_loader(validate_instances, load_param['batch_size'], item_dict, MAX_SEQ_LENGTH, is_bseq=True, is_shuffle=False)
test_loader = data_utils.generate_data_loader(test_instances, batch_size, item_dict, MAX_SEQ_LENGTH, is_bseq=True, is_shuffle=True)
model_path = 'best_' + prefix_model_name + '.pt'
load_model = torch.load(ckpt_dir+'/'+model_path)


def HLU_score_for_data(model, A, data_loader, batch_size):
    device = model.device
    nb_batch = len(data_loader.dataset) // batch_size
    if len(data_loader.dataset) % batch_size == 0:
        total_batch = nb_batch
    else :
        total_batch = nb_batch + 1
    print(total_batch)
    list_HLU_score = []
    C = 100
    beta = 5
    model.eval()
    for i, data_pack in enumerate(data_loader, 0):
        data_x, data_seq_len, data_y = data_pack
        x_ = data_x.to_dense().to(dtype=model.dtype, device=device)
        real_batch_size = x_.size()[0]
        # hidden = model.init_hidden(real_batch_size)
        y_ = data_y.to(dtype=model.dtype, device=device)
        predict_ = model(A, data_seq_len, x_)
        sigmoid_pred = torch.sigmoid(predict_)
        sorted_rank, indices = torch.sort(sigmoid_pred, descending = True)
        for seq_idx, a_seq_idx in enumerate(y_):
            # print(seq_idx)
            idx_item_in_target_basket = (a_seq_idx == 1.0).nonzero()
            # print(idx_item_in_target_basket)
            sum_of_rank_score = 0
            for idx_item in idx_item_in_target_basket:
                item_rank = (indices[seq_idx] == idx_item).nonzero().item()
                rank_score = 2**((1-(item_rank+1))/(beta-1))
                sum_of_rank_score += rank_score

            sum_of_rank_target_basket = 0
            # tinh mau so cua HLU
            target_basket_size = idx_item_in_target_basket.size()[0]
            for r in range(1, target_basket_size+1):
                target_rank_score = 2**((1-r)/(beta-1))
                sum_of_rank_target_basket += target_rank_score
            HLU_score = C * sum_of_rank_score / sum_of_rank_target_basket
            list_HLU_score.append(HLU_score)

    print("HLU list len: %d" % len(list_HLU_score))
    return np.array(list_HLU_score).mean()

avg_HLU_score = HLU_score_for_data(load_model, test_loader, batch_size)
print("HLU score: %.6f" % avg_HLU_score)