import os
import torch
import utils
import argparse
import check_point
import model
import scipy.sparse as sp
import data_utils

def generate_predict(model, A, data_loader, result_file, reversed_item_dict, number_predict, batch_size):
    device = model.device
    print("device of model", next(model.parameters()).device)
    nb_test_batch = len(data_loader.dataset) // batch_size
    if len(data_loader.dataset) % model.batch_size == 0:
        total_batch = nb_test_batch
    else :
        total_batch = nb_test_batch + 1
    print("Total Batch in data set %d" % total_batch)
    model.eval()
    with open(result_file, 'w') as f:
        f.write('Predict result: ')
        for i, data_pack in enumerate(data_loader,0):
            data_x, data_seq_len, data_y = data_pack
            x_ = data_x.to_dense().to(dtype = model.dtype, device = device)
            print("Device: ",device)
            real_batch_size = x_.size()[0]
            # hidden = model.init_hidden(real_batch_size)
            y_ = data_y.to(dtype = model.dtype, device = device)
            predict_ = model(A, data_seq_len, x_)
            sigmoid_pred = torch.sigmoid(predict_)
            topk_result = sigmoid_pred.topk(dim=-1, k= number_predict, sorted=True)
            indices = topk_result.indices
            # print(indices)
            values = topk_result.values

            for row in range(0, indices.size()[0]):
                f.write('\n')
                f.write('ground truth: ')
                ground_truth = y_[row].nonzero().squeeze(dim=-1)
                for idx_key in range(0, ground_truth.size()[0]):
                    f.write(str(reversed_item_dict[ground_truth[idx_key].item()]) + " ")
                f.write('\n')
                f.write('predicted items: ')
                for col in range(0, indices.size()[1]):
                    f.write('| ' + str(reversed_item_dict[indices[row][col].item()]) + ': %.8f' % (values[row][col].item()) + ' ')

# def recall_for_data(model, A, data_loader, topK, batch_size):
#     device = model.device
#     nb_batch = len(data_loader.dataset) // batch_size
#     if len(data_loader.dataset) % batch_size == 0:
#         total_batch = nb_batch
#     else :
#         total_batch = nb_batch + 1
#     print(total_batch)
#     list_correct_predict = []
#     list_actual_size = []
#
#     model.eval()
#     for idx, data_pack in enumerate(data_loader,0):
#         x_, data_seq_len, y_ = data_pack
#         x_test = x_.to_dense().to(dtype = model.d_type, device = device)
#         real_batch_size = x_test.size()[0]
#         hidden = model.init_hidden(real_batch_size)
#         y_test = y_.to(device = device, dtype = model.d_type)
#
#         logits_predict = model(A, data_seq_len, x_test)
#
#         predict_basket = utils.predict_top_k(logits_predict, topK, real_batch_size, model.nb_items)
#         correct_predict = predict_basket * y_test
#         nb_correct = (correct_predict != 0.0).sum(dim = -1)
#         actual_basket_size = (y_test != 0.0).sum(dim = -1)
#         for i in range(0, real_batch_size):
#             list_correct_predict.append(nb_correct[i].item())
#             list_actual_size.append(actual_basket_size[i].item())

parser = argparse.ArgumentParser(description='Generate predict')
parser.add_argument('--ckpt_dir', type=str, help='folder contains check point', required=True)
parser.add_argument('--model_name', type=str, help='name of model', required=True)
# parser.add_argument('--epoch', type=int, help='last epoch before interrupt', required=True)
parser.add_argument('--data_dir', type=str, help='folder contains data', required=True)
parser.add_argument('--device', type=str, help='device for train and predict', default='cpu')
parser.add_argument('--num_edges', type=int, help='Number of edges', default=1)
# parser.add_argument('--nb_hop', type=int, help='level of correlation matrix', default=1)
parser.add_argument('--batch_size', type=int, help='batch size predict', default=8)
parser.add_argument('--nb_predict', type=int, help='number items predicted', default=10)
parser.add_argument('--log_result_dir', type=str, help='folder to save result', required=True)

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
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_data_type = torch.float32

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

# edges.clear()
num_nodes = len(item_dict)
# A = torch.cat([A,torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
A = A.to(device = exec_device, dtype = data_type)

batch_size = args.batch_size
# train_loader = data_utils.generate_data_loader(train_instances, load_param['batch_size'], item_dict, MAX_SEQ_LENGTH, is_bseq=True, is_shuffle=True)
# valid_loader = data_utils.generate_data_loader(validate_instances, load_param['batch_size'], item_dict, MAX_SEQ_LENGTH, is_bseq=True, is_shuffle=False)
test_loader = data_utils.generate_data_loader(test_instances, batch_size, item_dict, MAX_SEQ_LENGTH, is_bseq=True, is_shuffle=True)
model_path = 'best_' + prefix_model_name + '.pt'
load_model = torch.load(ckpt_dir+'/'+model_path, map_location='cpu')
load_model = load_model.to(device = exec_device, dtype = data_type)
load_model.device = exec_device
load_model.threshold = load_model.threshold.to(device = exec_device, dtype = data_type)

log_folder = os.path.join(args.log_result_dir, prefix_model_name)
if(not os.path.exists(log_folder)):
  try:
    os.makedirs(log_folder, exist_ok = True)
    print("Directory '%s' created successfully" % log_folder)
  except OSError as error:
      print("OS folder error")

nb_predict = args.nb_predict
result_file = log_folder + '/' + prefix_model_name + '_predict_top_' + str(nb_predict) + '.txt'
generate_predict(load_model, A, test_loader, result_file, reversed_item_dict, nb_predict, batch_size)