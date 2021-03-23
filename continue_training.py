import argparse
import random
import torch
import numpy as np
import scipy.sparse as sp
import utils
import data_utils
import check_point
import model
import loss
from torch.utils.tensorboard import SummaryWriter
from main import train_model, validate_model, test_model

torch.set_printoptions(precision=8)
parser = argparse.ArgumentParser(description='Continue training model')

parser.add_argument('--ckpt_dir', type=str, help='folder contains check point', required=True)
parser.add_argument('--epoch', type=int, help='number epoch to train', required=True)
parser.add_argument('--model_name', type=str, help='name of model', required=True)
parser.add_argument('--data_dir', type=str, help='folder contains data', required=True)
parser.add_argument('--output_dir', type=str, help='folder to save model', required=True)
parser.add_argument('--config_param_path', type=str, help='folder to save config param', required=True)
parser.add_argument('--ckpt_path', type=str, help='folder to save checkpoint', required=True)
parser.add_argument('--lr', type=float, help='learning rate of optimizer', default=0.01)
parser.add_argument('--top_k', type=int, help='top k predict', default=10)
# parser.add_argument('--cur_epoch', type=int, help='last epoch before interrupt', required=True)
parser.add_argument('--epsilon', type=float, help='different between loss of two consecutive epoch ', default=0.00000001)
parser.add_argument('--nb_hop', type=int, help='level of correlation matrix', default=1)
parser.add_argument('--num_edges', type=int, help='number of adj matrix', default=2)
parser.add_argument('--seed', type=int, help='random seed', default=0)
parser.add_argument('--device', type=str, help='device for train and predict', default='cpu')

args = parser.parse_args()

epoch = args.epoch
model_name = args.model_name
data_dir = args.data_dir
output_dir = args.output_dir
ckpt_dir = args.ckpt_dir
best_ckpt_dir = output_dir
nb_hop = args.nb_hop
config_param_file = args.config_param_path
checkpoint_fpath = args.ckpt_path
config_param = check_point.load_config_param(config_param_file)

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

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

print("---------------------@Build knowledge-------------------------------")
MAX_SEQ_LENGTH, item_dict, reversed_item_dict, item_probs = utils.build_knowledge(train_instances, validate_instances)

print('---------------------Create data loader--------------------')
train_loader = data_utils.generate_data_loader(train_instances, config_param['batch_size'], item_dict, MAX_SEQ_LENGTH, is_bseq=True, is_shuffle=True)
valid_loader = data_utils.generate_data_loader(validate_instances, config_param['batch_size'], item_dict, MAX_SEQ_LENGTH, is_bseq=True, is_shuffle=False)
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
# A = torch.cat([A,torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
A = A.to(device = exec_device, dtype = data_type)
config_param['num_edge'] = len(edges)
config_param['num_class'] = len(item_dict) # number items

rec_sys_model = model.GTN_Rec(config_param, MAX_SEQ_LENGTH, item_probs, exec_device, data_type, num_nodes, norm)
# multiple_gpu = args.multiple_gpu
# if multiple_gpu and torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     rec_sys_model = nn.DataParallel(rec_sys_model)

#### loss and optim ######
loss_func = loss.Weighted_BCE_Loss()
# optimizer = torch.optim.Adam(rec_sys_model.parameters(), lr=0.0001)
optimizer = torch.optim.RMSprop(rec_sys_model.parameters(), lr=args.lr)

rec_sys_model, optimizer = check_point.load_ckpt(checkpoint_fpath, model, optimizer)

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
f1_max = 0.0
loss_min = 10000

log_dir = 'seed_{}'.format(seed)
# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)
########## train #################
writer = SummaryWriter(log_dir='runs/'+log_dir, comment=args.model_name)

for ep in range(epoch):
    avg_train_loss, avg_train_recall, avg_train_prec, avg_train_f1 = train_model(rec_sys_model, exec_device, data_type, config_param['batch_size'], loss_func, optimizer, A, train_loader, ep, top_k, train_display_step)
    # train_losses.append(avg_train_loss)
    # train_recalls.append(avg_train_recall)

    writer.add_scalar("Loss/train", avg_train_loss, ep)
    writer.add_scalar("Recall/train", avg_train_recall, ep)
    writer.add_scalar("Precision/train", avg_train_prec, ep)
    writer.add_scalar("F1/train", avg_train_f1, ep)

    avg_val_loss, avg_val_recall, avg_val_prec, avg_val_f1 = validate_model(rec_sys_model, exec_device, data_type, config_param['batch_size'], loss_func, A, valid_loader,
                                                              ep, top_k, val_display_step)
    writer.add_scalar("Loss/val", avg_val_loss, ep)
    writer.add_scalar("Recall/val", avg_val_recall, ep)
    writer.add_scalar("Precision/val", avg_val_prec, ep)
    writer.add_scalar("F1/val", avg_val_f1, ep)

    avg_test_loss, avg_test_recall, avg_test_prec, avg_test_f1 = test_model(rec_sys_model, exec_device, data_type, config_param['batch_size'], loss_func, A, test_loader,
                                                            ep, top_k, test_display_step)
    writer.add_scalar("Loss/test", avg_test_loss, ep)
    writer.add_scalar("Recall/test", avg_test_recall, ep)
    writer.add_scalar("Precision/test", avg_test_prec, ep)
    writer.add_scalar("F1/test", avg_test_f1, ep)

    state = {'state_dict': rec_sys_model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'lr': args.lr,
             'seed': args.seed
    }
    check_point.save_ckpt(state, args.model_name, output_dir, ep)
    if (avg_test_f1 > f1_max):
        score_matrix = []
        print('Test loss decrease from ({:.6f} --> {:.6f}) '.format(loss_min, avg_test_loss))
        print('Test f1 increase from {:.6f} --> {:.6f}'.format(f1_max, avg_test_f1))
        # check_point.save_ckpt(checkpoint, True, model_name, checkpoint_dir, best_model_dir, ep)
        check_point.save_config_param(output_dir, args.model_name, config_param)
        loss_min = avg_test_loss
        f1_max = avg_test_f1
        torch.save(rec_sys_model, output_dir+'/best_'+args.model_name+'.pt')
        print('Can save model')