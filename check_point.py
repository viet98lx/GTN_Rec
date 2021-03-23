import torch
import os
import json
import shutil

######## SAVE AND LOAD CHECKPOINT ##########

def save_ckpt(state, model_name, checkpoint_dir, epoch):
    """
    state: checkpoint we want to save
    checkpoint_path: path to save checkpoint
    """
    # parent_folder = os.path.join(checkpoint_dir, model_name)
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print("Directory '%s' created successfully" % checkpoint_dir)
    except OSError as error:
        print("Directory '%s' can not be created" % checkpoint_dir)

    f_path = checkpoint_dir + '/' + model_name + '_ep{}_checkpoint.pt'.format(epoch)
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    print("Save checkpoint successfully at: " + f_path)


def load_ckpt(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])

    # lr = checkpoint['lr']

    return model, optimizer

def load_model_statedict(checkpoint_fpath, model):
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    return model

def save_config_param(model_dir, prefix_name, config_param):
    config_file = model_dir + '/' + prefix_name + '_config.json'
    with open(config_file, 'w') as fp:
        json.dump(config_param, fp)


def load_config_param(path):
    with open(path, 'r') as fp:
        data = json.load(fp)
        return data

def save_score_matrix(model_dir, prefix_name, score_matrix):
    score_file = model_dir + prefix_name + '/' + prefix_name + '_score_matrix.txt'
    with open(score_file, 'a') as fp:
        fp.write("R: %.6f / P: %.6f / F1: %.6f " % (score_matrix[0], score_matrix[1], score_matrix[2]))
        fp.write("MRR: %.6f / HLU: %.6f \n" % (score_matrix[3], score_matrix[4]))

# def save_log_result(log_result_file, train_result, val_result, test_result):
#     with pd.ExcelWriter(log_result_file, mode='w') as writer:
#         train_result.to_excel(writer, sheet_name='train_result_sheet')
#         val_result.to_excel(writer, sheet_name='val_result_sheet')
#         test_result.to_excel(writer, sheet_name='test_result_sheet')