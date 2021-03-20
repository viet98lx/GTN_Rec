import torch
import utils
import argparse
import check_point
import model
import scipy.sparse as sp
import numpy as np
import data_utils

def F1_matrix_score_for_data(model, A, data_loader, batch_size, top_k):
    device = model.device
    nb_batch = len(data_loader.dataset) // batch_size
    if len(data_loader.dataset) % batch_size == 0:
        total_batch = nb_batch
    else:
        total_batch = nb_batch + 1
    print("Total batch: ", total_batch)
    list_R_score = []
    list_P_score = []
    list_F1_score = []
    model.eval()
    with torch.no_grad():
        for i, data_pack in enumerate(data_loader, 0):
            data_x, data_seq_len, data_y = data_pack
            x_ = data_x.to_dense().to(dtype=model.dtype, device=device)
            real_batch_size = x_.size()[0]
            hidden = model.init_hidden(real_batch_size)
            y_ = data_y.to(dtype=model.dtype, device=device)
            logits_ = model(A, data_seq_len, x_, hidden)
            predict_basket = utils.predict_top_k(logits_, top_k, real_batch_size, model.nb_items)
            target_basket_np = y_.cpu().numpy()
            correct_predict = predict_basket * target_basket_np
            nb_correct = np.count_nonzero(correct_predict, axis=1)
            actual_basket_size = np.count_nonzero(target_basket_np, axis=1)
            batch_recall = nb_correct / actual_basket_size
            batch_precision = nb_correct / top_k
            batch_f1 = np.zeros_like(nb_correct, dtype=float)
            for i in range(batch_f1.size()[0]):
                if (nb_correct[i] > 0):
                    batch_f1[i] = (2 * (batch_precision[i] * batch_recall[i])) / (batch_precision[i] + batch_recall[i])
                list_P_score.append(batch_precision[i])
                list_R_score.append(batch_recall[i])
                list_F1_score.append(batch_f1[i])

            # print(list_MRR_score)
            # print("MRR score: %.6f" % np.array(list_MRR_score).mean())

    return np.array(list_R_score).mean(), np.array(list_P_score).mean(), np.array(list_F1_score).mean()

def MRR_score_for_data(model, A, data_loader, batch_size):
    device = model.device
    nb_batch = len(data_loader.dataset) // batch_size
    if len(data_loader.dataset) % batch_size == 0:
        total_batch = nb_batch
    else:
        total_batch = nb_batch + 1
    print("Total batch: ",total_batch)
    list_MRR_score = []
    model.eval()
    with torch.no_grad():
        for i, data_pack in enumerate(data_loader, 0):
            data_x, data_seq_len, data_y = data_pack
            x_ = data_x.to_dense().to(dtype=model.dtype, device=device)
            real_batch_size = x_.size()[0]
            hidden = model.init_hidden(real_batch_size)
            y_ = data_y.to(dtype=model.dtype, device=device)
            predict_ = model(A, data_seq_len, x_, hidden)
            sigmoid_pred = torch.sigmoid(predict_)
            sorted_rank, indices = torch.sort(sigmoid_pred, descending=True)
            for seq_idx, a_seq_idx in enumerate(y_):
                # print(seq_idx)
                idx_item_in_target_basket = (a_seq_idx == 1.0).nonzero(as_tuple=True)[0]
                # print(idx_item_in_target_basket)
                sum_of_rank_score = 0
                for idx_item in idx_item_in_target_basket:
                    item_rank = (indices[seq_idx] == idx_item).nonzero(as_tuple=True)[0].item()
                    # print("Rank %d" % (item_rank + 1))
                    rank_score = 1 / (item_rank + 1)
                    sum_of_rank_score += rank_score
                # print("sum of rank item in target: %.6f" % sum_of_rank_score)

                target_basket_size = idx_item_in_target_basket.size()[0]
                MRR_score = sum_of_rank_score / target_basket_size
                # print(MRR_score)
                list_MRR_score.append(MRR_score)

            # print(list_MRR_score)
            # print("MRR score: %.6f" % np.array(list_MRR_score).mean())
        # print("MRR list len: %d" % len(list_MRR_score))
    return np.array(list_MRR_score).mean()

def HLU_score_for_data(model, A, data_loader, batch_size):
    device = model.device
    nb_batch = len(data_loader.dataset) // batch_size
    if len(data_loader.dataset) % batch_size == 0:
        total_batch = nb_batch
    else :
        total_batch = nb_batch + 1
    print("Total batch: ", total_batch)
    list_HLU_score = []
    C = 100
    beta = 5
    model.eval()
    with torch.no_grad():
        for i, data_pack in enumerate(data_loader, 0):
            data_x, data_seq_len, data_y = data_pack
            x_ = data_x.to_dense().to(dtype=model.dtype, device=device)
            real_batch_size = x_.size()[0]
            hidden = model.init_hidden(real_batch_size)
            y_ = data_y.to(dtype=model.dtype, device=device)
            predict_ = model(A, data_seq_len, x_, hidden)
            sigmoid_pred = torch.sigmoid(predict_)
            sorted_rank, indices = torch.sort(sigmoid_pred, descending = True)
            for seq_idx, a_seq_idx in enumerate(y_):
                # print(seq_idx)
                idx_item_in_target_basket = (a_seq_idx == 1.0).nonzero(as_tuple=True)[0]
                # print(idx_item_in_target_basket)
                sum_of_rank_score = 0
                for idx_item in idx_item_in_target_basket:
                    item_rank = (indices[seq_idx] == idx_item).nonzero(as_tuple=True)[0].item()
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

        # print("HLU list len: %d" % len(list_HLU_score))
    return np.array(list_HLU_score).mean()