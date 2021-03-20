import glob, os
import matplotlib.pyplot as plt
import torch

def  generate_top_k_result(data_loader, save_folder, A, model, reversed_item_dict, top_k, batch_size):
    result_file = os.path.join(save_folder, 'result_top' + str(top_k) + '.txt')
    device = model.device
    dtype = model.dtype
    nb_test_batch = len(data_loader.dataset) // batch_size
    if len(data_loader.dataset) % model.batch_size == 0:
        total_batch = nb_test_batch
    else :
        total_batch = nb_test_batch + 1
    print("Total Batch in data set %d" % total_batch)
    model.eval()
    with open(result_file, 'w') as f:
        f.write('Predict result: ')
        for test_i, test_data in enumerate(data_loader, 0):
            test_in, test_seq_len, test_out = test_data
            x_test = test_in.to_dense().to(dtype=dtype, device=device)
            real_test_batch_size = x_test.size()[0]
            hidden = model.init_hidden(real_test_batch_size)
            y_test = test_out.to(device=device, dtype=dtype)

            logits = model(A, test_seq_len, x_test, hidden)
            sigmoid_pred = torch.sigmoid(logits)
            topk_result = sigmoid_pred.topk(dim=-1, k= top_k, sorted=True)
            indices = topk_result.indices
            # print(indices)
            values = topk_result.values

            for row in range(0, indices.size()[0]):
                f.write('\n')
                f.write('ground truth: ')
                ground_truth = y_test[row].nonzero(as_tuple=True)[0].squeeze(dim=-1)
                for idx_key in range(0, ground_truth.size()[0]):
                    f.write(str(reversed_item_dict[ground_truth[idx_key].item()]) + " ")
                f.write('\n')
                f.write('predicted items: ')
                for col in range(0, indices.size()[1]):
                    f.write('| ' + str(reversed_item_dict[indices[row][col].item()]) + ': %.8f' % (values[row][col].item()) + ' ')

def stattistic_result(result_file, top_k):
  list_seq = []
  list_seq_topk_predicted = []
  with open(result_file, 'r') as f:
      lines = f.readlines()
      for i, line in enumerate(lines):
          if(i == 0):
              continue
          if(i % 2 == 1):
              ground_truth = line.split(':')[-1]
              list_item = ground_truth.split()
              list_seq.append(list_item.copy())
              list_item.clear()
          if(i % 2 == 0):
              predicted_items = line.split('|')[1:top_k+1]
              list_top_k_item = []
              for item in predicted_items:
                  item_key = item.strip().split(':')[0]
                  list_top_k_item.append(item_key)
              list_seq_topk_predicted.append(list_top_k_item.copy())
              list_top_k_item.clear()
  return list_seq, list_seq_topk_predicted

def plot_correct_result(list_seq, list_seq_topk_predicted):
  list_correct=[]
  for i, ground_truth in enumerate(list_seq):
    correct = 0
    # print(i)
    for item in list_seq_topk_predicted[i]:
      if (item in ground_truth):
          correct += 1
  # recall_score = float(correct) / float(len(ground_truth))
    list_correct.append(correct)
  list_index_seq = [*range(len(list_correct))]
  fig= plt.figure()
  fig.patch.set_facecolor('xkcd:mint green')
  ax = fig.add_subplot(111)
  ax.bar(list_index_seq,list_correct)

def get_freq_dict_from_list(list_seq):
  ground_truth_item_freq = {}
  for i, ground_truth in enumerate(list_seq):
    for item in ground_truth:
      if item in ground_truth_item_freq.keys():
        ground_truth_item_freq[item] += 1
      else:
        ground_truth_item_freq[item] = 1
  return ground_truth_item_freq

def plot_item_freq(freq_dict):
  fig = plt.figure(figsize=(20, 3))  # width:20, height:3
  fig.patch.set_facecolor('xkcd:mint green')
  ax = fig.add_subplot(111)
  item_id = freq_dict.keys()
  freq = [freq_dict[i] for i in item_id]
  ax.bar(item_id, freq, align='edge', width=0.3)
  # plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right')
  fig.autofmt_xdate()

def correct_item_freq(ground_truth_item_freq, predict_item_freq):
  correct_freq_dict = dict()
  for i in predict_item_freq.keys():
    correct_freq_dict[i] = ground_truth_item_freq[i] if i in ground_truth_item_freq.keys() else 0

  plot_item_freq(correct_freq_dict)

