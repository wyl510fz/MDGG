import os
from warnings import simplefilter
import numpy as np
import torch
from train import train
from DataLoader import LoadMatData, generate_permutation
import random
import datetime
from args import parameter_parser

simplefilter(action='ignore', category=FutureWarning)


args = parameter_parser()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.enabled = True



adjs, features, labels, nfeats, num_view, num_class = LoadMatData(args.dataset, args.k, 'D:\博士\data\MV/')
idx_train, idx_test = generate_permutation(labels, args)
print("dataset loading finished, num_view: {}, num_feat: {}, num_class: {}".format(num_view, nfeats, num_class))

#acc, f1, loss = train(features, adjs, labels, idx_train, idx_test, num_view, nfeats, num_class, args, device)

acc_list = []
f1_list = []
for i in range(args.rep_num):
    print('rep_num:', i + 1)
    acc, f1, loss = train(features, adjs, labels, idx_train, idx_test, num_view, nfeats, num_class, args, device)

    acc_list.append(acc * 100)
    f1_list.append(f1 * 100)
acc_list = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in acc_list]
f1_list = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in f1_list]

print("Optimization Finished!")

print("accuracy_mean= {:.4f}".format(np.array(acc_list).mean()), "accuracy_std= {:.4f}".format(np.array(acc_list).std()))
print("f1_mean= {:.4f}".format(np.array(f1_list).mean()), "f1_std= {:.4f}".format(np.array(f1_list).std()))

isExists = os.path.exists(args.res_path)
if not isExists:
    os.makedirs(args.res_path)
with open(args.res_path + '/{}.txt'.format(args.dataset), 'a', encoding='utf-8') as f:
    f.write(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S') + '\n'
        'dataset:{} | rep_num：{} | Ratio：{} | forward_diff_layer{} | gcn_layers:{}'.format(
        args.dataset, args.rep_num, args.train_ratio, args.forward_diff_layer, args.gcn_layers) + '\n'
        'dropout:{} | epochs:{} | lr:{} | wd:{} | hidden:{}'.format(
        args.dropout, args.epoch, args.lr, args.weight_decay, args.nhid) + '\n'
        'ACC_mean:{:.4f} | ACC_std: {:.4f} | ACC_max:{:.4f}'.format(
        np.array(acc_list).mean(), np.array(acc_list).std(), np.array(acc_list).max()) + '\n'
        'F1_mean:{:.4f} | F1_std: {:.4f} | F1_max:{:.4f}'.format(
        np.array(f1_list).mean(), np.array(f1_list).std(), np.array(f1_list).max()) + '\n'
        '----------------------------------------------------------------------------' + '\n')
