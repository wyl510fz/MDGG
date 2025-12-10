import time
from copy import deepcopy
from DataLoader import sparse_mx_to_torch_sparse_tensor, normalize
import torch.nn.functional as F
import torch.optim as optim
from utils import accuracy, f1_test
from tqdm import tqdm
from Diff import diff
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp
from model import MvGCN
import torch
import scipy.io as sio


def train(features, adjs, labels, idx_train, idx_test, num_view, nfeats, num_class, args, device):
    for i in range(num_view):
        # exec("features_{}= torch.from_numpy(features[{}]/1.0).float().to(device)".format(i, i))
        exec("features[{}]= torch.Tensor(features[{}] / 1.0).to(device)".format(i, i))
        exec("features[{}] = F.normalize(features[{}])".format(i, i))
        exec("adjs[{}]=adjs[{}].to_dense().float().to(device)".format(i, i))

    features_diff = diff(adjs, features, args, device)

    adjs_diff = []
    for feat in features_diff:
        temp = kneighbors_graph(feat.cpu().numpy(), args.k)
        temp = sp.coo_matrix(temp)
        temp = temp + temp.T.multiply(temp.T > temp) - temp.multiply(temp.T > temp)
        temp = normalize(temp + sp.eye(temp.shape[0]))
        temp = sparse_mx_to_torch_sparse_tensor(temp)
        adjs_diff.append(temp)

    model = MvGCN(nfeats, num_class, nhid=args.nhid, dropout=args.dropout, layers=args.gcn_layers, batch_norm=1)
    total_para = sum(x.numel() for x in model.parameters())
    print("Total number of paramerters in networks is {}  ".format(total_para / 1e6))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for i in range(num_view):
        # exec("features_{}= torch.from_numpy(features[{}]/1.0).float().to(device)".format(i, i))
        exec("features_diff[{}]= torch.Tensor(features_diff[{}] / 1.0).to(device)".format(i, i))
        exec("features_diff[{}] = F.normalize(features_diff[{}])".format(i, i))
        exec("adjs_diff[{}]=adjs_diff[{}].to_dense().float().to(device)".format(i, i))

    if args.cuda:
        model.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()  # [:10]
        idx_test = idx_test.cuda()
    # f_loss = open('./results/loss_and_acc_curve/loss/' + args.dataset + '.txt', 'w')
    # f_ACC = open('./results/loss_and_acc_curve/ACC/' + args.dataset + '.txt', 'w')
    # f_F1 = open('./results/loss_and_acc_curve/F1/' + args.dataset + '.txt', 'w')
    t1 = time.time()

    best_acc_train = 0.0
    with tqdm(total=args.epoch) as pbar:
        pbar.set_description('Training:')

        for i in range(args.epoch):
            t = time.time()
            model.train()
            optimizer.zero_grad()

            output = model(features, adjs_diff)

            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            acc_train = accuracy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()


            if acc_train > best_acc_train:
                best_acc_train = acc_train
                weights = deepcopy(model.state_dict())

            # # loss曲线
            # isExists = os.path.exists("./results_linear/loss/{}".format(args.dataset))
            # if not isExists:
            #     os.mkdir("./results_linear/loss/{}".format(args.dataset))
            # with open("./results_linear/loss/{}".format(args.dataset) + '/loss.txt', 'a', encoding='utf-8') as f:
            #     f.write(str(loss_test.detach().cpu().numpy()) + '\n')
            # with open("./results_linear/loss/{}".format(args.dataset) + '/acc.txt', 'a', encoding='utf-8') as f:
            #     f.write(str(acc_test.detach().cpu().numpy()) + '\n')
            # with open("./results_linear/loss/{}".format(args.dataset) + '/f1.txt', 'a', encoding='utf-8') as f:
            #     f.write(str(f1) + '\n')

            outstr = 'Epoch: {:04d} '.format(i + 1) + \
                     'loss_train: {:.4f} '.format(loss_train.item()) + \
                     'acc_train: {:.4f} '.format(acc_train.item()) + \
                     'time: {:.4f}s'.format(time.time() - t)
            pbar.set_postfix_str(outstr)
            # f_loss.write(str(loss_train.item()) + '\n')
            # f_ACC.write(str(acc_test.item()) + '\n')
            # f_F1.write(str(f1.item()) + '\n')
            pbar.update(1)

    model.load_state_dict(weights)
    model.eval()
    output = model(features, adjs_diff)

    loss = F.nll_loss(output[idx_test], labels[idx_test])
    acc = accuracy(output[idx_test], labels[idx_test])
    f1 = f1_test(output[idx_test], labels[idx_test])
    sio.savemat("YMITIndoor.mat", {'labe':output[idx_test].cpu().detach().numpy()})
    sio.savemat("respNMITIndoor.mat", {'resp': labels[idx_test].cpu().detach().numpy()})
    #新增
    # draw_plt(output[idx_test], labels[idx_test], args.dataset)
    print('total_time:', time.time() - t1)

    del model

    return acc, f1, loss
