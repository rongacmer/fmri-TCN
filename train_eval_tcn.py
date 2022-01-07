import sys
import time

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold,train_test_split
from torch.utils.data import DataLoader

from utils import print_weights
import os
import numpy as np
from TCN import TemporalConvNet
from net_fmri import ResGCN
import HyperGCN
from config import args
from ZGCNETs import NNLSGCRNCNN
from torch.nn import LSTM
from torch.nn import GRU
torch.backends.cudnn.enabled = True

torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from hypermodel import hypermodel
from net.STGCN import  Model
from SAGnetwork import Net
from MCGRU import MCGRU

import Criterion
from sklearn.metrics import roc_auc_score
if 'GCN' in args.model:
    print('combine_network')
    from combine_network import GCN_TCN
elif 'MLP' in args.model:
    print('MLP_combine')
    from MLP_combine import GCN_TCN



def get_STMODEL(time_length):
    net = Model(time_length,2, 2, None, True, 90).to(device)
    return net

def get_HYPERMODEL():
    GCNnet = HyperGCN.ResGCN(hidden=128,edge_norm=False,num_feature=130,num_class=2,dropout=0.2)
    net = hypermodel(GCNnet)
    return net
def get_GCN_TCN_model(hidden = 64,time_length = 137,GCN_input = 64,TCN_input = 64):
    '''
    get GCN-TCN模型
    :param hidden: GCN维度长度
    :param time_length: 输入序列长度
    :param GCN_input: 缺省值
    :param TCN_input: TCN通道数
    :return:
    '''
    # GCN = ResGCN(hidden)
    GCN = Net(hidden,num_features=2)
    # TCN = GRU(TCN_input,hidden_size=64,num_layers=1,batch_first=True)
    TCN = TemporalConvNet(TCN_input,[64,32,16,8,1],ts=time_length)
    net = GCN_TCN(time_length = time_length,GCN_net = GCN,TCN_net = TCN,GCN_output = hidden, TCN_input = TCN_input)
    return net

def get_MCGRU(time_length):
    net = MCGRU(ts_length = time_length,roi_cnt=90)
    return net

def get_zig_zag_model(num_node = 90,time_length = 12):
    net = NNLSGCRNCNN(num_nodes = num_node,input_dim=2,rnn_units = 64, output_dim = 1, num_layers = 2, default_graph= True, embed_dim=10, window_len= time_length, link_len= 2)
    return net
def cross_validation_with_test(dataset,

                                  folds,
                                  epochs,
                                  batch_size,
                                  lr,
                                  lr_decay_factor,
                                  lr_decay_step_size,
                                  weight_decay,
                                  epoch_select,
                                  with_eval_mode=True,
                                  logger=None, model_PATH=None, semi_split = None, fold_list = None,time_length = 137,test_start = 0):

    assert epoch_select in ['val_max', 'test_max'], epoch_select

    val_losses, train_accs, test_accs, durations = [], [], [], []
    time_test_accs, time_val_accs, val_accs = [], [], []
    if fold_list is None:
        fold_list = [str(i) for i in range(folds)]
    else:
        fold_list = fold_list.split(',')
    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*k_fold(dataset, folds, epoch_select, semi_split, 'ADNI'))):
        idx = torch.cat([train_idx, val_idx])
        # idx = train_idx + val_idx
        dataset = dataset[idx]
        break

    ALL_cri = []
    ALL_label = []
    ALL_prob = []

    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*k_fold(dataset, folds, epoch_select, semi_split, 'ADNI'))):
        if str(fold) not in fold_list:
            continue
        # val_idx,test_idx = test_idx,val_idx
        test_dataset = dataset[test_idx]  # 测试集
        test_dataset.status = 'test'
        test_dataset.set_test_start(test_start)
        val_dataset = dataset[val_idx]  # 验证集
        val_dataset.status = 'test'
        val_dataset.set_test_start(test_start)

        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model = select_model(time_length)
        model = model.to(device)
        # model = model_func(dataset).to(device)
        # if os.path.exists(model_PATH):
        #     print('{} load successful'.format(model_PATH))
        #     model.load_state_dict(torch.load(model_PATH))

        # if fold == 0:
        #     print_weights(model)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        min_loss = 100
        model_filename = os.path.join(model_PATH, 'model_fold{}.pt'.format(fold))
        # model_filename = os.path.join(model_PATH, 'final_model_fold{}.pt'.format(fold))
        print('{} load success'.format(model_filename))
        model.load_state_dict(torch.load(model_filename))

        for epoch in range(1):

            torch.cuda.empty_cache()

            # val_losses.append(eval_loss(
            #     model, val_loader, device, with_eval_mode))  # 验证集损失
            test_cri, label,prob = eval_cri(model, test_loader, device, with_eval_mode,
                                               ts=time_length)  # 每一个样本的准确率，每个时间点
            ALL_label.append(label)
            ALL_prob.append(prob)
            print(test_cri)
            ALL_cri.append(test_cri.Cri)
            test_acc, time_test_acc = eval_acc(model, test_loader, device, with_eval_mode, ts=time_length)  # 每一个样本的准确率，每个时间点
            test_accs.append(test_acc)  # 测试集准确率
            time_test_accs.append(time_test_acc)
            val_acc, time_val_acc = eval_acc(model, val_loader, device, with_eval_mode, ts = time_length)
            val_accs.append(val_acc)
            # val_losses.append(-1)
            val_losses.append(eval_loss(
                model, val_loader, device, with_eval_mode))
            time_val_accs.append(time_val_acc)
            train_loss = 1
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_loss,
                'val_loss': val_losses[-1],
                'test_acc': test_accs[-1],
                'val_acc': val_accs[-1],
                'time_test_accs': time_test_accs[-1],
                'time_val_accs': time_val_accs[-1]
            }  # 记录的日志数据

            if logger is not None:
                logger(eval_info)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    duration = tensor(durations)
    test_acc, val_acc = tensor(test_accs), tensor(val_accs)
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    duration_mean = duration.mean().item()

    print(test_acc)
    print(val_acc)
    print(' Test Acc: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(test_acc_mean, test_acc_std, duration_mean))
    print('mean:\n')
    print(np.mean(ALL_cri,axis=0))
    print('std:\n')
    print(np.std(ALL_cri,axis=0))
    sys.stdout.flush()
    train_acc_mean = 1.0
    return train_acc_mean, test_acc_mean, test_acc_std, duration_mean,ALL_label,ALL_prob


def select_model(time_length):
    if 'GCN' in args.model:
        model = get_GCN_TCN_model(time_length=time_length)
    elif 'MLP' in args.model:
        model = get_GCN_TCN_model(time_length=time_length)
    elif 'ZIG' in args.model:
        model = get_zig_zag_model(time_length=time_length)
    elif 'ST' in args.model:
        model = get_STMODEL(time_length=time_length)
    elif 'MCG' in args.model:
        model = get_MCGRU(time_length=time_length)
    elif 'HYPER' in args.model:
        model = get_HYPERMODEL()

    return model
def cross_validation_with_val_set(dataset,
                                  folds,
                                  epochs,
                                  batch_size,
                                  lr,
                                  lr_decay_factor,
                                  lr_decay_step_size,
                                  weight_decay,
                                  epoch_select,
                                  with_eval_mode=True,
                                  logger=None, model_PATH=None, semi_split = None,fold_list = None,time_length = 137):
    '''

    :param dataset:数据集
    :param model_func:
    :param folds: k-folds
    :param epochs: 迭代次数
    :param batch_size:
    :param lr: 学习率
    :param lr_decay_factor: 学习率衰减率
    :param lr_decay_step_size: 学习率衰减步数
    :param weight_decay: Adam参数
    :param epoch_select: 'val_max':根据验证集的损失保存模型，‘test_max'保存在测试集最好的模型
    :param with_eval_mode:
    :param logger:log函数
    :param model_PATH:模型路径文件夹
    :param fold_list:训练的折数，example:0,1,2,3
    :return:train_acc:训练集准确率 test_acc:测试集准确率 test_std:测试集标准差 duration:训练时长
    '''
    assert epoch_select in ['val_max', 'test_max'], epoch_select

    val_losses, train_accs, test_accs, durations = [], [], [], []
    time_test_accs, time_val_accs, val_accs = [], [], []
    #如果fold_list为None，则训练所有的折，否则，只训练特定的折
    if fold_list is None:
        fold_list = [str(i) for i in range(folds)]
    else:
        fold_list = fold_list.split(',')

    # print(len(dataset))

    # for i in dataset:
    #     print(i['id'][0].item())
    # for fold, (train_idx, test_idx, val_idx) in enumerate(
    #         zip(*k_fold(dataset, folds, epoch_select, semi_split, 'ADNI'))):
    #     idx = torch.cat([train_idx, val_idx])
    #     # idx = train_idx + val_idx
    #     dataset = dataset[idx]
    #     break
    # for fold, (train_idx, test_idx, val_idx) in enumerate(
    #         zip(*k_fold(dataset, folds, epoch_select, semi_split, 'ADNI'))):
    #
    #     # dataset = dataset[train_idx]
    #     # break
    #     if fold == 2:
    #         idx = torch.cat([train_idx,val_idx])
    #         # idx = train_idx + val_idx
    #         dataset = dataset[idx]
    #         break


    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*k_fold(dataset, folds, epoch_select, semi_split, 'ADNI'))):
        #如果当前折不在需要训练的折中，则跳过
        if str(fold) not in fold_list:
            continue
        print("FOLD:{}".format(fold))
        # val_idx,test_idx = test_idx,val_idx
        train_dataset = dataset[train_idx] #训练集
        train_dataset.status = 'train' #设置数据集状态
        test_dataset = dataset[test_idx]  #测试集
        test_dataset.status = 'test'
        val_dataset = dataset[val_idx] #验证集
        val_dataset.status = 'test'

        #数据集迭代器
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model = select_model(time_length)
        model = model.to(device)
        # model = model_func(dataset).to(device)


        # if fold == 0:
        #     print_weights(model)
        # for value in model.GCN.parameters():
        #     value.requires_grad = False
        # for value in model.fn.parameters():
        #     value.requires_grad = False
        # model.fn.requires_grad = False
        # model.GCN.requires_grad = False
        optimizer = Adam(filter(lambda p:p.requires_grad,model.parameters()), lr=lr, weight_decay=weight_decay) #优化器

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        min_loss = 100 #当前的最小loss
        max_acc = 0 #当前最大的acc
        model_filename = os.path.join(model_PATH, 'model_fold{}.pt'.format(fold)) #model的保存路径，如第0折的路径为model_PATH/model_fold0.pt
        if os.path.exists(model_filename) and args.continuetrain:  # 接着训练
            # final_model_filename = os.path.join(model_PATH, 'final_model_fold{}.pt'.format(fold))
            print('{} load successful'.format(model_filename))
            model.load_state_dict(torch.load(model_filename))


        pre_train_epoch = 0
        for epoch in range(1,pre_train_epoch):
            train_loss, train_acc = train(
                model, optimizer, train_loader, device, ts=time_length)  # 训练一次
            print('#{} pre_train:{}'.format(epoch,train_acc))
        for epoch in range(1, epochs + 1):

            torch.cuda.empty_cache() #清空显存
            # test_acc, time_test_acc = eval_acc(model, test_loader, device, with_eval_mode, ts=time_length)
            # val_acc, time_val_acc = eval_acc(model, val_loader, device, with_eval_mode, ts=time_length)
            train_loss, train_acc = train(
                model, optimizer, train_loader, device,ts=time_length) #训练一次
            train_accs.append(train_acc) #加入当前代的训练准确率

            val_losses.append(eval_loss(
                model, val_loader, device, with_eval_mode)) #验证集损失
            test_acc, time_test_acc = eval_acc(model, test_loader, device, with_eval_mode, ts=time_length) #每一个样本的准确率，每个时间点
            test_accs.append(test_acc) #测试集准确率
            time_test_accs.append(time_test_acc)
            val_acc, time_val_acc = eval_acc(model, val_loader, device, with_eval_mode, ts=time_length)
            val_accs.append(val_acc)
            time_val_accs.append(time_val_acc)
            # val_accs.append(eval_acc(model,val_loader,device,with_eval_mode)) #验证集准确率

            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_accs[-1],
                'val_loss': val_losses[-1],
                'test_acc': test_accs[-1],
                'val_acc': val_accs[-1],
                'time_test_accs': time_test_accs[-1],
                'time_val_accs' : time_val_accs[-1]
            } #记录的日志数据

            if val_accs[-1] > max_acc: #保存最好的模型
                max_acc = val_accs[-1]
                min_loss = val_losses[-1]
                print("save best model:{}".format(epoch))
                torch.save(model.state_dict(),model_filename)
            if logger is not None: #写入log
                logger(eval_info)

            if epoch % lr_decay_step_size == 0: #学习率衰减
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']



        if torch.cuda.is_available():
            torch.cuda.synchronize()
        final_model_filename =  os.path.join(model_PATH, 'final_model_fold{}.pt'.format(fold))
        torch.save(model.state_dict(),final_model_filename)
        t_end = time.perf_counter() #当前时间
        durations.append(t_end - t_start) #运行时间

    #list 转为 tensor#
    duration = tensor(durations)
    train_acc, test_acc, val_acc = tensor(train_accs), tensor(test_accs), tensor(val_accs)
    val_loss = tensor(val_losses)
    ####################
    true_folds = len(fold_list) #运行的折数
    ######reshape 为[折数，迭代次数]
    train_acc = train_acc.view(true_folds, epochs)
    test_acc = test_acc.view(true_folds, epochs)
    val_loss = val_loss.view(true_folds, epochs)
    val_acc = val_acc.view(true_folds,epoch)
    ######################################
    if epoch_select == 'test_max':  # take epoch that yields best test results.
        _, selected_epoch = test_acc.mean(dim=0).max(dim=0)
        selected_epoch = selected_epoch.repeat(true_folds)
    else:  # take epoch that yields min val loss for each fold individually.
        # _, selected_epoch = val_loss.min(dim=1)
        _,selected_epoch = val_acc.max(dim=1)

    ##########挑选出val_loss最小时的test_acc和val_acc##################################
    test_acc = test_acc[torch.arange(true_folds, dtype=torch.long), selected_epoch]
    val_acc = val_acc[torch.arange(true_folds, dtype=torch.long), selected_epoch]
    ######################最终的训练准确率#################
    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item() #平均准确率
    test_acc_std = test_acc.std().item() #准确率标准差
    duration_mean = duration.mean().item() #平均消耗时间

    #################输出##########
    print(test_acc)
    print(val_acc)
    print('Train Acc: {:.4f}, Test Acc: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(train_acc_mean, test_acc_mean, test_acc_std, duration_mean))
    sys.stdout.flush()
    ########################
    return train_acc_mean, test_acc_mean, test_acc_std, duration_mean


def get_indicate(dataset,subject_list):
    '''
    得到在subject_list中id的样本
    :param dataset:
    :param subject_list:
    :return:
    '''
    indicate = []
    for cnt,sample in enumerate(dataset):
        if sample['id'][0].item() in subject_list:
            indicate.append(cnt)
    return indicate

def leave_out(dataset,folds,epoch_select,semi_split,name = None):
    '''
    留一法
    :param dataset:
    :param folds:
    :param epoch_select:
    :param semi_split:
    :param name:
    :return:
    '''
    dataset.set_state_check()#dataset改为check模式，该模式下，数据只返回标签和id
    test_indices, train_indices = [], []  # id下标_list
    subject = {}  # subject字典，记录每个subject的类别
    subject_id = []  # subject
    label = []  # 标签
    for cnt, sample in enumerate(dataset):  # 制作字典
        # print(cnt,sample['id'][0])
        if sample['id'][0].item() not in subject:
            subject[sample['id'][0].item()] = sample['y'][0].item()
            subject_id.append(sample['id'][0].item())
            label.append(sample['y'][0].item())
    subject_id = np.array(subject_id)
    # print(subject_id)
    label = np.array(label)
    val_indices = []
    for cnt,id in enumerate(subject_id):
        test_indices.append(torch.tensor(get_indicate(dataset, [id])))  # 根据样本划分
        train_mask = torch.ones(len(subject_id), dtype=torch.uint8)
        train_mask[cnt] = 0
        idx_train = train_mask.nonzero().view(-1)
        train_label = label[idx_train]
        # train_subject = subject_id[idx_train]

        train_idx,val_idx,train_label,val_label = train_test_split(idx_train,train_label,test_size=0.1,stratify=train_label,random_state=12345)
        train_indices.append(torch.tensor(get_indicate(dataset, subject_id[train_idx])))
        val_indices.append(torch.tensor(get_indicate(dataset, subject_id[val_idx])))
    dataset.set_state_train()
    return train_indices,  test_indices,val_indices


    # print(label)
def k_fold(dataset, folds, epoch_select, semi_split,name = None):
    '''
    多折交叉验证
    :param dataset:
    :param folds:
    :param epoch_select:
    :param semi_split:
    :param name:
    :return:
    '''
    dataset.set_state_check() #dataset改为check模式，该模式下，数据只返回标签和id
    ######AD_NC 12345
    ######eMCI_NC 1515
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345) #k-fold
    test_indices, train_indices = [], [] #id下标_list

    if 'ADNI' in name:
        test_subject, train_subject = [], [] #
        subject = {} #subject字典，记录每个subject的类别
        subject_id = [] #subject
        label = [] #标签
        for cnt,sample in enumerate(dataset): #制作字典
            # print(cnt,sample['id'][0])
            if sample['id'][0].item() not in subject:
                subject[sample['id'][0].item()] = sample['y'][0].item()
                subject_id.append(sample['id'][0].item())
                label.append(sample['y'][0].item())
        subject_id = np.array(subject_id)
        # print(subject_id)
        label = np.array(label)
        # print(label)
        for _,idx in skf.split(subject_id, label):
            print(label[idx])   #根据subject_id进行划分
            test_subject.append(subject_id[idx]) #按人划分
            test_indices.append(torch.tensor(get_indicate(dataset,subject_id[idx]))) #根据样本划分
            print(test_indices[-1])
    else: #没啥用了
        for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
            test_indices.append(torch.from_numpy(idx))

    if epoch_select == 'test_max':
        val_indices = [test_indices[i] for i in range(folds)]
    else:
        val_indices = [test_indices[i - 1] for i in range(folds)] #划分验证集

    # skf_semi = StratifiedKFold(semi_split, shuffle=True, random_state=100) #用于半监督
    #semi 半监督？
    for i in range(folds):
        #去掉验证集和测试集，剩下的就是训练集
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i].long()] = 0
        train_mask[val_indices[i].long()] = 0
        idx_train = train_mask.nonzero().view(-1)

        # for _, idx in skf_semi.split(torch.zeros(idx_train.size()[0]), dataset.data.y[idx_train]):
        #     idx_train = idx_train[idx]
        #     break
        ################平衡数据集##################
        # cat_train_data = [dataset[i.item()]['y'][0].item() for i in idx_train]
        # sum_pos = np.sum(cat_train_data)
        # pos_cnt = min(sum_pos,len(idx_train)-sum_pos)
        # pos = 0
        # neg = 0
        # final_idx_train = []
        # for i in range(len(idx_train)):
        #     if dataset[i]['y'][0].item() == 0 and pos < pos_cnt:
        #         final_idx_train.append(idx_train[i])
        #         pos += 1
        #     elif dataset[i]['y'][0].item() == 0 and neg < pos_cnt:
        #         final_idx_train.append(idx_train[i])
        #         neg += 1
        ################################################################
        train_indices.append(idx_train)
        # train_indices.append(final_idx_train)
    dataset.set_state_train()
    return train_indices, test_indices,val_indices


def num_graphs(data):
    # if data.batch is not None:
    #     return data.num_graphs
    # else:
    return data.size(0)


def train(model, optimizer, loader, device, ts = 130):
    '''
    训练
    '''
    model.train()
    node_cnt = 90
    total_loss = 0
    correct = 0
    L1_weight=0.005
    # print(len(loader))
    for data in loader:
        for j in data.keys():
            data[j] = data[j].to(device) #放入到gpu中
        #debug#
        # print(data['x'].size())
        optimizer.zero_grad()
        # x = data['x'].to(device)
        label = data['y']
        out,time_clf,node_clf,node_label = model(data) #final_clf,time_clf
        # print(out)
        # l1_penalty = L1_weight*sum([p.abs().sum() for p in model.parameters()])
        l1_penalty = 0
        loss = 0.5*F.nll_loss(out, label.view(-1)) + l1_penalty
        #############node_loss####################
        if args.node_clf:
            # node_label = label.view(-1,1)
            # node_label = node_label.expand(-1,ts*node_cnt)
            # node_label = node_label.reshape(-1)
            # print(node_clf.size())
            node_loss = 0.2*F.nll_loss(node_clf,node_label)
            # loss = node_loss
            # pred = out.max(1)[1]
            # correct += pred.eq(node_label.view(-1)).sum().item()
            node_loss.backward(retain_graph=True)
        ####time_loss#########
        ############time_label变为[B×ts]##########
        if args.time_clf:
            time_label = label.view(-1,1)
            time_label = time_label.expand(-1,ts)
            time_label = time_label.reshape(-1)
        ##########################################
            ts_loss = 0.3*F.nll_loss(time_clf,time_label)
            ts_loss.backward(retain_graph=True)
        if(torch.isnan(loss).sum()>0):
            # out, time_clf, node_clf, node_label = model(data)
            print(loss)
            print(data['id'])
        loss.backward()
        #################################
        pred = out.max(1)[1] #预测值
        correct += pred.eq(label.view(-1)).sum().item() #预测正确的个数
        total_loss += loss.item() * num_graphs(data['x']) #总的loss
        torch.nn.utils.clip_grad_norm(model.parameters(),max_norm=20,norm_type=2) #梯度截断
        optimizer.step()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def eval_cri(model, loader, device,with_eval_mode,ts = 130):
    '''
    计算模型各种分类性能
    :param model:
    :param loader:
    :param device:
    :param with_eval_mode:
    :param ts:
    :return:
    '''
    model.eval()
    ALLpredict = []
    ALLlabel = []
    ALLprob = []
    for data in loader:
        for j in data.keys():
            data[j] = data[j].to(device)
        label = data['y']
        with torch.no_grad():
            pred,time_pred,node_pred,node_label = model(data) #两种准确率
            preds = pred.max(1)[1] #最终结果分类
            final_prob = F.softmax(pred,dim=-1)
            # print('predict:',pred)
            # print('label:',label.view(-1))
            # pred = torch.rand(10)

        ALLpredict = ALLpredict + preds.tolist()
        ALLlabel = ALLlabel + label.view(-1).tolist()
        ALLprob = ALLprob + final_prob[:,1].tolist()
    cri = Criterion.criterion(ALLpredict,ALLlabel)
    cri.Cri.append(roc_auc_score(ALLlabel,ALLprob))
    return cri,ALLlabel,ALLprob


def eval_acc(model, loader, device, with_eval_mode,ts = 130):
    '''
    计算准确率
    :param model:
    :param loader:
    :param device:
    :param with_eval_mode:
    :return:
    '''
    model.eval()
    # print(len(loader.dataset))
    correct = 0
    time_correct = 0
    node_correct = 0
    node_cnt  = 90
    for data in loader:
        for j in data.keys():
            data[j] = data[j].to(device) #数据放入到gpu中
        label = data['y'] #标签
        ############time_label变为[B×ts]##########
        if args.time_clf:
            time_label = label.view(-1, 1)
            time_label = time_label.expand(-1, ts)
            time_label = time_label.reshape(-1)
        ##########################################
        with torch.no_grad():
            pred,time_pred,node_pred,node_label = model(data) #两种准确率
            pred = pred.max(1)[1] #最终结果分类
            # print('predict:',pred)
            # print('label:',label.view(-1))
            if args.time_clf:
                time_pred = time_pred.max(1)[1] #但时间点分类
            # print(pred)
        correct += pred.eq(label.view(-1)).sum().item() #正确的数量
        if args.node_clf:
            # node_label = label.view(-1, 1)
            # node_label = node_label.expand(-1, ts*node_cnt)
            # node_label = node_label.reshape(-1)
            node_pred = node_pred.max(1)[1]
            node_correct += node_pred.eq(node_label.view(-1)).sum().item()

        if args.time_clf:
            time_correct += time_pred.eq(time_label.view(-1)).sum().item()
    # print(correct)
    return correct / len(loader.dataset), time_correct / (ts*len(loader.dataset)) #准确率


def eval_loss(model, loader, device, with_eval_mode):
    model.eval()

    loss = 0
    node_cnt = 0
    for data in loader:
        for j in data.keys():
            data[j] = data[j].to(device) #数据放入到gpu中
        label = data['y']
        with torch.no_grad():
            out,_,_,_ = model(data) #输出
        loss += F.nll_loss(out, label.view(-1), reduction='sum').item() #total_loss

        # if args.node_clf:
        #     node_label = label.view(-1, 1)
        #     node_label = node_label.expand(-1, 90)
        #     node_label = node_label.reshape(-1)
        #     # loss += F.nll_loss(out,node_label.view(-1),reduction='sum').item()
        # # print(loss)
    return loss / len(loader.dataset) #平均损失
