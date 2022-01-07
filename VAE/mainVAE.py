import math
import torch
from torch import optim
# from VAE import  VanillaVAE
from SparseAutoencoder import AutoEncoder
from ROI_dataset import ROI_dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
import time
import os
from config import args
import sys
from utils import logger
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def train(model, optimizer, loader, device):
    '''
    训练
    '''
    model.train()
    total_loss = 0
    # print(len(loader))
    for (i,data) in enumerate(loader):
        data = data.to(device)
        # data.to(torch.float32)
        #debug#
        # print(data['x'].size())
        optimizer.zero_grad()
        results = model(data)
        loss = model.loss_function(*results)
        total_loss += loss['loss']
        # torch.autograd.set_detect_anomaly(True)
        # with torch.autograd.detect_anomaly():
        loss['loss'].backward()
        #################################
        # if i%10 == 0:
        #     print("{}/{} loss: {}".format(i,len(loader),loss['loss']))
        optimizer.step()
    return total_loss / len(loader)


def train_allepoch(dataset,
                   epochs,
                   batch_size,
                   ROI_num,
                   lr,
                   lr_decay_factor,
                   lr_decay_step_size,
                   weight_decay,
                    logger=None,
                   model_PATH=None,
                   ):
    train_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    in_channel = dataset.get_subject_len()
    lateen_dim = 64
    model = AutoEncoder(in_channel,lateen_dim).to(device)
    # model = VanillaVAE(in_channel,lateen_dim).to(device)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)  # 优化器
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    min_loss = 100  # 当前的最小loss
    t_start = time.perf_counter()
    model_filename = os.path.join(model_PATH,'model_ROI_{}.pt'.format(ROI_num))
    if os.path.exists(model_filename) and args.continuetrain:  # 接着训练
        final_model_filename = os.path.join(model_PATH, 'final_model_ROI_{}.pt'.format(ROI_num))
        print('{} load successful'.format(final_model_filename))
        model.load_state_dict(torch.load(final_model_filename))
    for epoch in range(1, epochs + 1):
        torch.cuda.empty_cache()  # 清空显存
        train_loss= train(
            model, optimizer, train_loader, device)  # 训练一次
        eval_info = {
            'fold': 0,
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_loss,
            'val_loss': 0,
            'test_acc': 0,
            'val_acc': 0,
            'time_test_accs': 0,
            'time_val_accs': 0
        }  # 记录的日志数据
        if train_loss < min_loss:  # 保存最好的模型
            min_loss = train_loss
            print("save best model:{}".format(epoch))
            torch.save(model.state_dict(), model_filename)
        if logger is not None:  # 写入log
            logger(eval_info)
        if epoch % lr_decay_step_size == 0:  # 学习率衰减
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    final_model_filename = os.path.join(model_PATH, 'final_model_fold{}.pt'.format(ROI_num))
    torch.save(model.state_dict(), final_model_filename)
    t_end = time.perf_counter()  # 当前时间
    durations=t_end - t_start  # 运行时间

    #################输出##########
    print('Train loss: {:.4f}, Duration: {:.3f}'.
          format(train_loss, durations))
    sys.stdout.flush()
    ########################
    return min_loss

def run_train_exp():
    data_root = args.data_root
    model_path = args.model_dir
    subject_txt = args.subject_dir
    dataset = ROI_dataset(data_root,subject_txt)
    train_allepoch(dataset,
                   epochs=args.epochs,
                   batch_size=args.batch_size,
                   ROI_num=args.ROI_num,
                   lr=args.lr,
                   lr_decay_factor=args.lr_decay_factor,
                   lr_decay_step_size=args.lr_decay_step_size,
                   weight_decay=0.0005,
                   logger=logger,
                   model_PATH=model_path,
                   )

def run_test_exp():
    data_root = args.data_root
    output_filename = args.output_filename
    if not os.path.exists(output_filename):
        os.makedirs(output_filename)
    model_path = args.model_dir
    subject_txt = args.subject_dir
    dataset = ROI_dataset(data_root,subject_txt)
    #loadmodel:
    in_channel = dataset.get_subject_len()
    lateen_dim = 64
    model = AutoEncoder(in_channel, lateen_dim).to(device)
    # model = VanillaVAE(in_channel, lateen_dim).to(device)
    model_filename = os.path.join(model_path,'final_model_fold{}.pt'.format(args.ROI_num))
    model.load_state_dict(torch.load(model_filename))
    model.eval()
    data_list = dataset.data_list
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    #load_output_dir
    # for i in os.listdir(args.data_root):
    for i in test_loader:
        # data = np.load(os.path.join(args.data_root,i)).astype(np.float32)
        data = i['data'].to(device)
        items = i['items']
        # data = torch.tensor(data).to(device)
        # data = data.view(1,-1)
        # data = i.to(device)
        # mu, log_var = model.encode(data)
        # z = model.reparameterize(mu, log_var)
        z = model.encoder(data)
        z = z.cpu().detach().numpy()
        for k in range(z.shape[0]):
            zz = z[k]
            zz = np.reshape(zz,[1,-1])
            # print(zz.shape)
            npy_filename = data_list[items[k]]
            np.save(os.path.join(output_filename,npy_filename),zz)
            print("{} save success".format(os.path.join(output_filename,npy_filename)))
    #savetodir

def getonematrix(filename,data_root):
    '''
    获得一个时间点的矩阵 90*64（脑区数*特征向量长度）
    :return:
    '''
    roi_num = 90
    feature_matrix = []
    for i in range(roi_num):
        npy_filename = os.path.join(data_root,'ROI_{}'.format(i),filename) #加载的npy文件
        # print(npy_filename)
        z = np.load(npy_filename)
        feature_matrix.append(z)
    feature_matrix = np.array(feature_matrix)
    feature_matrix = np.reshape(feature_matrix,(roi_num,64))
    return feature_matrix


def trun2matrix():
    subject_list = args.subject_dir
    output_filename = args.output_filename
    data_root = args.data_root
    # output_filename = "/home/sharedata/gpuhome/rhb/fmri/no_mean_data/PostProcessing/Graph_test"
    # data_root = '/home/sharedata/gpuhome/rhb/fmri/AAL_output/'
    subject_txt = np.loadtxt(subject_list,dtype=str)
    ts = 130
    for i in subject_txt:
        save_foldname = os.path.join(output_filename,i)
        if not os.path.exists(save_foldname):
            os.makedirs(save_foldname)
        for j in range(ts):
            feature_matrix = getonematrix('{}_{}.npy'.format(i,j),data_root)
            savename = os.path.join(save_foldname,'S_index_{}'.format(j))
            np.save(savename,feature_matrix)
            print('{} save success'.format(savename,j))
if __name__ == '__main__':
    if args.trainortest == 'train':
        run_train_exp()
    elif args.trainortest == 'test':
        run_test_exp()
    elif args.trainortest == 'turn':
        trun2matrix()
