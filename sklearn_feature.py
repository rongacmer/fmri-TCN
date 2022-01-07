import scipy.io as scio
import os
from config import args
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
from create_feature import TDP_featrue
import re
class FC_data(Dataset):

    def __init__(self,conf,subject_list, label, node = 90,ts = 130, transform=None):
        self.conf = conf
        self.subject_list = subject_list
        # self.subject_list = self.subjectlist2rid()
        self.transform = transform
        self.node = node
        self.label = label
        self.ts = ts
        self.data = "/home/gpusharedata/rhb/fmri/no_detla_data/PostProcessing/Graph/"
        self.tsdata = scio.loadmat('/home/sharedata/gpuhome/rhb/IPF_v1.0/data/ALLHCC_feature.mat') #超图实验
        self.tsdata = self.tsdata['timecourse']
        self.subject_csv_dict = self.loadsubjectcsv()
        self.matrix_data,self.true_label = self.getsubjectlist()


    def loadsubjectcsv(self):
        # subject_csv = np.loadtxt('/home/sharedata/gpuhome/rhb/fmri/subject/test.txt',dtype=str)
        subject_csv = np.loadtxt('/home/gpusharedata/rhb/fmri/no_detla_subject_bakeup/ALL_subject.list', dtype=str)
        subject_csv_dict = {}
        for i in range(len(subject_csv)):
            # subject_csv_dict[int(subject_csv[i])] = i
            subject_csv_dict[subject_csv[i]] = i
        # print(subject_csv_dict)
        return subject_csv_dict

    def subjectlist2rid(self):
        pattern = re.compile('S_(\d+)')
        subject_list = [int(pattern.findall(i)[0]) for i in self.subject_list]
        return subject_list

    def __len__(self):
        return len(self.matrix_data)
        # return 10
    def __getitem__(self, item):
        data = {'node':torch.ones((1,self.node)),
                'adj':torch.tensor(self.matrix_data[item]),
                'label':torch.tensor(self.true_label[item])}
        return data

    def get_ts_data(self,directory,time_length = 130):
        X = []
        num_node = 90
        for i in range(time_length):
            roi_stat_filename = os.path.join(directory, 'V_index_{}.npy'.format(i))
            roi_stat = np.load(roi_stat_filename)[:num_node]
            X.append(roi_stat[:,0])
        return np.array(X)


    def loadOnedata(self,filename,num_feature = None):
        '''
        加载数据
        :param filename:fc的文件名
        :return: data
        '''
        # data = scio.loadmat(filename)
        # data = data['ROICorrelation']
        # data = data[:self.node,:self.node]
        # print(data[0][0])
        feature_class = TDP_featrue(filename,self.node,ts = self.ts,num_feature=num_feature)
        data = feature_class.cal_TDP()
        return data

    def label2numlabel(self,label):
        numlabel = np.unique(label)
        label_dict = {}
        for index,x in enumerate(numlabel):
            label_dict[x] = index
        true_label = [label_dict[i] for i in label]
        return true_label

    def data2feature(self,data):
        # true_data = []
        # for i in range(len(data)):
        #     for j in range(i):
        #         true_data.append(data[i][j])
        # true_data = np.nan_to_num(true_data)
        # true_data = np.clip(true_data,-1,1)
        # return true_data
        #############12.20 保持矩阵的形式########
        true_data = np.nan_to_num(data)
        true_data = np.clip(true_data, -1, 1)
        return true_data

    def getsubjectlist(self):
        data = []
        subject_list = self.subject_list
        label = self.label
        true_label = self.label2numlabel(label)

        # for i,j in zip(subject_list,label):
        #     One_data = self.get_ts_data(os.path.join(self.data,i))
        #     One_data = self.loadOnedata(i,One_data)
        #     data.append(One_data)
        #     # print(One_data.shape)
        # return data,true_label


        for i in self.subject_list:
                One_data = self.tsdata[:,self.subject_csv_dict[i]]
                # One_data = self.loadOnedata(i,self.tsdata[:,:,self.subject_csv_dict[i]])
                data.append(One_data)
        return data,true_label



if __name__ == '__main__':
    '''
    用于测试
    '''
    pass
    # print('test')
    # # conf = All_conf
    # dataset= FC_data(conf,['002_S_0295_0','002_S_0295_1'],['NC','NC'])
    # dataloder = DataLoader(dataset=dataset,batch_size=2,shuffle=True)
    # for i,data in enumerate(dataloder):
    #     print(data['adj'].size(),data['label'].size())
    # data,true_label = dataloader.getsubjectlist()
    # print(true_label)
    # print(data)
    # loadOnedata('../data/PostProcessing/eMCI/FC/ROICorrelation_FisherZ_053_S_2396_1.mat')