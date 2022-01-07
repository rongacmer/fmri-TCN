import os.path as osp
import re

import os

import torch
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.utils import degree
import torch_geometric.transforms as T
# from ADNI_fmri_dataset import fmriDataset
from zigzag_dataset import zigzagDataset
from ASD_fmri_dataset import fmriDataset
# from ADNI_fmri_dataset_updata import fmriDataset
from ADNI_fmri_dataset_hypergraph import fmriDataset as hyperDataset
#'/home/gpusharedata/rhb/fmri/output/AD_NC_no_P_network' 邻接矩阵为全1矩阵
#'/home/gpusharedata/rhb/fmri/output/AD_NC_has_P_network' 主对角线是0
#'/home/gpusharedata/rhb/fmri/no_mean_data/PostProcessing/Graph_MAesp'马氏距离
#'/home/sharedata/gpuhome/rhb/fmri/NYU_GRAPH' #NYU数据集
#Template = '/home/sharedata/gpuhome/rhb/fmri/data/Template/AAL_61x73x61_YCG.nii',
# '/home/gpusharedata/rhb/fmri/no_mean_data/PostProcessing/Graph_BN' 246模板
def get_dataset(name, root_dir = '/home/gpusharedata/rhb/fmri/output/AD_NC_no_P_network',
                niidatadir = '/home/sharedata/gpuhome/rhb/fmri/GRAPH',
                p_network='/home/gpusharedata/rhb/fmri/data/PostProcessing/P_network',
                subjectdir = '/home/gpusharedata/rhb/fmri/no_detla_subject/',
                Template = '/home/gpusharedata/rhb/fmri/no_detla_data/PostProcessing/MASKs/',
                hyperdir = '/home/gpusharedata/rhb/fmri/no_mean_data/PostProcessing/hypergraph',
                clf = 'AD_NC',
                roi_cnt = 90,
                time_length = 115,
                time_slice = 115,
                threshold = 100):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    clf = clf.split('_')
    print('time_slice:{}'.format(time_slice))

    if 'fmri' in name:
        # root_dir = '/home/rhb/fmri/output/P_network_backup' #AD_NC
        # # root_dir = '/home/rhb/fmri/output/P_network_eMCI_NC' #eMCI_NC/
        # niidatadir = '/home/rhb/fmri/data/PostProcessing/ARFWS'
        # p_network = '/home/rhb/fmri/data/PostProcessing/P_network'
        # subjectdir = '/home/rhb/fmri/subject/'
        # Template = '/home/sharedata/rhb/fMRI/data/Template/AAL_61x73x61_YCG.nii'
        # if not os.path.exists(root_dir):
        #     os.makedirs(root_dir)
        # clf = clf.split('_')
        # print('time_slice:{}'.format(time_slice))
        # dataset = fmriDataset(root_dir, roi_cnt, niidatadir,
        #                       time_length, clf, Template,
        #                       subjectdir, p_network,time_slice = time_slice,
        #                       test_start = 0,
        #                       threshold = threshold)
        # else:
        #     path = '/home/rhb/fmri/output/P_network'
        #     niidatadir = '/home/rhb/fmri/data/PostProcessing/ARFWS'
        #     p_network = '/home/rhb/fmri/data/PostProcessing/P_network'
        #     subjectdir = '/home/rhb/fmri/subject/'
        #     Template = '/home/sharedata/rhb/fMRI/data/Template/AAL_61x73x61_YCG.nii'
        #     clf = ['AD', 'NC']
        #     dataset = ADNIDataset(path, 90, True, niidatadir, 137, clf, Template, subjectdir, p_network)
        dataset = fmriDataset(root_dir, roi_cnt, niidatadir, time_length, clf, time_slice=time_slice, status='train',
                                threshold=threshold, subjectdir=subjectdir)
    if 'zig' in name:
        dataset = zigzagDataset(root_dir,roi_cnt,niidatadir,time_length,clf,time_slice=time_slice,status='train',threshold = threshold,subjectdir=subjectdir)

    if 'hyper' in name:
        dataset = hyperDataset(root_dir, roi_cnt, niidatadir, hyperdir, time_length, clf, time_slice=time_slice, status='train',
                                threshold=threshold, subjectdir=subjectdir)
    return dataset
