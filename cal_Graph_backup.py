# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 14:07:18 2020

@author: Administrator
"""
import os
import scipy.io as sio
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp
import argparse
import multiprocessing
import time
import nibabel as nib
from scipy import stats
parse = argparse.ArgumentParser()
parse.add_argument("--nii", type=str,default="/home/sharedata/gpuhome/rhb/fmri/paper_data_no_mean/PostProcessing/ARWSDCF/002_S_5018_0.nii")
parse.add_argument('--output_filename', type=str,default="")
parse.add_argument('--mask',type=str,default='/home/sharedata/gpuhome/rhb/fmri/data/Template/AAL_61x73x61_YCG.nii')
parse.add_argument('--brain_mask',type=str,default='/home/sharedata/gpuhome/rhb/fmri/data/Template/GreyMask_02_61x73x61.nii')
config = parse.parse_args()


# matlab文件名C:\Users\Administrator.XSGK6PRHVPCQKJD\Desktop\p值矩阵
# boold=u'C:/Users/Administrator.XSGK6PRHVPCQKJD/Desktop/p值矩阵/boold_volex_002_S_5018.mat'
# D=u'C:/Users/Administrator.XSGK6PRHVPCQKJD/Desktop/p值矩阵/diminsions_002_S_5018.mat'
num_nodes = 116

def zscore(volumn,mask,summask):
    # mean = np.sum(volumn)/summask
    # std = np.sum((volumn-mean)**2 * mask)/summask
    # std = std**0.5
    # # print(mean, std)
    mean = np.mean(volumn)
    std = np.std(volumn)
    volumn = ((volumn - mean) / std)
    return volumn

def cal_roi(volumn, mask,brain_mask,summask):
    '''
    计算每个时间点的特征
    :param volumn: 每个时间点的图像
    :param mask:  模板
    :return: size【90，2】 90个脑区，每个脑区有平均值和标准差两个特征
    '''
    ROI = [[] for i in range(num_nodes)]  # 统计每个脑区的体素值
    # volumn = stats.zscore(volumn)  # 标准化
    volumn = zscore(volumn,brain_mask,summask)
    # volumn = (volumn-np.mean(volumn))/np.std(volumn)
    # 遍历每个体素，统计每个脑区的体素
    for i in range(volumn.shape[0]):
        for j in range(volumn.shape[1]):
            for k in range(volumn.shape[2]):
                # print(i,j,k)
                roi = int(mask[i, j, k])
                if roi and roi <= num_nodes:
                    ROI[roi - 1].append(volumn[i, j, k])
    return ROI
def get_matrix(mask, volumn,P_filename, voxel_filename,brain_mask,summask):
    start_time = time.time()
    Matrix = np.eye(116)
    ROI = cal_roi(volumn,mask,brain_mask,summask)
    ROI_stat = np.zeros((num_nodes, 2))  # 返回值，每个脑区的特征
    print(ROI_stat.shape)
    for i in range(num_nodes):
        ROI_stat[i, 0] = np.mean(ROI[i])  # 平均值
        ROI_stat[i, 1] = np.std(ROI[i])  # 标准差
        # print(ROI_stat[i])
        # print(ROI_stat[i])
        # print(ROI_stat[i][0] >= 100)
        if np.isnan(ROI[i][0]):
            print(ROI_stat[i,0])
            print(ROI[i])
        # print("i,",ROI_stat.shape)
    # print("x:",ROI_stat.shape)
    # print(np.max(ROI_stat),np.min(ROI_stat))
    '''构建基于KS双样本检测的p值矩阵'''
    # for i in range(0, num_nodes):
    #     for j in range(0, i):
    #         Matrix[i, j] = Matrix[j, i] = ks_2samp(ROI[i], ROI[j])[1]
    # # np.save(P_filename, Matrix)
    # # np.save(voxel_filename,ROI_stat)
    # end_time = time.time()
    # print('{} and {} save successful cost {}mins'.format(P_filename,voxel_filename, (end_time - start_time) / 60))
    return ROI_stat



def main(nii):
    # pool = multiprocessing.Pool()
    start_time = time.time()
    # nii = config.nii
    # mask = config.mask
    # brainmask = config.brainmask
    # mask = nib.load(mask)  # 加载模板
    # mask = mask.get_fdata()
    #
    # brainmask = nib.load(brainmask)
    # brainmask = brainmask.get_fdata()
    # summask = sum(brainmask)


    nii = nib.load(nii) #加载nii图像
    nii = nii.get_fdata()
    nii = np.transpose(nii,[3,0,1,2])


    # volx1 = volx[t, :]  # 第一个时间点
    # if not os.path.exists(config.output_filename):
    #     os.makedirs(config.output_filename)
    subject_time_course = []
    for i in range(nii.shape[0]):
        print(i)
        ROI_stat = get_matrix(mask,nii[i],os.path.join(config.output_filename, 'P_index_{}'.format(i)),
                            os.path.join(config.output_filename, 'V_index_{}'.format(i)),
                              brainmask,summask)
        # print(ROI_stat[0])
        # print(ROI_stat[1])
        # print(ROI_stat.shape)
        subject_time_course.append(ROI_stat)
        # pool.apply_async(get_matrix, (mask,nii[i],
        #                               os.path.join(config.output_filename, 'P_index_{}'.format(i)),
        #                               os.path.join(config.output_filename, 'V_index_{}'.format(i)),))
    # pool.close()
    # pool.join()
    end_time = time.time()
    print('运行时间 {}mins'.format((end_time - start_time) / 60))
    print(np.array(subject_time_course).shape)
    return np.array(subject_time_course)

import re
if __name__ == '__main__':
    mask = config.mask
    brainmask = config.brain_mask
    mask = nib.load(mask)  # 加载模板
    mask = mask.get_fdata()

    brainmask = nib.load(brainmask)
    brainmask = brainmask.get_fdata()
    brainmask[np.where(brainmask>0)] = 1
    summask = np.sum(brainmask)
    time_course = []
    folds = "/home/sharedata/gpuhome/rhb/fmri/paper_data_no_mean/PostProcessing/ARWSDCF/"
    riddir = {}
    pattern = re.compile('S_(\d+)')
    for i in os.listdir(folds):
        rid = int(pattern.findall(i)[0])
        riddir[rid] = i
    print(riddir)
    subject_txt = '/home/sharedata/gpuhome/rhb/fmri/subject/test.txt'
    cnt = 0
    with open(subject_txt) as f:
        for i in f:
            i = i.strip()
            print(riddir[int(i)])
            ROI_stat = main(os.path.join(folds,riddir[int(i)]))
            # ROI_stat = main(os.path.join(folds, '129_S_4422_0.nii'))
            time_course.append(ROI_stat[:,:,0])
            # if cnt >= 3:
            #     break
            # cnt += 1
    time_course = np.array(time_course)
    # print(time_course.shape)
    time_course = time_course.transpose([1,2,0])
    print(time_course.shape)
    sio.savemat('/home/sharedata/gpuhome/rhb/IPF_v1.0/mytimecourse_AAL116.mat',{'timecourse':time_course})
    # main()
# pool = multiprocessing.Pool()
#
# p=ks_2samp(ROIBig,ROISmall)[1]
# np.savetxt('bold_p_value.txt', Matrix)
# Matrix = np.load('bold_p_value.npy')
# Matrix = Matrix > 0.
# ROI89 = volx1[0, D[89]:D[89] + D[90]];
# M = np.mean(ROI89)
# ROIBig = ROI89[ROI89 > M]
# ROISmall = ROI89[ROI89 <= M]
# f,ax = plt.subplots(figsize=(20,20))
# sns.heatmap(Matrix,annot=True,cmap='YlGnBu')
# plt.show()

from torch.distributed.nn import RemoteModule














