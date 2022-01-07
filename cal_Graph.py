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
parse.add_argument("--nii", type=str)
parse.add_argument('--output_filename', type=str)
parse.add_argument('--mask',type=str,default='/home/sharedata/gpuhome/rhb/fmri/data/Template/AAL_61x73x61_YCG.nii')
parse.add_argument('--brain_mask',type=str,default='/home/sharedata/gpuhome/rhb/fmri/data/Template/GreyMask_02_61x73x61.nii')
config = parse.parse_args()


# matlab文件名C:\Users\Administrator.XSGK6PRHVPCQKJD\Desktop\p值矩阵
# boold=u'C:/Users/Administrator.XSGK6PRHVPCQKJD/Desktop/p值矩阵/boold_volex_002_S_5018.mat'
# D=u'C:/Users/Administrator.XSGK6PRHVPCQKJD/Desktop/p值矩阵/diminsions_002_S_5018.mat'
num_nodes = 116

def zscore(volumn,mask,summask):
    #mean = np.sum(volumn*mask)/summask
    #std = np.sum((volumn-mean)**2 * mask)/summask
    #std = std**0.5
    #print(mean, std)
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
    #volumn = stats.zscore(volumn)  # 标准化
    #volumnmean = np.mean(volumn)
    #volumnstd = np.std(volumn)
    #print(volumnmean,volumnstd)
    #volumn = (volumn-volumnmean)/volumnstd
    volumn=zscore(volumn,brain_mask,summask)
    # 遍历每个体素，统计每个脑区的体素
    for i in range(volumn.shape[0]):
        for j in range(volumn.shape[1]):
            for k in range(volumn.shape[2]):

                roi = int(mask[i, j, k])
                if roi and roi <= num_nodes:
                    ROI[roi - 1].append(volumn[i, j, k])
    return ROI

def cal_Mahalanobis(a,b):
    c = a + b
    var = np.std(c)**2
    meana = np.mean(a)
    meanb = np.mean(b)
    distance = np.abs(meana-meanb)*np.sqrt(var**(-1))
    np.exp()
    return distance
def get_matrix(mask, volumn,P_filename, voxel_filename,brainmask,summask):
    start_time = time.time()
    Matrix = np.eye(116)
    ROI = cal_roi(volumn,mask,brainmask,summask)
    ROI_stat = np.zeros((num_nodes, 2))  # 返回值，每个脑区的特征
    for i in range(num_nodes):
        ROI_stat[i, 0] = np.mean(ROI[i])  # 平均值
        ROI_stat[i, 1] = np.std(ROI[i])  # 标准差
    # print(np.max(ROI_stat),np.min(ROI_stat))
    '''构建基于KS双样本检测的p值矩阵'''
    # for i in range(0, num_nodes):
    #     for j in range(0, i):
    #         Matrix[i, j] = Matrix[j, i] = ks_2samp(ROI[i], ROI[j])[1]

    '''马氏距离？'''
    for i in range(0,num_nodes):
        for j in range(0,i):
            Matrix[i,j] = Matrix[j,i] = cal_Mahalanobis(ROI[i],ROI[j])
    for i in range(num_nodes):
        Matrix[i,i] = 0
    Matrix = 1-(Matrix-np.mean(Matrix))/np.std(Matrix)
    #####################################################

    ###################################
    np.save(P_filename, Matrix)
    np.save(voxel_filename,ROI_stat)
    end_time = time.time()
    print('{} and {} save successful cost {}mins'.format(P_filename,voxel_filename, (end_time - start_time) / 60))



def main():

    pool = multiprocessing.Pool()
    start_time = time.time()
    nii = config.nii
    mask = config.mask
    print(mask)
    brainmask = config.brain_mask
    brainmask = nib.load(brainmask)
    brainmask = brainmask.get_fdata()
    brainmask[np.where(brainmask > 0)] = 1
    summask = np.sum(brainmask)
    nii = nib.load(nii) #加载nii图像
    nii = nii.get_fdata()
    nii = np.transpose(nii,[3,0,1,2])
    mask = nib.load(mask) #加载模板
    mask = mask.get_fdata()
    # volx1 = volx[t, :]  # 第一个时间点
    if not os.path.exists(config.output_filename):
        os.makedirs(config.output_filename)
    for i in range(nii.shape[0]):
         #get_matrix(mask,nii[i],os.path.join(config.output_filename, 'P_index_{}'.format(i)),
         #                    os.path.join(config.output_filename, 'V_index_{}'.format(i)),brainmask,summask)
         pool.apply_async(get_matrix, (mask,nii[i],
                                      os.path.join(config.output_filename, 'P_index_{}'.format(i)),
                                      os.path.join(config.output_filename, 'V_index_{}'.format(i)),
                                      brainmask,
                                      summask))

    pool.close()
    pool.join()
    end_time = time.time()
    print('运行时间 {}mins'.format((end_time - start_time) / 60))


main()
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















