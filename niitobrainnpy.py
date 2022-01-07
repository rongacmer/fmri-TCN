import nibabel as nib
import numpy as np
import os
import multiprocessing
import argparse
import time
num_nodes = 90
parse = argparse.ArgumentParser()
parse.add_argument("--nii", type=str,default='/home/sharedata/gpuhome/rhb/fmri/no_mean_data/PostProcessing/ARWDCFS/002_S_0295_0.nii')
parse.add_argument("--name",type=str,default='002_S_0295_0')
parse.add_argument('--output_filename', type=str,default='/home/sharedata/gpuhome/rhb/fmri/no_mean_data/PostProcessing/AAL')
parse.add_argument('--mask',type=str,default='/home/sharedata/gpuhome/rhb/fmri/data/Template/AAL_61x73x61_YCG.nii')
parse.add_argument('--brain_mask',type=str,default='/home/sharedata/gpuhome/rhb/fmri/data/Template/GreyMask_02_61x73x61.nii')
config = parse.parse_args()

def zscore(volumn,mask,summask):
    #mean = np.sum(volumn*mask)/summask
    #std = np.sum((volumn-mean)**2 * mask)/summask
    #std = std**0.5
    #print(mean, std)
    # mean = np.mean(volumn)
    # std = np.std(volumn)
    # volumn = ((volumn - mean) / std)
    return volumn


def cal_roi(volumn, mask,brain_mask,summask):
    '''
    计算每个时间点的特征
    :param volumn: 每个时间点的图像
    :param mask:  模板
    :return: size【90，2】 90个脑区，每个脑区有平均值和标准差两个特征
    '''
    ROI = [[] for i in range(num_nodes)]  # 统计每个脑区的体素值
    volumn=zscore(volumn,brain_mask,summask)
    # 遍历每个体素，统计每个脑区的体素
    for i in range(volumn.shape[0]):
        for j in range(volumn.shape[1]):
            for k in range(volumn.shape[2]):
                roi = int(mask[i, j, k])
                if roi and roi <= num_nodes:
                    ROI[roi - 1].append(volumn[i, j, k])
    return ROI
def createROIdir(savedir):
    '''
    创建保存ROI向量的路径，如AAL/ROI_0,AAL/ROI_1
    :param savedir:
    :return:
    '''
    for i in range(num_nodes):
        if not os.path.exists(os.path.join(savedir,'ROI_{}'.format(i))):
            os.makedirs(os.path.join(savedir,'ROI_{}'.format(i)))

def saveROI(ROI,subject_name,ts,savedir):
    '''
    将各个脑区的BOLD信号保存为npy文件
    :param ROI: ROI矩阵
    :param subject_name:
    :param ts: 时间点
    :param savedir: 保存路径
    :return:
    '''
    for i in range(num_nodes):
        np.save(os.path.join(savedir,'ROI_{}'.format(i),'{}_{}'.format(subject_name,ts)),ROI[i])

def SaveOneVolumn(mask,volumn,subject_name,ts,savedir,brain_mask,summask):
    start_time = time.time()
    ROI = cal_roi(volumn,mask,brain_mask,summask)
    saveROI(ROI,subject_name,ts,savedir)
    end_time = time.time()
    print('{}_{} save successful cost {}mins'.format(subject_name,ts, (end_time - start_time) / 60))


def main():

    pool = multiprocessing.Pool()
    start_time = time.time()
    nii = config.nii
    mask = config.mask
    subject_name = config.name
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
    # createROIdir(config.output_filename)
    for i in range(nii.shape[0]):
        # SaveOneVolumn(mask,nii[i],subject_name,i,config.output_filename,brainmask,summask)
        pool.apply_async(SaveOneVolumn, (mask, nii[i],
                                      subject_name,
                                      i,config.outpt_filename,
                                      brainmask,
                                      summask))
    pool.close()
    pool.join()
    end_time = time.time()
    print('运行时间 {}mins'.format((end_time - start_time) / 60))




if __name__ == '__main__':
    main()