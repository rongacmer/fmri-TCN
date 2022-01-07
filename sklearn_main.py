# coding: utf-8
import classify_model
import pandas as pd
from config import args
import numpy as np
from sklearn.model_selection import StratifiedKFold,train_test_split
import re
from sklearn_feature import FC_data
import os
# def get_label():
#     cls = conf.cls.split('_') #AD_NC
#     label_dict = dict()
#     label_excel = pd.read_excel(conf.label_dir)
#     for i in range(len(label_excel)):
#         # print(label_excel.iloc[i]['name'])
#         label_dict[label_excel.iloc[i]['name']] = cls.index(label_excel.iloc[i]['group'])
#     # print(label_dict)
#     return label_dict
#     # print(len(label_excel))
#
#
# def get_data(select_index):
#     data_dict = dict()
#     data_excel = pd.read_excel(conf.data_excel)
#     for i in range(len(data_excel)):
#         data_dict[data_excel.iloc[i][0]] = np.array(data_excel.iloc[i][1:])
#         data_dict[data_excel.iloc[i][0]] = data_dict[data_excel.iloc[i][0]][select_index]
#     return data_dict
#
# def get_data_label(data_dict,label_dict):
#     data = []
#     label = []
#     for i in data_dict.keys():
#         if i in label_dict.keys():
#             data.append(data_dict[i])
#             label.append(label_dict[i])
#     # print(len(data))
#     # print(len(label))
#     return np.array(data),np.array(label)
#
#
def create_train_test(df_data,df_label,train_txt:np.array,test_txt,dev_data = None,dev_label = None):
    #########随便分，不按subject分###################
    # df_data = np.array(df_data)
    # df_label = np.array(df_label)
    # train_dataloder = FC_data(All_conf,df_data[train_txt],df_label[train_txt])
    # test_dataloader = FC_data(All_conf,df_data[test_txt],df_label[test_txt])
    # train_data,train_label = train_dataloder.getsubjectlist()
    # test_data,test_label = test_dataloader.getsubjectlist()

    #######################按照subject_id分类#######################################
    train_data,train_label,test_data,test_label = [],[],[],[]
    pattern = re.compile('\d+_S_\d+')
    for i,j in zip(df_data,df_label):
        id = pattern.findall(i)[0]
        if id in train_txt:
            train_data.append(i)
            train_label.append(j)
        else:
            test_data.append(i)
            test_label.append(j)
    ###########让数据集平衡##################
    clf = {'AD':0,'NC':1}
    # clf = {'MCI':0,'NC':1}
    cat_train_data = [clf[i] for i in train_label]
    sum_pos = np.sum(cat_train_data)
    pos_cnt = min(sum_pos,len(train_label)-sum_pos)
    pos = 0
    neg = 0
    final_train_data = []
    final_train_label = []
    for i in range(len(train_label)):
        if clf[train_label[i]] == 0 and pos < pos_cnt:
            final_train_data.append(train_data[i])
            final_train_label.append(train_label[i])
            pos += 1
        elif clf[train_label[i]] == 1 and neg < pos_cnt:
            final_train_data.append(train_data[i])
            final_train_label.append(train_label[i])
            neg += 1
    # final_train_data = np.array(final_train_data)
    # final_train_label = np.array(final_train_label)
    # train_dataloder = FC_data(args, train_data, train_label)
    train_dataloder = FC_data(args,final_train_data,final_train_label)
    test_dataloader = FC_data(args,test_data,test_label)
    train_data,train_label = train_dataloder.getsubjectlist()
    test_data,test_label = test_dataloader.getsubjectlist()

    # train_data,train_label,test_data,test_label  = \
    #     df_data[train_txt],np.ravel(df_label[train_txt]),df_data[test_txt],np.ravel(df_label[test_txt])
    return train_data,train_label,test_data,test_label
#
def get_subject_id(subject_list,label):
    '''
    返回 subject_id && label
    :param subject_list:
    :param label:
    :return:
    '''
    s = dict()
    pattern = re.compile('\d+_S_\d+')
    for i,j in zip(subject_list,label):
        x = pattern.findall(i)
        if len(x):
            if x[0] not in s:
                s[x[0]] = j
    subject_id = []
    subject_label = []
    for i in s.items():
        subject_id.append(i[0])
        subject_label.append(i[1])
    return np.array(subject_id),np.array(subject_label)
#
def run_one_cri(df_data,df_label,train_txt,test_txt,dev_data = None,dev_label = None):
    train_data,train_label,test_data,test_label = create_train_test(df_data,df_label,train_txt,test_txt,dev_data,dev_label)
    return classify_model.classify_model(train_data,train_label,test_data,test_label,dev_data,dev_label)

def cross_validation():

    ALL_cri = []

    subject = []
    label = []
    clf = args.clf.split('_')
    # subject_dir = args.subject_dir
    subject_dir = '/home/gpusharedata/rhb/fmri/no_detla_subject_bakeup/'
    # subject_dir = '/home/sharedata/gpuhome/rhb/fmri/no_detla_subject'
    # subject_dir = '/home/sharedata/gpuhome/rhb/fmri/papersubject'
    clf_subject = []
    for i in clf:
        One_subject = np.loadtxt(os.path.join(subject_dir,'{}.list'.format(i)),dtype=str)
        One_subject = One_subject.tolist()
        subject += One_subject
        clf_subject.append(One_subject)
        label += [i]*len(One_subject)

    # 随便分
    # cross = StratifiedKFold(n_splits=5, shuffle=True)
    # for train,test in cross.split(subject,label):
    #     # print(train,test)
    #     # print(test)
    #     one_cri = run_one_cri(subject,label,train,test)
    #     ALL_cri.append(one_cri)
    # print(np.mean(ALL_cri,axis=0))

    #按照id分
    subject_id, subject_label = get_subject_id(subject,label)
    print(len(subject_id),len(subject_label))
    subject_id = np.array(list(subject_id))
    # print(subject_id)
    cross = StratifiedKFold(n_splits=10, shuffle=True,random_state=10)
    for train, test in cross.split(subject_id, subject_label):
        one_cri = run_one_cri(subject, label, subject_id[train], subject_id[test])
        print(one_cri)
        ALL_cri.append(one_cri)
        # print(subject_id[train], subject_label[train])

    #hold-out
    # myrandom_seed = [123,321,458,156,546]
    # clf_subject[0], _ = get_subject_id(clf_subject[0], np.ones(len(clf_subject[0])))
    # clf_subject[1], _ = get_subject_id(clf_subject[1], np.ones(len(clf_subject[1])))
    # for i in range(5):
    #     train_x_1,test_x_1,_,_ = train_test_split(clf_subject[0],np.ones(len(clf_subject[0])),train_size=20,random_state=myrandom_seed[i],shuffle=True)
    #     train_x_2,test_x_2,_,_ = train_test_split(clf_subject[1],np.ones(len(clf_subject[1])),train_size=20,random_state=myrandom_seed[i],shuffle=True)
    #     train_x = train_x_1.tolist() + train_x_2.tolist()
    #     test_x = test_x_1.tolist() + test_x_2.tolist()
    #     one_cri = run_one_cri(subject, label, train_x, test_x)
    #     ALL_cri.append(one_cri)
    print(np.mean(ALL_cri, axis=0))
    print(np.std(ALL_cri,axis=0))
# get_data_label()

if __name__ == '__main__':
    print('test')
    cross_validation()