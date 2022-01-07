import numpy as np
import os
import scipy.io as sio
# Graph_directory = '/home/gpusharedata/rhb/fmri/no_mean_data/PostProcessing/Graph'
# timecourse = []
# with open('/home/gpusharedata/rhb/fmri/no_detla_subject_bakeup/ALL_subject.list','r') as f:
#     for i in f:
#         i = i[:-1]
#         print(i)
#         mysubjecttimecourse = []
#         for j in range(130):
#             V_index = os.path.join(Graph_directory,i,'V_index_{}.npy'.format(j))
#             timefeature = np.load(V_index)
#             mysubjecttimecourse.append(timefeature[:,0])
#         timecourse.append(mysubjecttimecourse)
# timecourse = np.array(timecourse)
# timecourse = np.transpose(timecourse,[1,2,0])
# sio.savemat('/home/sharedata/gpuhome/rhb/IPF_v1.0/ALLtimecourse.mat', {'timecourse': timecourse})
# print(timecourse.shape)

import pandas as pd

subject_path = '/home/gpusharedata/rhb/fmri/no_detla_subject_bakeup'
subject_label = {}
clf = {'AD':1,'eMCI':2,'NC':3}
for i in clf:
    with open(os.path.join(subject_path,'{}.list'.format(i))) as f:
        for j in f:
            j = j[:-1]
            subject_label[j] = clf[i]


subject = np.loadtxt('/home/gpusharedata/rhb/fmri/no_detla_subject_bakeup/ALL_subject.list',dtype=str)
group = [subject_label[i] for i in subject]
group = np.array(group)
# group = group.reshape((-1,1))
# sio.savemat('/home/sharedata/gpuhome/rhb/IPF_v1.0/ALLsubject.mat',{'subjects':group})
data = {"subject":subject,'group':group}
df = pd.DataFrame(data)
df.to_csv('/home/sharedata/gpuhome/rhb/IPF_v1.0/ALLsubject.csv',index=None,encoding='utf-8')
# print(df)



