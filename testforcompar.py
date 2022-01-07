import numpy as np

import os
import re
# graph = np.loadtxt("Graph.list",str)
# subject = np.loadtxt("subject.list",str)
# graph_set = set(graph)
# subject_set = set(subject)
# diff_set = subject_set - graph_set
# print(len(diff_set))
# diff_set = list(diff_set)
# diff_set = sorted(diff_set)
# with open('diff_set.list','w') as f:
#     for i in diff_set:
#         # print(i)
#         f.write(i+'\n')

database_dict = {}
path = "/home/sharedata/gpuhome/rhb/fmri/ASD_GRAPH"
for i in os.listdir(path):
    # print(i)
    database_name = re.findall('(.*)_\d\d+',i)[0]
    # print(database_name)
    if database_name not in database_dict:
        database_dict[database_name] = 0
        for j in os.listdir(os.path.join(path,i)):
            # print(j)
            if 'V_index' in j:
                print(j)
                index = re.findall('V_index_(\d+)',j)[0]
                database_dict[database_name] = max(int(index),database_dict[database_name])
print(database_dict)
# np.savetxt('diff_set.list',diff_set,fmt=str)