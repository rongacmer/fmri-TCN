


# class minimun_tree:
#     def __init__(self,matrix,node_num):
#         self.node_num = node_num
#         self.matrix = matrix

import numpy as np
# import dionysus as d
from ripser import ripser
from scipy.spatial.distance import squareform

num_nodes = 4
data = np.ones((num_nodes,num_nodes))
# data[0,1] = data[1,0] = 1
data[range(num_nodes),range(num_nodes)] = 0
data[0,3] = data[3,0] = 1.4
data[1,2] = data[2,1] = 1.4

a = ripser(data,maxdim=2,thresh=2,distance_matrix=True)
print(a['dgms'][0])

# print(data)
# data = squareform(data)
# ripsAux = d.fill_rips(data, 2, 0.5)
# f = d.homology_persistence(ripsAux)
# dgms = d.init_diagrams(f,ripsAux)
# a = dgms[0][0]
# print(dgms[0][0].death == np.inf)
# for i,dgm in enumerate(dgms):
#     print('Dimension:',i)
#     for p in dgm:
#         print(p)