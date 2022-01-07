from Floyd import  floyd
import minimum_tree
import numpy as np
import os
import dionysus as d
# from ripser import ripser
from scipy.spatial.distance import squareform
from sklearn.linear_model import LinearRegression
class TDP_featrue:
    def __init__(self,directory,num_nodes,ts,num_feature=None):
        self.directory = directory
        self.num_nodes = num_nodes
        self.ts = ts #时间片的数量
        if not (num_feature is None):
            self.num_feature = num_feature
        else:
            self.load_data()
        self.distance = self.cal_distance()
        self.check(self.distance)
        print(directory)




    def load_data(self):
        num_feature = []
        for i in range(self.ts):
            filename = os.path.join(self.directory,'V_index_{}.npy'.format(i))
            one_voxel = np.load(filename)
            num_feature.append(one_voxel)

        self.num_feature = np.array(num_feature)

    def check(self,data):
        for i in range(self.num_nodes):
            if self.distance[i,i]:
                print(i,self.distance[i,i])
            for j in range(i):
                if self.distance[i,j] != self.distance[j,i]:
                    print(i,j,self.distance[i,j],self.distance[j,i])



    def cal_distance(self):
        perason = np.corrcoef(self.num_feature[:, :self.num_nodes].T)
        print(perason.shape)
        perason = (perason + perason.T)/2
        perason = 1 - perason
        perason[range(self.num_nodes),range(self.num_nodes)] = 0
        # distance = floyd(perason) #单链聚合
        return perason

    def count_inf(self,betla):
        cnt = 0
        for i in betla:
            if i.death ==  np.inf:
                cnt += 1
        return cnt

    def LinearRegression(self,x,y):
        # x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
        # y = np.array([5, 20, 14, 32, 22, 38])
        x = np.array(x).reshape((-1,1))
        y = np.array(y)
        model = LinearRegression()
        model.fit(x,y)
        print('coef:{}'.format(model.coef_))

        return model.coef_

    def cal_TDP(self):
        min_tree = minimum_tree.minimum_tree(self.distance)
        edge_list = min_tree.create_tree()
        edge_summary = [0 for i in range(self.num_nodes)]
        for i in range(self.num_nodes-2,-1,-1):
            edge_summary[i] = edge_summary[i+1] + edge_list[i+1]
        distance_matrix = squareform(self.distance)
        point = []
        cnt = 0
        for i in edge_list:
            ripsAux = d.fill_rips(distance_matrix,2,i)
            f = d.homology_persistence(ripsAux)
            dgms = d.init_diagrams(f,ripsAux)
            # ripsAux = ripser(self.distance, maxdim=2, thresh=i, distance_matrix=True)
            # dgms = ripsAux['dgms']
            # print(len(dgms[1]),len(dgms[0]))
            # for j, dgm in enumerate(dgms):
            #     if j == 0:
            #         continue
                # print('Dimension:', j,len(dgm))
                # for p in dgm:
                #     print(p)
            if len(dgms) >= 2:
                B0 = self.count_inf(dgms[0])
                B1 = self.count_inf(dgms[1])
            else:
                B0 = self.count_inf(dgms[0])
                B1 = 0
            # print(B0,B1)
            IPF = B0*edge_summary[cnt]/(self.num_nodes*(self.num_nodes-1))
            one_point = [i,B0,B1,IPF]
            point.append(one_point)
            cnt += 1
        BNP = self.LinearRegression([i[0] for i in point],[i[1] for i in point])
        SIP = self.LinearRegression([i[0] for i in point],[i[3] for i in point])
        feature = []
        for i in point:
            feature.append(i[1])
            feature.append(i[2])
            feature.append(i[3])
        feature.append(BNP[0])
        feature.append(SIP[0])
        print(feature)
        return feature


if __name__ == '__main__':
    directory = "/home/gpusharedata/rhb/fmri/no_detla_data/PostProcessing/Graph/002_S_0295_0"
    num_feature = 90
    ts = 118
    tdp_feature = TDP_featrue(directory,num_feature,ts)
    point = tdp_feature.cal_TDP()
    print(point)