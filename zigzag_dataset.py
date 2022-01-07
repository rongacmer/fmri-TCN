import numpy as np
import os

from torch_geometric.data import InMemoryDataset, Data
import torch
import re
import networkx as nx
from abc import ABC
import torch_geometric.data
import numpy as np
import nibabel as nib
import torch_geometric.utils as pyg_utils
import scipy.stats.distributions
from scipy import stats
from itertools import repeat, product
# from net_fmri import ResGCN
from torch.utils.data import DataLoader


def graph_threshold(adj_array, threshold, num_nodes):
    num_to_filter: int = int((threshold / 100.0) * (num_nodes * (num_nodes - 1) / 2))

    # For threshold operations, zero out lower triangle (including diagonal)
    adj_array[np.tril_indices(num_nodes)] = 0 #上三角矩阵设为0

    # Following code is similar to bctpy
    indices = np.where(adj_array)
    sorted_indices = np.argsort(adj_array[indices])[::-1] #将值按从大到小的顺序排列，返回索引值
    adj_array[(indices[0][sorted_indices][num_to_filter:], indices[1][sorted_indices][num_to_filter:])] = 0 #设为0

    # Just to get a symmetrical matrix
    adj_array = adj_array + adj_array.T #对称化

    # Diagonals need connection of 1 for graph operations
    adj_array[np.diag_indices(num_nodes)] = 1.0 #对角线设为1

    return adj_array

class zigzagDataset(InMemoryDataset):
    def __init__(self, root,  num_nodes: int,graphdir:str,
                 time_length: int = 118,clf = ['AD','NC'],Template = None,subjectdir = None,
                 transform=None, pre_transform=None,time_slice = 12,status = 'train',test_start = 0, threshold = 100):
        '''
        fmri数据集
        :param root:数据集的存储路径
        :param num_nodes: roi脑区数量，default：90
        :param niidatadir: nii文件夹路径
        :param time_length:时间点的数量，default: 137
        :param clf: 分类类型，如['AD','NC']
        :param Template: 模板路径，使用ALL模板
        :param subjectdir:subject文件list的文件夹，
        :param pvaluedir:p-value值文件夹路径
        :param time_slice: 随机截取一段时间点的长度
        :param status: 数据集状态，’train' or 'test'
        :param test_start: 测试时开始的时间点
        '''
        self.num_nodes = num_nodes
        self.time_lenght = time_length
        self.graphdir = graphdir
        self.clf = clf
        self.Template = Template
        self.clf_dir = {i:self.graphdir for i in clf} #补全为绝对路径
        # self.pnewtrok_dir = {i:os.path.join(self.pvaluedir,i) for i in clf} #p_value路径
        self.subjectlist = {i:os.path.join(subjectdir,'{}.list'.format(i)) for i in self.clf} #xxx/AD.list,xxx/NC.list
        self.threshold = threshold #阈值
        super(zigzagDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0]) #加载数据集
        self.cache = {} #缓存
        # self.gcn_model = ResGCN(128)
        self.state = 'processing' #状态1 'processing' or 'check' 处理或检查数据
        self.status = status #状态2 'train' or 'test
        self.time_slice = time_slice #截取的时间的长度
        # if model_path and os.path.exists(model_path):
        #     print('{} load successful'.format(model_path))
        #     self.gcn_model.load_state_dict(torch.load(model_path))
        # self.gcn_model.eval()
        self.test_start = test_start #测试时开始的时间点


    @property
    def processed_file_names(self):
        return ['data_fmri_brain.dataset']

    def set_state_train(self):
        self.state = 'processing'

    def set_state_check(self):
        self.state = 'check'

    def set_test_start(self,test_start):
        '''
        设置开始的标签
        :param test_start:
        :return:
        '''
        self.test_start = test_start

    def cal_roi(self,volumn,mask):
        '''
        计算每个时间点的特征
        :param volumn: 每个时间点的图像
        :param mask:  模板
        :return: size【90，2】 90个脑区，每个脑区有平均值和标准差两个特征
        '''
        ROI = [[] for i in range(self.num_nodes)] #统计每个脑区的体素值
        volumn = stats.zscore(volumn) #标准化
        #遍历每个体素，统计每个脑区的体素
        for i in range(volumn.shape[0]):
            for j in range(volumn.shape[1]):
                for k in range(volumn.shape[2]):
                    roi = int(mask[i,j,k])
                    if roi and roi <= self.num_nodes:
                        ROI[roi-1].append(volumn[i,j,k])

        ROI_stat = np.zeros((self.num_nodes,2)) #返回值，每个脑区的特征
        for i in range(self.num_nodes):
            if len(ROI[i]):
                ROI_stat[i,0] = np.mean(ROI[i]) #平均值
                ROI_stat[i,1] = np.std(ROI[i]) #标准差
        # print(np.max(ROI_stat),np.min(ROI_stat))
        return ROI_stat

    def __load_data(self,subject_directory,y,subject_name):
        '''

        :param subject_directory: fmri文件名
        :param y: 标签
        :param subject_name: 文件名
        :return: deal_data: 处理后的fmri数据
        '''

        subject_data = []
        subject_id = re.findall('(\d+)_S_(\d+)', subject_name) #subject_id
        ts = re.findall('\d+_S_\d+_(\d+)',subject_name) #时间点的数据
        subject_id = int(''.join(subject_id[0])) #转换为数字id，如002_S_003 -> 2003
        edge_index = None #边的索引
        index_cnt = [0] #边的切片
        edge_attr = [] #边的属性
        x = [] #节点
        total_WDP = [] #持续同调特征图
        for i in range(self.time_lenght): #遍历所有时间点
            # roi_stat = self.cal_roi(data[i],self.mask) #节点 size：[90,2]\
            ##########node_feature,p_net,wdp_net文件名##############
            roi_stat_filename = os.path.join(subject_directory,  'V_index_{}.npy'.format(i))
            P_net_filename = os.path.join(subject_directory,'P_index_{}.npy'.format(i))
            WDP_filename = os.path.join(subject_directory,'WDP_index{}.npy'.format(i))
            ################################################################

            ######################加载#######################
            roi_stat = np.load(roi_stat_filename)[:self.num_nodes]

            p_value_network = np.load(P_net_filename)[:self.num_nodes,:self.num_nodes]
            WDP = np.load(WDP_filename)
            if np.isnan(roi_stat).any():
                print(subject_name, roi_stat.max(), roi_stat.min())
                return None
            if np.isnan(WDP).any():
                print(subject_name,WDP.max(),WDP.min())
                return None
            print(roi_stat.shape,p_value_network.shape,WDP.shape)

            ####################p-value矩阵###########################
            # p_value_network_filename = os.path.join(pvaluesubject_directory,'index_{}.npy'.format(i))
            # p_value_network = np.load(p_value_network_filename)[:self.num_nodes,:self.num_nodes]
            p_value_network = graph_threshold(p_value_network,self.threshold,self.num_nodes)


            #######用于填充##############
            # p_value_network = np.ones((self.num_nodes,self.num_nodes))
            # for ns in range(self.num_nodes):
            #     p_value_network[ns,ns] = 1
            ############################
            # x = torch.tensor(roi_stat,dtype=torch.float)
            x.append(roi_stat) #加入一个时间点的节点数据
            total_WDP.append(WDP)
            G = nx.from_numpy_array(p_value_network, create_using=nx.DiGraph) #无向图G

            #######图索引###################
            if edge_index is None:
                edge_index = np.array(G.edges()).transpose()
            else:
                edge_index = np.concatenate((edge_index,np.array(G.edges()).transpose()),axis=1)
            # # #######################################
            index_cnt.append(edge_index.shape[1]) #边的切片，如index[0]-index[1]为第一个时间点的图
            edge_attr += list(list(nx.get_edge_attributes(G, 'weight').values())) #拼接图的属性
            #############################################################
            print('subject:{} time:{}'.format(subject_name,i)) #输出信息，完成到哪个样本的哪个时间点
        x = torch.tensor(x,dtype=torch.float) #size：[137,90,2]
        y = torch.tensor([y]) #label
        #########皮尔逊##########################
        # perason = np.corrcoef(x[:,:,0].T)
        # perason = np.abs(perason)
        # G = nx.from_numpy_array(perason, create_using=nx.DiGraph)
        # for i in range(self.time_lenght):
        #     if edge_index is None:
        #         edge_index = np.array(G.edges()).transpose()
        #     else:
        #         edge_index = np.concatenate((edge_index, np.array(G.edges()).transpose()), axis=1)
        #     index_cnt.append(edge_index.shape[1])  # 边的切片，如index[0]-index[1]为第一个时间点的图
        #     edge_attr += list(nx.get_edge_attributes(G, 'weight').values())  # 拼接图的属性
        ######################################

        edge_index = torch.tensor(edge_index,dtype=torch.long) #边的索引 size:[2,index_cnt[-1]]
        edge_attr = torch.tensor(edge_attr,dtype=torch.float) #边的属性 size:[index_cnt[-1]]
        deal_data = Data(x=x,edge_index=edge_index,edge_attr=edge_attr,y=y) #将数据压缩为一个元数据
        deal_data.id = torch.tensor([subject_id]) #subject_id
        deal_data.ts = torch.tensor([int(ts[0])]) #subject_id的第ts个样本
        deal_data.index_cnt = torch.tensor(index_cnt,dtype=torch.long) #切片索引 size:[138]
        deal_data.wdp = torch.tensor(total_WDP,dtype=torch.float) #size[118,100,100]
        print(subject_id,deal_data.ts)
        print(len(index_cnt))
        return deal_data

    def get(self,idx):
        '''
        训练时获取数据
        :param idx:
        :return:
        '''
        data = self.data.__class__()
        threshold = 1e-5
        max_edge = self.num_nodes * self.num_nodes * self.time_lenght #边的最大数量，使用dataloader时，要保持每个tensor的长度一致
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key,
                                        item)] = slice(slices[idx],
                                                       slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        subject_name = str(data['id'][0].item()) + '_' + str(data['ts'][0].item())
        if self.state == 'processing':
            if subject_name not in self.cache:
                # data_list = []
                # for i in range(data['x'].size()[0]):
                #     start = data['index_cnt'][i]
                #     end = data['index_cnt'][i+1]
                    # newdata = Data(x=data['x'][i], edge_index=data['edge_index'][:,start:end], edge_attr=data['edge_attr'][start:end])
                    # data_list.append(newdata)
                # loader = torch_geometric.data.DataLoader(data_list,batch_size=data['x'].size()[0])
                # # res = self.gcn_model.forward_cl(next(loader))
                # # self.cache[subject_name] = res
                # for i in loader:
                #     res = self.gcn_model.forward_cl(i)
                #     res = torch.tensor(res)
                #     # print(res)
                #     # print(res)
                #     self.cache[subject_name] = res.t().contiguous()

                # self.cache[subject_name] = torch.reshape(data['x'],(self.time_lenght,-1)).t().contiguous()
                # self.cache[subject_name] = self.gcn_model.forward_cl(data)
                # data['edge_attr'][data['edge_attr'] < threshold] = 0
                # data['edge_attr'][data['edge_attr'] > threshold] = 1
                edge_index = np.zeros((2,max_edge)) #边索引
                edge_attr = np.zeros(max_edge) #边属性
                edge_cnt = data['edge_index'].size()[1] #真正边的数量
                edge_index[:,:edge_cnt] = data['edge_index']
                edge_attr[:edge_cnt] = data['edge_attr']
                self.cache[subject_name] = [edge_index.astype(np.long),edge_attr.astype(np.float32)] #载入缓存
            if self.status == 'train':
                # ‘train'的时候，随机取一个时间切片
                if self.time_lenght >= self.time_slice:
                    r = 0
                else:
                    r = np.random.randint(self.time_lenght - self.time_slice)
            else: #'test'的时候，固定一个时间切片
                r = self.test_start
            # print(r,r+self.time_slice)
            # print(subject_name)
            # print(self.cache[subject_name])
            # print(subject_name,self.cache[subject_name].max(),self.cache[subject_name].min())
            # print(self.cache[subject_name].min())


            #返回一个字典
            return {
                'index_cnt':data['index_cnt'][r:r+self.time_slice+1],
                'wdp':data['wdp'][r:r+self.time_slice],
                'edge_index':self.cache[subject_name][0],
                'edge_attr':self.cache[subject_name][1],
                'x':data['x'][r:r+self.time_slice],
                'y':data['y'],
                'id':data['id']
            }

        else:
            return {
                'y':data['y'],
                'id':data['id']
            }
    # def __create_data_object(self,id:int,ts:np.ndarray,edge_attr:torch.tensor,edge_index:torch.tensor,y):
    #     timeseries = normalise_timeseries(timeseries=ts,normalisation=self.normalisation)
    #     x = torch.tensor(timeseries.T,dtype=torch.float)
    #     y = torch.tensor([y])
    #     data = Data(x=x,edge_index=edge_index,edge_attr=edge_attr,y=y)
    #     data.id = torch.tensor([id])
    #     return data

    # def get(self,idx):
    #     pass
    def process(self):
        data_list = []
        # self.mask = nib.load(self.Template)
        # self.mask = self.mask.get_fdata()
        self.mask = self.Template

        for label,cat in enumerate(self.clf):
            cnt = 0
            subject_list = self.subjectlist[cat]
            subject_txt = np.loadtxt(subject_list,dtype=str) #subject文件
            for s in subject_txt:
                # subject_filename = os.path.join(self.clf_dir[cat],'{}.nii'.format(s))
                # p_network_dir = os.path.join(self.pnewtrok_dir[cat],'{}'.format(s))
                subject_directory = os.path.join(self.clf_dir[cat],s)
                subject_data = self.__load_data(subject_directory,label,s)
                if subject_data is None:
                    continue
                data_list.append(subject_data)
                # cnt += 1
                # if cnt >= 3:
                #     break
        data,slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    pass
    # dataset =  ADNIDataset(All_conf.save_data, 90, 100, ConnType.FMRI, Normalisation.SUBJECT, True, subjectfilename=All_conf.demo, datadir = All_conf.data_root)
    # print(dataset[0])


