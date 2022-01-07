from net_fmri import ResGCN
from TCN import TemporalConvNet
import torch
import torch.nn as nn
from torch_geometric.data import InMemoryDataset, Data
from torch.utils.data import DataLoader
import torch_geometric
import torch.nn.functional as F
import os
from ADNI_fmri_dataset import fmriDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ATT(nn.Module):
    def __init__(self,net_num):
        super(ATT, self).__init__()
        self.net_num = net_num
        self.fc1 = nn.Linear(self.net_num,self.net_num)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,data):
        w = self.fc1(data)
        w = self.softmax(w)
        output = data * w
        return output

class hypermodel(nn.Module):
    def __init__(self,GCN_net:ResGCN):
        super(hypermodel,self).__init__()
        self.GCN = GCN_net
        # self.fc2 = nn.Linear(TCN_output,2)


    def forward(self,data):
        x,edge_index,edge_attr,index_cnt = data['x'],data['edge_index'],data['edge_attr'],data['index_cnt']
        data_list = []
        B = x.size()[0]
        ######################graph############################################
        for i in range(B):
                start = 0
                end = index_cnt[i]
                newdata = Data(x=x[i], edge_index=edge_index[i][:,start:end], edge_attr=edge_attr[i][start:end])
                data_list.append(newdata)

        loader = torch_geometric.data.DataLoader(data_list, batch_size=B)
        res = []
        # res = torch.empty(size=(B*self.time_length,self.GCN_output)).to(device)
        for idx,i in enumerate(loader):
            output,node_output = self.GCN.forward(i)
            # res.append(self.GCN.forward(i))
            # res[idx * self.time_length:(idx + 1) * self.time_length, :] = self.GCN.forward_cl(i)
        # res = torch.stack(res)
        # del data_list[:]
        # torch.cuda.empty_cache()
        #######################################
        # output = x.view((B,-1))

        #concat
        return output,node_output


if __name__ == '__main__':
    root_dir = '/home/rhb/fmri/output/P_network_backup'
    niidatadir = '/home/rhb/fmri/data/PostProcessing/ARFWS'
    p_network = '/home/rhb/fmri/data/PostProcessing/P_network'
    subjectdir = '/home/rhb/fmri/subject/'
    Template = '/home/sharedata/rhb/fMRI/data/Template/AAL_61x73x61_YCG.nii'

    clf = ['AD', 'NC']

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    dataset = fmriDataset(root_dir, 90, True, niidatadir, 137, clf, Template, subjectdir, p_network)
    loader = DataLoader(dataset, batch_size=12)

    GCN = ResGCN(128)
    TCN = TemporalConvNet(128,[64,32,1])
    # net = GCN_TCN(137,GCN,TCN,128)
    # net = net.to(device)
    # # net = net.double()
    # for i in loader:
    #     for j in i.keys():
    #         i[j] = i[j].to(device)
    #     output = net(i)
    #     print(output.size())