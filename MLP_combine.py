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

class GCN_TCN(nn.Module):
    def __init__(self,time_length,GCN_net:ResGCN, TCN_net:TemporalConvNet,GCN_output = 128,TCN_input = 64):
        super(GCN_TCN,self).__init__()
        self.GCN = GCN_net
        self.TCN = TCN_net
        self.time_length = time_length
        self.GCN_output = GCN_output
        self.fcinput = 180
        self.cof = torch.nn.Parameter(torch.ones(self.fcinput))
        # self.cof = ATT(self.fcinput)
        self.fn = nn.Linear(self.fcinput, TCN_input)
        self.fc1 = nn.Linear(TCN_input, 2)
        self.dp1 = nn.Dropout(0.5)


    def forward(self,data):
        x,edge_index,edge_attr,index_cnt = data['x'],data['edge_index'],data['edge_attr'],data['index_cnt']
        data_list = []
        B = x.size()[0]
        ######################graph############################################
        # for i in range(B):
        #     for j in range(self.time_length):
        #         start = index_cnt[i][j]
        #         end = index_cnt[i][j+1]
        #         newdata = Data(x=x[i][j], edge_index=edge_index[i][:,start:end], edge_attr=edge_attr[i][start:end])
        #         data_list.append(newdata)
        #
        # loader = torch_geometric.data.DataLoader(data_list, batch_size=B*self.time_length)
        # for i in loader:
        #     res = self.GCN.forward_cl(i)
        #######################################
        res_x = x.view((B,self.time_length,-1))

        #concat
        # res = res.view(B,self.time_length,self.GCN_output)
        # res_x = torch.cat((res,res_x),-1)
        ##################

        # res_x = self.cof(res_x)
        # res_x = res_x * self.cof
        # res = res.permute([0,2,1])

        res = self.fn(res_x)
        res = F.relu(res)
        res = self.dp1(res)
        clf = self.fc1(res)
        clf = clf.view(-1,2)
        clf = F.log_softmax(clf, dim=-1)
        res = res.permute([0,2,1])
        # print(res.size())


        output = self.TCN(res)
        return output,clf,None,None


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
    net = GCN_TCN(137,GCN,TCN,128)
    net = net.to(device)
    # net = net.double()
    for i in loader:
        for j in i.keys():
            i[j] = i[j].to(device)
        output = net(i)
        print(output.size())