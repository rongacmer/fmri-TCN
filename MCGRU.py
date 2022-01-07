import torch.nn as nn
import torch
import torch.nn.functional as F
class MCGRU(nn.Module):
    def __init__(self,ts_length,roi_cnt,dropout_ratio = 0.5,):
        super(MCGRU,self).__init__()
        self.conv1 = nn.Conv1d(roi_cnt,32,kernel_size=1,stride=1,padding=0)
        self.conv2 = nn.Conv1d(roi_cnt,32,kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv1d(roi_cnt,32,kernel_size=5,stride=1,padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool1d(3,stride=3)
        self.GRU1 = nn.GRU(input_size=96,hidden_size=32,num_layers=1,batch_first=True,dropout=dropout_ratio)
        self.relu2 = nn.ReLU()
        self.GRU2 = nn.GRU(input_size=32,hidden_size=96,num_layers=1,batch_first=True,dropout=dropout_ratio)
        self.relu3 = nn.ReLU()
        self.GLP = nn.AdaptiveMaxPool1d(output_size=1)
        self.mlp = nn.Linear(96,2)
        self.dropout_ratio = dropout_ratio

    def forward(self,data):
        x = data['x']
        # x = data
        x = x[:,:,:,0]
        x = x.permute([0,2,1])
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat([x1,x2,x3],dim=1)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = x.permute([2,0,1])
        x,hn = self.GRU1(x)
        x = self.relu2(x)
        x,hn = self.GRU2(x)
        x = self.relu3(x)
        x = x.permute([1,2,0])
        x = self.GLP(x)
        x = x.view(-1,96)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.mlp(x)
        x = F.log_softmax(x, dim=-1)
        # print(x.shape)
        return x,x,x,x




if __name__ == '__main__':
    dx = torch.rand((32,90,118))
    net = MCGRU(130,90)
    x = net(dx)