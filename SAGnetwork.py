import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from SAGPool import SAGPool


class Net(torch.nn.Module):
    def __init__(self, hidden, num_features=2, num_classes=2,pooling_ratio=0.8,dropout_ratio=0.5):
        super(Net, self).__init__()
        # self.args = args
        # self.num_features = args.num_features
        # self.nhid = args.nhid
        # self.num_classes = args.num_classes
        # self.pooling_ratio = args.pooling_ratio
        # self.dropout_ratio = args.dropout_ratio
        self.num_features = num_features
        self.nhid = hidden
        self.num_classes = num_classes
        self.pooling_ratio = pooling_ratio
        self.dropout_ratio = dropout_ratio

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.pool1 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.pool2 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.pool3 = SAGPool(self.nhid, ratio=self.pooling_ratio)

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

        self.node_clf = torch.nn.Linear(self.nhid,self.num_classes)
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = F.relu(self.conv1(x, edge_index,edge_weight = edge_attr))
        x, edge_index, edge_attr, batch, _ = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index,edge_weight = edge_attr))
        x, edge_index, edge_attr, batch, _ = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, edge_attr, batch, _ = self.pool3(x, edge_index, edge_attr, batch)



        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x

    def forward_cl(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = F.relu(self.conv1(x, edge_index,edge_weight = edge_attr))
        x, edge_index, edge_attr, batch, _ = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index,edge_weight = edge_attr))
        x, edge_index, edge_attr, batch, _ = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, edge_attr, batch, _ = self.pool3(x, edge_index, edge_attr, batch)
        node_clf = self.node_clf(x)
        node_clf = F.log_softmax(node_clf,dim=-1)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        # x = F.relu(self.lin2(x))

        return x,node_clf