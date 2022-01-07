import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim
from config import args
from torch.nn import functional as F
import torch.utils.data.dataloader as dataloader

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

expect_tho = 0.05
hidden_size = 64
batch_size = args.batch_size

# 定义期望平均激活值和KL散度的权重
tho_tensor = torch.FloatTensor([expect_tho for _ in range(hidden_size)])
if torch.cuda.is_available():
    tho_tensor = tho_tensor.cuda()
_beta = 3

def KL_devergence(p, q):
    """
    Calculate the KL-divergence of (p,q)
    :param p:
    :param q:
    :return:
    """
    q = torch.nn.functional.softmax(q, dim=0)
    q = torch.sum(q, dim=0)/batch_size  # dim:缩减的维度,q的第一维是batch维,即大小为batch_size大小,此处是将第j个神经元在batch_size个输入下所有的输出取平均
    s1 = torch.sum(p*torch.log(p/q))
    s2 = torch.sum((1-p)*torch.log((1-p)/(1-q)))
    return s1+s2


class AutoEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim=30, hidden_dims = None):
        super(AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.raw_channels = in_channels
        modules = []
        if hidden_dims is None:
            hidden_dims = [512,256,128,latent_dim]

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels,h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)

        #Build decoder
        modules = []
        hidden_dims.reverse()

        for i in range(len(hidden_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i],hidden_dims[i+1]),
                    nn.BatchNorm1d(hidden_dims[i+1]),
                    nn.LeakyReLU())
                )
        modules.append(
            nn.Sequential(
                nn.Linear(hidden_dims[-1],self.raw_channels),
                nn.LeakyReLU()
            )
        )
        self.decoder = nn.Sequential(*modules)

    def forward(self, x):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        return [x,encoder_out, decoder_out]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        _beta = 3
        input_x = args[0]
        encoder_out = args[1]
        decoder_out = args[2]
        loss = F.mse_loss(input_x,decoder_out)
        _kl = KL_devergence(tho_tensor, encoder_out)
        loss += _beta * _kl
        return {'loss': loss}

