from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np
import re

class ROI_dataset(Dataset):
    def __init__(self,root_dir,subject_txt):
        self.subject_list = np.loadtxt(subject_txt,dtype=str)
        self.subject_list = set(self.subject_list)
        self.cache = {}
        self._root_dir = root_dir
        self.data_path = os.listdir(root_dir)
        self.data_list = [] #根据suject_txt过滤获得的数据列表
        self.filter_data()


    def filter_data(self):
        for i in self.data_path:
            match = re.match('\d+_S_\d+_\d+',i)
            if match[0] in self.subject_list:
                self.data_list.append(i)
    def get_subject_len(self):
        tmp = np.load(os.path.join(self._root_dir,self.data_list[0]))
        return len(tmp)

    def __getitem__(self, item):
        if self.data_list[item] in self.cache:
            pass
        else:
            self.cache[self.data_list[item]] = np.load(os.path.join(self._root_dir,self.data_list[item]))
            self.cache[self.data_list[item]] = self.cache[self.data_list[item]].astype(np.float32)
            sumx = np.isnan(self.cache[self.data_list[item]]).sum()
            if sumx:
                print(self.data_list[item],sumx)
        return {'data': self.cache[self.data_list[item]],
                'items':item
                }


    def __len__(self):
        return len(self.data_list)

if __name__ == '__main__':
    mydataset = ROI_dataset("/home/sharedata/gpuhome/rhb/fmri/no_mean_data/PostProcessing/AAL/ROI_0")
    dataloader = DataLoader(mydataset,batch_size=36,shuffle=True)
    for i in dataloader:
        print(i.shape)