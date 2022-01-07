import os
import shutil
import nibabel as nib
directory = '/home/sharedata/gpuhome/rhb/fmri/new_data/T1Img'

for i in os.listdir(directory):
    filename = os.path.join(directory,i)
    for j in os.listdir(filename):
        target = os.path.join(filename,j)
        data = nib.load(target)
        data = data.get_fdata()
        print(target,data.shape)