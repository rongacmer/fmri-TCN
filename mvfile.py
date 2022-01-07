import os
import shutil

# source_directory = '/userdata/sharedata/gpuhome/rhb/fmri/new_data/FunImgARCFWS/'
# target_directory = '/userdata/sharedata/gpuhome/rhb/fmri/new_data/Postprocessing/ARCFWS'
#
# for i in os.listdir(source_directory):
#     subject_directory = os.path.join(source_directory,i)
#     for j in os.path.join(subject_directory):
#         if 'nii' in j:
#             source_fileanme = os.path.join(subject_directory,j)
#             target_filename = os.path.join(target_directory,i+'.nii')
#             shutil.copy(source_fileanme,target_filename)
#             print(source_fileanme,target_filename)

####################创建list############################
# import numpy as np
# import re
# categroy = ['AD','NC','eMCI']
# path = '/home/sharedata/gpuhome/rhb/fmri/no_detla_subject'
# filename = [os.path.join(path,'raw_{}.txt'.format(i)) for i in categroy]
#
# pattern = re.compile('\d+_S_\d+')
#
# Alltxt = []
# Allsubject_filename = '/home/sharedata/gpuhome/rhb/fmri/no_detla_subject/ALL_subject.list'
# Allsubject = np.loadtxt(Allsubject_filename,dtype=str)
# outputtxt = [[] for  i in range(len(categroy))]
# for i in filename:
#     txt = np.loadtxt(i,dtype=str)
#     Alltxt.append(txt)
#
# for i in Allsubject:
#     for j in range(len(Alltxt)):
#         subject = pattern.findall(i)[0]
#         if subject in Alltxt[j]:
#             outputtxt[j].append(i)
#             continue
#
# sum = 0
# for cnt,i in enumerate(outputtxt):
#     sum += len(i)
#     output_filename = os.path.join(path,'{}.list'.format(categroy[cnt]))
#     np.savetxt(output_filename,i,fmt='%s')
#     # print(len(i))

# print(sum)
# import shutil
# import re
# ###############mv mask###################
# mask_filename = '/home/sharedata/gpuhome/rhb/fmri/no_detla_data/Masks'
# target_filename = '/home/sharedata/gpuhome/rhb/fmri/no_detla_data/PostProcessing/MASKs/'
# pattern = re.compile('\d+_S_\d+_\d')
# for i in os.listdir(mask_filename):
#     subject = pattern.findall(i)
#     if len(subject):
#         # print(subject[0])
#         source = os.path.join(mask_filename,i)
#         target = os.path.join(target_filename,'{}.nii'.format(subject[0]))
#         # print(source)
#         # print(target)
#         shutil.copy(source,target)

directory = '/userdata/sharedata/gpuhome/rhb/fmri/fmri_for_paper/ADNI'
target = '/userdata/sharedata/gpuhome/rhb/fmri/fmri_for_paper/FunImg'

for subject in os.listdir(directory):
    subject_dir = os.path.join(directory,subject)
    cnt = 0
    for i in sorted(os.listdir(subject_dir)):
        subsub_dir = os.path.join(subject_dir,i)
        print(os.listdir(subsub_dir))
        for j in sorted(os.listdir(subsub_dir)):
            print(j)
            subusubsub_dir = os.path.join(subsub_dir,j)
            # print(subusubsub_dir)
            for k in sorted(os.listdir(subusubsub_dir)):
                subsubsubsub_dir = os.path.join(subusubsub_dir,k)
                for kk in sorted(os.listdir(subsubsubsub_dir)):
                    filename = os.path.join(subsubsubsub_dir,kk)
                    # print(filename)
                    if 'dcm' in filename:
                        print(filename)
                        target_filename = '{}_{}'.format(subject,cnt)
                        # cnt += 1
                        target_filename = os.path.join(target,target_filename)
                        if not os.path.exists(target_filename):
                            os.makedirs(target_filename)
                            cnt += 1
                        target_filename = os.path.join(target_filename,kk)
                        # shutil.move(filename,target_filename)
                        print(target_filename)

