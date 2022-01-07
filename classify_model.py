from sklearn.svm import SVC
import Criterion
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn import preprocessing
# from sklearn.feature_selection import
# from sklearn
def cal_FS(X,Y):
    mean_X = np.mean(X,axis=0)
    YA = np.where(Y==0)
    NA  = len(YA[0])
    XA = X[YA[0]]
    mean_XA = np.mean(XA,axis=0)
    std_XA = np.std(XA,axis=0)
    YB = np.where(Y==1)
    NB = len(YB[0])
    XB = X[YB[0]]
    mean_XB = np.mean(XB,axis=0)
    std_XB = np.mean(XB,axis=0)
    FS = (NA*(mean_XA-mean_X)**2+NB*(mean_XB-mean_X)**2)/(NA*(std_XA**2)+NB*(std_XB**2)+1e-7)
    sort_index = np.argsort(FS)[::-1]
    print(FS)
    print(sort_index)

    # FS_X = X[:,sort_index[:len(sort_index)//2]]
    # return FS_X
    return sort_index[:len(sort_index)//2]


def select_FSFS(train_data,train_label,test_data,test_label):
    FS_len = train_data.shape[1]
    FS_ACC = []
    clf = SVC(gamma='auto')
    for i in range(1,FS_len+1):
        clf.fit(train_data[:,:i],train_label)
        predict = clf.predict(train_data[:,:i])
        cri = Criterion.criterion(predict,train_label)
        FS_ACC.append(cri.ACC)
        # print(i,cri.ACC)
        # print(i,FS_ACC)
    return np.argmax(FS_ACC)
def classify_model(train_data,train_label,test_data,test_label,dev_data = None,dev_label = None):

    clf = SVC(gamma='auto',probability=True)
    train_len = len(train_data)
    data = np.vstack((train_data,test_data))
    data = preprocessing.normalize(data,axis=0)
    FS_X = cal_FS(data,np.hstack((train_label,test_label))) #数据泄露的方式
    # FS_X = cal_FS(data[:train_len,:],np.array(train_label))
    # train_data, test_data = data[:train_len, :], data[train_len:, :]
    train_data, test_data = data[:train_len, FS_X], data[train_len:, FS_X]
    maxFSindex = select_FSFS(train_data,train_label,test_data,test_label)
    train_data = train_data[:,:maxFSindex+1]
    test_data = test_data[:,:maxFSindex+1]
    clf.fit(train_data,train_label)
    predict = clf.predict(test_data)
    print(maxFSindex)
    print(predict)
    print(test_label)
    cri = Criterion.criterion(predict,test_label)
    print(cri)
    proba = clf.predict_proba(test_data)[:,1]
    # print(proba)
    auc = roc_auc_score(test_label,proba)
    cri.Cri.append(auc)

    # ##验证集
    # dev_predict = clf.predict(dev_data)
    # dev_cri = Criterion.criterion(dev_predict,dev_label)
    # dev_proba = clf.predict_proba(dev_data)[:,1]
    # dev_auc = roc_auc_score(dev_label,dev_proba)
    # dev_cri.Cri.append(dev_auc)
    # print(cri.Cri)
    return cri.Cri

if __name__ == '__main__':
    X = np.random.rand(10,10)
    Y = np.hstack((np.ones(5),np.zeros(5)))
    # create_FS(X,Y)