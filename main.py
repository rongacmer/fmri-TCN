from config import args
from utils import logger
from datasets import get_dataset
from train_eval_tcn import cross_validation_with_val_set,  cross_validation_with_test
import os
import Criterion
from sklearn.metrics import roc_auc_score
import numpy as np
def run_exp_feat_study():
    print('[INFO] training ..')
    print('threshold:{}'.format(args.threshold))
    dataset = get_dataset(args.dataset, clf=args.clf, root_dir = args.data_root, time_slice = args.ts,subjectdir=args.subject_dir,
                          threshold = args.threshold) #获取数据集
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    #训练
    train_acc, acc, std, duration = cross_validation_with_val_set(
        dataset,
        folds=5,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_decay_factor=args.lr_decay_factor,
        lr_decay_step_size=args.lr_decay_step_size,
        weight_decay=0.0005,
        epoch_select=args.epoch_select,
        with_eval_mode=args.with_eval_mode,
        logger=logger, model_PATH=args.model_dir, semi_split=args.semi_split, fold_list=args.fold_list,time_length=args.ts)
    #汇总最终结果，包括dataset,model,评选方法,模型保存路径
    summary1 = 'data={}, model={},  eval={}, save_model_directory={}'.format(
        args.dataset, args.model,  args.epoch_select, args.model_dir)
    #输出训练集准确率，测试集准确率
    summary2 = 'train_acc={:.2f}, test_acc={:.2f} ± {:.2f}, sec={}'.format(
        train_acc * 100, acc * 100, std * 100, round(duration, 2))
    print('{}: {}, {}'.format('result', summary1, summary2))

def test_model(test_start = 0):
    #测试模型
    print('[INFO] testing..')
    dataset = get_dataset(args.dataset, clf=args.clf, root_dir=args.data_root,time_slice=args.ts,subjectdir=args.subject_dir,
                          threshold=args.threshold) #获取数据集
    #训练
    train_acc, acc, std, duration,ALL_label,ALL_prob = cross_validation_with_test(
        dataset,
        folds=5,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_decay_factor=args.lr_decay_factor,
        lr_decay_step_size=args.lr_decay_step_size,
        weight_decay=0,
        epoch_select=args.epoch_select,
        with_eval_mode=args.with_eval_mode,
        logger=logger, model_PATH=args.model_dir, semi_split=args.semi_split, fold_list=args.fold_list,time_length=args.ts,test_start=test_start)
    #汇总最终结果，包括dataset,model,评选方法,模型保存路径
    summary1 = 'data={}, model={},  eval={}, save_model_directory={}'.format(
        args.dataset, args.model_name,  args.epoch_select, args.model_dir)
    #输出训练集准确率，测试集准确率
    summary2 = 'train_acc={:.2f}, test_acc={:.2f} ± {:.2f}, sec={}'.format(
        train_acc * 100, acc * 100, std * 100, round(duration, 2))
    print('{}: {}, {}'.format('result', summary1, summary2))
    return acc,std,ALL_label,ALL_prob

def cal_final_cri(prob,label):
    final_prob = [[0] * len(label[i]) for i in range(5)]
    for i in range(len(prob)):
        for j in range(5):
            for k in range(len(label[j])):
                final_prob[j][k] += prob[i][j][k]
    pred = [[] for i in range(5)]
    for i in range(5):
        for j in range(len(label[i])):
            final_prob[i][j] /= len(prob)
            pred[i].append(final_prob[i][j] >= 0.5)
    ALL_cri = []
    for i in range(5):
        one_cri = Criterion.criterion(pred[i],label[i])
        print(one_cri)
        auc = roc_auc_score(label[i],final_prob[i])
        one_cri.Cri.append(auc)
        ALL_cri.append(one_cri.Cri)
    print('mean:\n')
    print(np.mean(ALL_cri, axis=0))
    print('std:\n')
    print(np.std(ALL_cri, axis=0))



if __name__ == '__main__':
    if args.trainortest == 'train':
        run_exp_feat_study()
    elif args.trainortest == 'test':
        # test_model(0)
        total_acc = []
        total_std = []
        # i = 0
        summary_prob = []
        for i in range(0, 175-args.ts+1, 10):
        # for i in range(0,130-args.ts + 1,10):
            print("test_start:{}".format(i))
            acc,std,ALL_label,ALL_prob = test_model(i)
            summary_prob.append(ALL_prob)
            total_acc.append(acc)
            total_std.append(std)
        print(ALL_prob)
        print(total_acc)
        print(total_std)
        cal_final_cri(summary_prob,ALL_label)