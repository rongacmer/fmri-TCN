import numpy as np
import sys
class criterion: #评价指标
    def __init__(self, prediction,label):
        esp = sys.float_info.min
        TP,FP,FN,TN = 0, 0, 0, 0
        for i in range(len(prediction)):
            if prediction[i] == label[i]:
                if prediction[i] == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if prediction[i] == 1:
                    FP += 1
                else:
                    FN += 1
        # self.AUC = roc_auc_score(label,score)
        self.ACC = (TP + TN) / (TP + FP + FN + TN + esp)
        self.SEN = TP / (TP + FN)
        self.Recall = TP/(TP+FP+esp)
        self.SPE = TN / (TN + FP + esp)
        self.F1 = 2*self.SEN*self.Recall/(self.SEN+self.Recall+esp)
        self.MCC = (TP*TN - FP * FN)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5+esp)
        self.Cri = [self.ACC,self.SEN,self.SPE,self.F1,self.MCC]
        self.TP =TP
        self.TN =TN
        self.FP =FP
        self.FN =FN


    def __repr__(self):
        output_str = 'ACC SEN SPE F1 MCC TP TN FP FN:{},{},{},{},{},{},{},{},{}'.format(self.ACC,self.SEN,self.SPE,self.F1,self.MCC,self.TP,self.TN,self.FP,self.FN)
        # output_str = 'ACC SEN SPE F1 MCC: {},{},{},{} {}'.format(self.ACC, self.SEN, self.SPE,self.F1, self.MCC)
        return output_str