# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np

class ConfusionMatrix(object):

    def __init__(self, n_classes, n_batch):
        self.n_classes = n_classes
        self.n_batch = n_batch
        self.epsilon = 0.000001
        # axis = 0: target
        # axis = 1: prediction
        self.IoUList = []

        self.confusion_matrix = np.zeros((n_classes, n_classes))

        # self.batch_confusion_matrix = np.zeros((n_batch, n_classes, n_classes))

    def tmp_IoU(self, tmp):
        # for ISIC
        intersect = np.diag(tmp)
        union = tmp.sum(axis=1) + tmp.sum(axis=0) - np.diag(tmp)
        iou = (intersect + self.epsilon) / (union + self.epsilon)
        # for ISIC
        iou[iou < 0.65] = 0
        # iou = np.mean(np.nan_to_num(iou))
        return iou

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        # hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
        hist = np.bincount(n_class * label_true[mask].int() + label_pred[mask].int(), minlength=n_class**2).reshape(n_class, n_class)
        return hist


    def update(self, label_trues, label_preds):
        #bindex = 0
        # print("label_trues:", label_trues.shape)
        # print("label_preds:", label_preds.shape)
        for lt, lp in zip(label_trues, label_preds):
            # print("lt:",lt.shape)
            # print("lp:",lp.shape)
            tmp = self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

            #self.batch_confusion_matrix[bindex] = tmp
            self.confusion_matrix += tmp
            # self.IoUList.append(self.tmp_IoU(tmp))
            #bindex += 1

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        # accuracy is recall/sensitivity for each class, predicted TP / all real positives
        # axis in sum: perform summation along
        acc = np.nan_to_num(np.diag(hist) / hist.sum(axis=1))
        acc_mean = np.mean(np.nan_to_num(acc))
        
        intersect = np.diag(hist)
        union = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)

        #union[union == 0] = 1
        iou = (intersect + self.epsilon) / (union + self.epsilon)


        # iou_mean = np.mean(np.nan_to_num(np.array(self.IoUList)))
        iou_mean = np.mean(np.nan_to_num(iou))
        # for ISIC
        # iou = np.mean(np.nan_to_num(np.array(self.IoUList)), axis=0)
        # iou_mean = np.sum(np.nan_to_num(iou)) / no_zero_union
        
        freq = hist.sum(axis=1) / hist.sum() # freq of each target
        # fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
        freq_iou = (freq * iou).sum()
        #dsc = (intersect * 2 + self.epsilon) / (hist.sum(axis=1) + hist.sum(axis=0) + self.epsilon)
        #dsc_overall = (np.sum(intersect) * 2 + self.epsilon) / ((hist.sum(axis=1) + hist.sum(axis=0)).sum(axis=-1) + self.epsilon)

        overall= {
            'accuracy': acc,
            'accuracy_mean': acc_mean,
            'freqw_iou': freq_iou,
            'iou': iou,
            'iou_mean': iou_mean,
            #'dsc': dsc,
            #'dsc_overall': dsc_overall
            # 'IoU_threshold': np.mean(np.nan_to_num(self.iou_threshold)),
         }


        # b_hist = self.batch_confusion_matrix
        # b_intersect = b_hist[:, [i for i in range(self.n_classes)], [i for i in range(self.n_classes)]]
        # b_union = b_hist.sum(axis=2) + b_hist.sum(axis=1) - b_intersect
        #
        # b_iou = (b_intersect + self.epsilon) / (b_union + self.epsilon)
        # # b_mean_iou = np.mean(np.nan_to_num(b_iou), axis=-1)
        #
        # b_iou_mean = np.sum(np.nan_to_num(b_iou), axis=-1) / no_zero_union
        # b_dsc = (b_intersect * 2 + self.epsilon) / ((b_hist.sum(axis=2) + b_hist.sum(axis=1)) + self.epsilon)
        # b_dsc_overall = (np.sum(b_intersect, -1) * 2 + self.epsilon) / ((b_hist.sum(axis=2) + b_hist.sum(axis=1)).sum(axis=-1) + self.epsilon)
        # batch = {
        #     'iou': b_iou,
        #     'iou_mean': b_iou_mean,
        #     'dsc': b_dsc,
        #     'dsc_overall': b_dsc_overall
        # }
        return overall #, batch

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.IoUList = []
        # self.batch_confusion_matrix = np.zeros((self.n_batch, self.n_classes, self.n_classes))