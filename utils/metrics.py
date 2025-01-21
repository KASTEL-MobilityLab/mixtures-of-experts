"""
Calculate metric after evaluation.
"""
import numpy as np


class Evaluator():
    """Evaluator to calculate metrics"""
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros(
            (self.num_class,) * 2)

    def pixel_accuracy(self):
        """Accuray of pixel"""
        acc = np.diag(self.confusion_matrix).sum() / \
              self.confusion_matrix.sum()
        return acc

    def pixel_accuracy_class(self):
        """Accuray of pixel from each class"""
        acc = np.diag(self.confusion_matrix) / \
              self.confusion_matrix.sum(axis=1)
        acc = np.nanmean(acc)
        return acc

    def mean_intersection_over_union(self, label_names):
        """Mean Intersection over Union for each class and whole model"""
        miou = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) +
            np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))
        assert len(miou) == len(label_names)
        # mean Miou for each class
        log_miou_cls = list()
        for i in range(len(miou)):
            log_miou_cls.append(
                '{:<15}: {:.4f}'.format(label_names[i], miou[i] * 100.0))
        miou = np.nanmean(miou)
        return miou, log_miou_cls

    def frequency_weighted_intersection_over_union(self):
        """Frequency Weighted Intersection over Union"""
        freq = np.sum(self.confusion_matrix, axis=1) / \
            np.sum(self.confusion_matrix)
        iou = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) +
            np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))

        fwiou = (freq[freq > 0] * iou[freq > 0]).sum()
        return fwiou

    def _generate_matrix(self, gt_image, pre_image):
        """Calculate confusion matrix"""
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        """Calculate confusion matrix for batch"""
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        """reset confusion matrix"""
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
