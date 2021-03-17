import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score

from argus.metrics.metric import Metric


class MultiAUC(Metric):
    name = 'multi_auc'
    better = 'max'

    def __init__(self, num_classes=11):
        self.num_classes = num_classes

    def reset(self):
        self.y_pred = []
        self.y_true = []

    def update(self, step_output: dict):
        pred = step_output['prediction'].cpu().numpy()
        trg = step_output['target'].cpu().numpy()

        self.y_pred.append(pred)
        self.y_true.append(trg)

    def compute(self):
        self.y_pred = np.concatenate(self.y_pred)
        self.y_true = np.concatenate(self.y_true)

        aucs = []
        for i in range(self.num_classes):
            aucs.append(roc_auc_score(self.y_true[:, i], self.y_pred[:, i]))
        return np.mean(aucs), aucs

    def epoch_complete(self, state):
        with torch.no_grad():
            score, aucs = self.compute()
        name_prefix = f"{state.phase}_" if state.phase else ''
        state.metrics[name_prefix + self.name] = score
        state.logger.info(f'AUC: {aucs}')
