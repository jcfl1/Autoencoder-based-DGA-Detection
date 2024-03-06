import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, ConfusionMatrixDisplay, confusion_matrix
import seaborn as sns

def plot_roc_curve(y_true, y_score, max_fpr=1.0):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    aucroc = roc_auc_score(y_true, y_score)
    plt.plot(100*fpr[fpr < max_fpr], 100*tpr[fpr < max_fpr], label=f'ROC Curve (AUC = {aucroc:.4f})')
    plt.xlim(-2,102)
    plt.xlabel('FPR (%)')
    plt.ylabel('TPR (%)')
    plt.legend()
    plt.title('ROC Curve and AUCROC')

def get_overall_metrics(y_true, y_pred):
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
  acc = (tp+tn)/(tp+tn+fp+fn)
  tpr = tp/(tp+fn)
  fpr = fp/(fp+tn)
  precision = tp/(tp+fp)
  f1 = (2*tpr*precision)/(tpr+precision)
  return {'acc':acc,'tpr':tpr,'fpr':fpr,'precision':precision,'f1-score':f1}