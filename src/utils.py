import timm
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import numpy as np

def build_model(backbone='resnet50', pretrained=True, num_classes=2):
    model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)
    return model

def compute_metrics(y_true, y_pred_probs, thresh=0.5):
    y_pred = (y_pred_probs >= thresh).astype(int)
    metrics = {}
    metrics['per_class_f1'] = f1_score(y_true, y_pred, average=None).tolist()
    metrics['per_class_precision'] = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    metrics['per_class_recall'] = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    try:
        metrics['per_class_auc'] = roc_auc_score(y_true, y_pred_probs, average=None).tolist()
    except Exception:
        metrics['per_class_auc'] = [None] * y_true.shape[1]
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro').item()
    return metrics
