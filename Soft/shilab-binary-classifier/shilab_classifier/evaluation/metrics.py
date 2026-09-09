"""Model-evaluation utilities."""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    roc_curve,
    auc
)


def calculate_metrics(y_true, y_pred, y_prob=None):
    """Calculate classification performance metrics."""
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape != (2, 2):
        print(f"Warning: unexpected confusion matrix shape {cm.shape}, expected (2, 2)")
        if cm.shape[0] == 2 and cm.shape[1] == 1:
            cm = np.column_stack((cm, np.zeros(2)))
        elif cm.shape[0] == 1 and cm.shape[1] == 2:
            cm = np.row_stack((cm, np.zeros(2)))
        else:
            return {
                'accuracy':             0,
                'balanced_accuracy':    0,
                'sensitivity':          0,
                'specificity':          0,
                'precision_benign':     0,
                'recall_benign':        0,
                'f1_benign':            0,
                'precision_malignant':  0,
                'recall_malignant':     0,
                'f1_malignant':         0,
                'mcc':                  0,
                'auc':                  0 if y_prob is not None else None,
                'benign_correct':       0,
                'benign_total':         0,
                'benign_acc':           0,
                'malignant_correct':    0,
                'malignant_total':      0,
                'malignant_acc':        0,
                'confusion_matrix':     np.zeros((2, 2))
            }

    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    benign_correct = tn
    benign_total   = tn + fp
    benign_acc     = tn / benign_total if benign_total > 0 else 0

    malignant_correct = tp
    malignant_total   = tp + fn
    malignant_acc     = tp / malignant_total if malignant_total > 0 else 0

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    balanced_accuracy = (sensitivity + specificity) / 2

    precision_benign = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_benign    = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_benign        = (2 * precision_benign * recall_benign /
                        (precision_benign + recall_benign)
                        if (precision_benign + recall_benign) > 0 else 0)

    precision_malignant = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_malignant    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_malignant        = (2 * precision_malignant * recall_malignant /
                           (precision_malignant + recall_malignant)
                           if (precision_malignant + recall_malignant) > 0 else 0)

    mcc = matthews_corrcoef(y_true, y_pred)

    auc_value = None
    if y_prob is not None:
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_value = auc(fpr, tpr)
        except Exception as e:
            print(f"Error computing AUC: {e}")
            auc_value = 0

    return {
        'accuracy':             accuracy,
        'balanced_accuracy':    balanced_accuracy,
        'sensitivity':          sensitivity,
        'specificity':          specificity,
        'precision_benign':     precision_benign,
        'recall_benign':        recall_benign,
        'f1_benign':            f1_benign,
        'precision_malignant':  precision_malignant,
        'recall_malignant':     recall_malignant,
        'f1_malignant':         f1_malignant,
        'mcc':                  mcc,
        'auc':                  auc_value,
        'benign_correct':       int(benign_correct),
        'benign_total':         int(benign_total),
        'benign_acc':           benign_acc,
        'malignant_correct':    int(malignant_correct),
        'malignant_total':      int(malignant_total),
        'malignant_acc':        malignant_acc,
        'confusion_matrix':     cm
    }


def print_metrics(metrics, dataset_name="Dataset"):
    """Print classification performance metrics."""
    print(f"\n{dataset_name}:")
    print(f"  Accuracy:              {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy:     {metrics['balanced_accuracy']:.4f}")
    print(f"  Sensitivity:           {metrics['sensitivity']:.4f}")
    print(f"  Specificity:           {metrics['specificity']:.4f}")
    print(f"  Precision (Benign):    {metrics['precision_benign']:.4f}")
    print(f"  Recall (Benign):       {metrics['recall_benign']:.4f}")
    print(f"  F1 (Benign):           {metrics['f1_benign']:.4f}")
    print(f"  Precision (Malignant): {metrics['precision_malignant']:.4f}")
    print(f"  Recall (Malignant):    {metrics['recall_malignant']:.4f}")
    print(f"  F1 (Malignant):        {metrics['f1_malignant']:.4f}")
    print(f"  MCC:                   {metrics['mcc']:.4f}")
    if metrics['auc'] is not None:
        print(f"  AUC:                   {metrics['auc']:.4f}")
    print(f"  Benign Acc:            {metrics['benign_acc']:.4f} "
          f"({metrics['benign_correct']}/{metrics['benign_total']})")
    print(f"  Malignant Acc:         {metrics['malignant_acc']:.4f} "
          f"({metrics['malignant_correct']}/{metrics['malignant_total']})")
    print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")