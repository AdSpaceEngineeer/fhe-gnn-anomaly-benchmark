"""Model quality metrics; never tune the threshold on a submission."""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


def quality(labels, scores, threshold):
    labels, scores = np.asarray(labels), np.asarray(scores)
    predicted = (scores >= threshold).astype(int)
    result = {"threshold": float(threshold),
              "accuracy": float(accuracy_score(labels, predicted)),
              "precision": float(precision_score(labels, predicted, zero_division=0)),
              "recall": float(recall_score(labels, predicted, zero_division=0)),
              "f1": float(f1_score(labels, predicted, zero_division=0))}
    two_classes = len(np.unique(labels)) == 2
    result["roc_auc"] = float(roc_auc_score(labels, scores)) if two_classes else None
    result["average_precision"] = float(average_precision_score(labels, scores)) if two_classes else None
    return result
