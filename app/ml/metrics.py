"""
Evaluation metrics for binary classification implemented from scratch.
"""
from typing import List, Tuple, Dict, Any
import math


def confusion_matrix(y_true: List[int], y_pred: List[int]) -> Dict[str, int]:
    """Compute confusion matrix for binary classification."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    # Get unique labels
    labels = sorted(set(y_true + y_pred))
    if len(labels) != 2:
        raise ValueError("Binary classification requires exactly 2 classes")
    
    # Assume labels are [negative_class, positive_class]
    neg_class, pos_class = labels[0], labels[1]
    
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == pos_class and pred == pos_class)
    tn = sum(1 for true, pred in zip(y_true, y_pred) if true == neg_class and pred == neg_class)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true == neg_class and pred == pos_class)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == pos_class and pred == neg_class)
    
    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'positive_class': pos_class, 'negative_class': neg_class
    }


def accuracy_score(y_true: List[int], y_pred: List[int]) -> float:
    """Compute accuracy score."""
    if len(y_true) == 0:
        return 0.0
    
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)


def precision_score(y_true: List[int], y_pred: List[int]) -> float:
    """Compute precision score."""
    cm = confusion_matrix(y_true, y_pred)
    tp, fp = cm['tp'], cm['fp']
    
    if tp + fp == 0:
        return 0.0  # No positive predictions
    
    return tp / (tp + fp)


def recall_score(y_true: List[int], y_pred: List[int]) -> float:
    """Compute recall score."""
    cm = confusion_matrix(y_true, y_pred)
    tp, fn = cm['tp'], cm['fn']
    
    if tp + fn == 0:
        return 0.0  # No positive ground truth
    
    return tp / (tp + fn)


def f1_score(y_true: List[int], y_pred: List[int]) -> float:
    """Compute F1 score."""
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    
    if prec + rec == 0:
        return 0.0
    
    return 2 * prec * rec / (prec + rec)


def roc_auc_score(y_true: List[int], y_scores: List[float]) -> float:
    """Compute ROC AUC score using trapezoidal rule."""
    if len(y_true) != len(y_scores):
        raise ValueError("y_true and y_scores must have the same length")
    
    # Get unique labels and map to 0/1
    labels = sorted(set(y_true))
    if len(labels) != 2:
        raise ValueError("ROC AUC requires exactly 2 classes")
    
    neg_class, pos_class = labels[0], labels[1]
    y_binary = [1 if label == pos_class else 0 for label in y_true]
    
    # Sort by scores (descending)
    sorted_pairs = sorted(zip(y_scores, y_binary), key=lambda x: x[0], reverse=True)
    
    # Count positives and negatives
    n_pos = sum(y_binary)
    n_neg = len(y_binary) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.5  # No discrimination possible
    
    # Compute ROC curve points and AUC
    tp = fp = 0
    auc = 0.0
    prev_fp = 0
    
    for score, label in sorted_pairs:
        if label == 1:  # Positive
            tp += 1
        else:  # Negative
            fp += 1
            # Add trapezoid area when FP increases
            if tp > 0:
                auc += tp * (fp - prev_fp)
            prev_fp = fp
    
    # Normalize by total area
    auc /= (n_pos * n_neg)
    return auc


def precision_recall_curve(y_true: List[int], y_scores: List[float]) -> Tuple[List[float], List[float], List[float]]:
    """Compute precision-recall curve."""
    if len(y_true) != len(y_scores):
        raise ValueError("y_true and y_scores must have the same length")
    
    # Get unique labels and map to 0/1
    labels = sorted(set(y_true))
    if len(labels) != 2:
        raise ValueError("PR curve requires exactly 2 classes")
    
    neg_class, pos_class = labels[0], labels[1]
    y_binary = [1 if label == pos_class else 0 for label in y_true]
    
    # Sort by scores (descending)
    sorted_pairs = sorted(zip(y_scores, y_binary), key=lambda x: x[0], reverse=True)
    
    n_pos = sum(y_binary)
    if n_pos == 0:
        return [0.0], [0.0], [0.0]
    
    precisions = []
    recalls = []
    thresholds = []
    
    tp = fp = 0
    
    # Add initial point
    precisions.append(1.0)
    recalls.append(0.0)
    if sorted_pairs:
        thresholds.append(sorted_pairs[0][0] + 1.0)
    else:
        thresholds.append(1.0)
    
    for score, label in sorted_pairs:
        if label == 1:  # Positive
            tp += 1
        else:  # Negative
            fp += 1
        
        # Compute precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / n_pos
        
        precisions.append(precision)
        recalls.append(recall)
        thresholds.append(score)
    
    return precisions, recalls, thresholds


def roc_curve(y_true: List[int], y_scores: List[float]) -> Tuple[List[float], List[float], List[float]]:
    """Compute ROC curve."""
    if len(y_true) != len(y_scores):
        raise ValueError("y_true and y_scores must have the same length")
    
    # Get unique labels and map to 0/1
    labels = sorted(set(y_true))
    if len(labels) != 2:
        raise ValueError("ROC curve requires exactly 2 classes")
    
    neg_class, pos_class = labels[0], labels[1]
    y_binary = [1 if label == pos_class else 0 for label in y_true]
    
    # Sort by scores (descending)
    sorted_pairs = sorted(zip(y_scores, y_binary), key=lambda x: x[0], reverse=True)
    
    n_pos = sum(y_binary)
    n_neg = len(y_binary) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]
    
    fprs = []  # False Positive Rate
    tprs = []  # True Positive Rate
    thresholds = []
    
    tp = fp = 0
    
    # Add initial point (0, 0)
    fprs.append(0.0)
    tprs.append(0.0)
    if sorted_pairs:
        thresholds.append(sorted_pairs[0][0] + 1.0)
    else:
        thresholds.append(1.0)
    
    for score, label in sorted_pairs:
        if label == 1:  # Positive
            tp += 1
        else:  # Negative
            fp += 1
        
        # Compute FPR and TPR
        fpr = fp / n_neg
        tpr = tp / n_pos
        
        fprs.append(fpr)
        tprs.append(tpr)
        thresholds.append(score)
    
    return fprs, tprs, thresholds


def classification_report(y_true: List[int], y_pred: List[int], y_scores: List[float] = None) -> Dict[str, Any]:
    """Generate a comprehensive classification report."""
    # Basic metrics
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'confusion_matrix': cm,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
    
    # Add AUC if scores provided
    if y_scores is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
        except Exception:
            metrics['roc_auc'] = None
    
    # Class-specific metrics
    pos_class = cm['positive_class']
    neg_class = cm['negative_class']
    
    metrics['class_metrics'] = {
        str(neg_class): {
            'precision': cm['tn'] / (cm['tn'] + cm['fn']) if (cm['tn'] + cm['fn']) > 0 else 0.0,
            'recall': cm['tn'] / (cm['tn'] + cm['fp']) if (cm['tn'] + cm['fp']) > 0 else 0.0,
            'support': cm['tn'] + cm['fp']
        },
        str(pos_class): {
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'support': cm['tp'] + cm['fn']
        }
    }
    
    return metrics


def pr_auc_score(y_true: List[int], y_scores: List[float]) -> float:
    """Compute Area Under Precision-Recall curve."""
    precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
    
    # Use trapezoidal rule to compute area
    auc = 0.0
    for i in range(1, len(recalls)):
        # Width of trapezoid
        width = recalls[i] - recalls[i-1]
        # Average height
        height = (precisions[i] + precisions[i-1]) / 2
        auc += width * height
    
    return auc


def log_loss(y_true: List[int], y_proba: List[List[float]]) -> float:
    """Compute logistic loss (cross-entropy loss)."""
    if len(y_true) != len(y_proba):
        raise ValueError("y_true and y_proba must have the same length")
    
    # Get unique labels
    labels = sorted(set(y_true))
    if len(labels) != 2:
        raise ValueError("Log loss requires exactly 2 classes")
    
    neg_class, pos_class = labels[0], labels[1]
    
    total_loss = 0.0
    epsilon = 1e-15  # Small value to prevent log(0)
    
    for true_label, proba in zip(y_true, y_proba):
        if len(proba) != 2:
            raise ValueError("Probability array must have 2 elements for binary classification")
        
        # Map true label to index (0 for neg_class, 1 for pos_class)
        true_idx = 1 if true_label == pos_class else 0
        
        # Clip probabilities to prevent log(0)
        clipped_proba = [max(epsilon, min(1 - epsilon, p)) for p in proba]
        
        # Compute cross-entropy loss
        loss = -math.log(clipped_proba[true_idx])
        total_loss += loss
    
    return total_loss / len(y_true)