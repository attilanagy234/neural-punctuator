import logging
import sys
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, precision_score, recall_score
import numpy as np

from neural_punctuator.utils.visualize import plot_confusion_matrix

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-9s %(message)s'))

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(handler)


def get_eval_metrics(targets, preds, config):
    # TODO: get the desired metric list from config-frozen.yaml
    """
    Calculates metrics on validation data
    """
    metrics = {}

    preds = np.exp(preds)
    preds = preds.reshape(-1, config.model.num_classes)
    targets = targets.reshape(-1)
    pred_index = preds.argmax(-1)

    # One-hot encode targets
    # metric_targets = np.zeros((targets.size, config.model.num_classes))
    # metric_targets[np.arange(targets.size), targets] = 1

    cls_report, cls_report_print = get_classification_report(targets, pred_index, config)
    print(cls_report_print)
    metrics['cls_report'] = cls_report


    macro_precision = precision_score(targets, pred_index, average='macro')
    log.info(f'Macro precision is: {macro_precision}')
    metrics['precision'] = macro_precision

    macro_recall = recall_score(targets, pred_index, average='macro')
    log.info(f'Macro recall is {macro_recall}')
    metrics['recall'] = macro_recall

    macro_f1_score = f1_score(targets, pred_index, average='macro')
    log.info(f'Macro f-score is {macro_f1_score}')
    metrics['f_score'] = macro_f1_score

    auc_score = roc_auc_score(targets, preds, average='macro', multi_class='ovo')
    log.info(f'AUC is: {auc_score}')
    metrics['auc'] = auc_score


    conf_mx = get_confusion_mx(targets, pred_index)
    if config.trainer.show_confusion_matrix:
        plot_confusion_matrix(conf_mx, config.data.output_labels)

    return metrics


def get_classification_report(target, preds, config):
    report = classification_report(target, preds, output_dict=True, target_names=config.data.output_labels)
    report_print = classification_report(target, preds, digits=3, target_names=config.data.output_labels)
    return report, report_print


def get_confusion_mx(target, preds):
    return confusion_matrix(target, preds)


def get_total_grad_norm(parameters, norm_type=2):
    total_norm = 0
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    return total_norm ** (1. / norm_type)