import logging
import sys
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, precision_score, recall_score


from src.neural_punctuator.utils.visualize import plot_confusion_matrix

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-9s %(message)s'))

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(handler)


def get_eval_metrics(preds, target, config=None):
    # TODO: get the desired metric list from config.yaml
    """
    Calculates metrics on validation data
    """
    metrics = {}

    cls_report = get_classification_report(preds)

    print(cls_report)

    # if 'precision' in self._config.trainer.metrics:
    if True:
        macro_precision = precision_score(target, preds, average='macro')
        log.info(f'Macro precision is: {macro_precision}')
        metrics['precision'] = macro_precision
    # if 'recall' in self._config.trainer.metrics:
    if True:
        macro_recall = recall_score(target, preds, average='macro')
        log.info(f'Macro recall is {macro_recall}')
        metrics['recall'] = macro_recall
    # if 'f_score' in self._config.trainer.metrics:
    if True:
        macro_f1_score = f1_score(target, preds, average='macro')
        log.info(f'Macro f-score is {macro_f1_score}')
        metrics['f_score'] = macro_f1_score
    # if 'auc' in self._config.trainer.metrics:
    if True:
        auc_score = roc_auc_score(target, preds, average='macro', multi_class='ovo')
        log.info(f'AUC is: {auc_score}')
        metrics['auc'] = auc_score

    # if self._config.trainer.visualize_conf_mx:
    if True:
        conf_mx = get_confusion_mx(target, preds)
        plot_confusion_matrix(conf_mx)

    return metrics


def get_classification_report(target, preds):
    return classification_report(target, preds, digits=3)


def get_confusion_mx(target, preds):
    return confusion_matrix(target, preds)
