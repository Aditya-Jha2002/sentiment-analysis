from src.utils import Utils
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, log_loss

class Metrics:
    """
    Metrics class to calculate the accuracy, precision, recall and f1 score
    Example:
        metrics = Metrics()
        metrics.calculate_metrics(y_true, y_pred)
    """

    def __init__(self, config_path):
        config = Utils().read_params(config_path)

        self.metrics = config["base"]["metrics"]

    def _eval_metrics(self, actual, pred, pred_proba):
        """ Takes in the ground truth labels, predictions labels, and prediction probabilities.
            Returns the accuracy, f1, auc_roc, log_loss scores.
        """
        metrics_dict = {}
        
        if "accuracy" in self.metrics:
            accuracy = accuracy_score(actual, pred)
            metrics_dict["accuracy"] = accuracy

        if "f1" in self.metrics:
            f1 = f1_score(actual, pred)
            metrics_dict["f1"] = f1
        
        if "precision" in self.metrics:
            precision = precision_score(actual, pred)
            metrics_dict["precision"] = precision
        
        if "recall" in self.metrics:
            recall = recall_score(actual, pred)
            metrics_dict["recall"] = recall

        if "roc_auc" in self.metrics:
            roc_auc = roc_auc_score(actual, pred_proba[:, 1])
            metrics_dict["roc_auc"] = roc_auc

        if "log_loss" in self.metrics:
            log_loss_score = log_loss(actual, pred_proba)
            metrics_dict["log_loss"] = log_loss_score

        return metrics_dict