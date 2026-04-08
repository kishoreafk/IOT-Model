import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from .metrics_collector import collector


class ModelPerformanceMonitor:
    """Monitor model performance metrics including accuracy, F1, and calibration."""

    def __init__(self):
        self.predictions: List[int] = []
        self.labels: List[int] = []
        self.confidences: List[float] = []
        self.num_classes = 50

    def update(self, predictions: List[int], labels: List[int], confidences: Optional[List[float]] = None):
        """Update with new predictions and labels."""
        self.predictions.extend(predictions)
        self.labels.extend(labels)
        if confidences:
            self.confidences.extend(confidences)

    def compute_metrics(self) -> Dict[str, float]:
        """Compute all performance metrics."""
        if not self.predictions:
            return {}

        accuracy = accuracy_score(self.labels, self.predictions)

        precision = precision_score(self.labels, self.predictions, average="weighted", zero_division=0)
        recall = recall_score(self.labels, self.predictions, average="weighted", zero_division=0)
        f1 = f1_score(self.labels, self.predictions, average="weighted", zero_division=0)

        ece, mce = self._compute_calibration_error()

        collector.set("accuracy", accuracy)
        collector.set("precision", precision)
        collector.set("recall", recall)
        collector.set("f1_score", f1)
        collector.set("ece", ece)
        collector.set("mce", mce)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ece": ece,
            "mce": mce,
        }

    def _compute_calibration_error(self) -> Tuple[float, float]:
        """Compute Expected Calibration Error and Maximum Calibration Error."""
        if not self.confidences or len(self.confidences) < 10:
            return 0.0, 0.0

        confidences = np.array(self.confidences)
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)

        correct = (predictions == labels).astype(int)

        bins = np.linspace(0, 1, 11)
        ece = 0.0
        mce = 0.0

        for i in range(len(bins) - 1):
            bin_mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
            if np.sum(bin_mask) > 0:
                bin_acc = np.mean(correct[bin_mask])
                bin_conf = np.mean(confidences[bin_mask])
                ece += (np.sum(bin_mask) / len(confidences)) * abs(bin_acc - bin_conf)
                mce = max(mce, abs(bin_acc - bin_conf))

        return ece, mce

    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        if not self.predictions:
            return np.zeros((self.num_classes, self.num_classes))
        return confusion_matrix(self.labels, self.predictions, labels=range(self.num_classes))

    def get_per_class_metrics(self) -> Dict[int, Dict[str, float]]:
        """Get per-class precision, recall, F1."""
        if not self.predictions:
            return {}

        precision = precision_score(self.labels, self.predictions, average=None, zero_division=0)
        recall = recall_score(self.labels, self.predictions, average=None, zero_division=0)
        f1 = f1_score(self.labels, self.predictions, average=None, zero_division=0)

        metrics = {}
        for i in range(len(precision)):
            metrics[i] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
            }

        return metrics

    def reset(self):
        """Reset stored predictions and labels."""
        self.predictions.clear()
        self.labels.clear()
        self.confidences.clear()


class AdaptationMonitor:
    """Monitor LoRA adaptation performance."""

    def __init__(self):
        self.pre_adaptation_accuracy: List[float] = []
        self.post_adaptation_accuracy: List[float] = []
        self.training_losses: List[float] = []
        self.validation_losses: List[float] = []

    def update(self, pre_acc: float, post_acc: float, train_loss: float, val_loss: float):
        """Update with adaptation metrics."""
        self.pre_adaptation_accuracy.append(pre_acc)
        self.post_adaptation_accuracy.append(post_acc)
        self.training_losses.append(train_loss)
        self.validation_losses.append(val_loss)

        delta = post_acc - pre_acc
        collector.set("lora_adaptation_delta", delta)
        collector.set("training_loss", train_loss)
        collector.set("validation_loss", val_loss)

    def get_summary(self) -> Dict[str, Any]:
        """Get adaptation summary."""
        if not self.pre_adaptation_accuracy:
            return {}

        return {
            "avg_pre_accuracy": np.mean(self.pre_adaptation_accuracy),
            "avg_post_accuracy": np.mean(self.post_adaptation_accuracy),
            "avg_delta": np.mean([p - o for p, o in zip(self.post_adaptation_accuracy, self.pre_adaptation_accuracy)]),
            "avg_training_loss": np.mean(self.training_losses),
            "avg_validation_loss": np.mean(self.validation_losses),
            "overfitting_gap": np.mean([t - v for t, v in zip(self.training_losses, self.validation_losses)]),
        }

    def reset(self):
        """Reset adaptation metrics."""
        self.pre_adaptation_accuracy.clear()
        self.post_adaptation_accuracy.clear()
        self.training_losses.clear()
        self.validation_losses.clear()