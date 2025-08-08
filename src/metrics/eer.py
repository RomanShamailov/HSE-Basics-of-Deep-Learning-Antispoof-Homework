import numpy as np
import torch
from torch import nn

from src.metrics.base_metric import BaseMetric


class EqualErrorRate(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        """
        Equal error rate class. Applies the equal error rate metric
        function object (for example, from TorchMetrics) on tensors.

        Args:
            metric (Callable): function to calculate metrics.
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # for per-epoch calculation
        self.scores = []
        self.labels = []

    def _compute_det_curve(self, target_scores, nontarget_scores):
        n_scores = target_scores.size + nontarget_scores.size
        all_scores = np.concatenate((target_scores, nontarget_scores))
        labels = np.concatenate(
            (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
        )

        # Sort labels based on scores
        indices = np.argsort(all_scores, kind="mergesort")
        labels = labels[indices]

        # Compute false rejection and false acceptance rates
        tar_trial_sums = np.cumsum(labels)
        nontarget_trial_sums = nontarget_scores.size - (
            np.arange(1, n_scores + 1) - tar_trial_sums
        )

        # false rejection rates
        frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
        far = np.concatenate(
            (np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size)
        )  # false acceptance rates
        # Thresholds are the sorted scores
        thresholds = np.concatenate(
            (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
        )

        return frr, far, thresholds

    def _compute_eer(self, bonafide_scores, other_scores):
        """
        Returns equal error rate (EER) and the corresponding threshold.
        """
        frr, far, thresholds = self._compute_det_curve(bonafide_scores, other_scores)
        abs_diffs = np.abs(frr - far)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((frr[min_index], far[min_index]))
        return eer, thresholds[min_index]

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        """
        Used for storing the global logits and labels inside the metric.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            metric (float): calculated metric.
        """
        self.scores += torch.nn.functional.softmax(logits, dim=1)[:, 1].tolist()
        self.labels += labels.tolist()

    def calculate_epoch(self):
        bonafide_scores = [
            self.scores[i] for i in range(len(self.scores)) if self.labels[i] == 1
        ]
        other_scores = [
            self.scores[i] for i in range(len(self.scores)) if self.labels[i] == 0
        ]
        eer, threshold = self._compute_eer(
            np.array(bonafide_scores), np.array(other_scores)
        )
        self.scores.clear()
        self.labels.clear()
        return eer * 100
