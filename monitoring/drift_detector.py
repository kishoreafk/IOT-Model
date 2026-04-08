import numpy as np
from typing import Dict, Any, Tuple, Optional
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
from .metrics_collector import collector


class DriftDetector:
    """Statistical drift detection for embeddings, confidence, and clusters."""

    def __init__(
        self,
        mmd_threshold: float = 0.1,
        ks_threshold: float = 0.1,
        chi_sq_threshold: float = 0.05,
    ):
        self.mmd_threshold = mmd_threshold
        self.ks_threshold = ks_threshold
        self.chi_sq_threshold = chi_sq_threshold

        self.reference_embeddings: Optional[np.ndarray] = None
        self.reference_confidences: Optional[np.ndarray] = None
        self.reference_cluster_counts: Optional[np.ndarray] = None

    def set_reference(
        self,
        embeddings: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        cluster_counts: Optional[np.ndarray] = None,
    ):
        """Set reference distribution for drift detection."""
        self.reference_embeddings = embeddings.copy()
        if confidences is not None:
            self.reference_confidences = confidences.copy()
        if cluster_counts is not None:
            self.reference_cluster_counts = cluster_counts.copy()

    def compute_mmd(self, current_embeddings: np.ndarray) -> float:
        """
        Compute Maximum Mean Discrepancy between reference and current embeddings.

        Returns:
            MMD score
        """
        if self.reference_embeddings is None:
            return 0.0

        if len(current_embeddings) == 0:
            return 0.0

        ref_mean = np.mean(self.reference_embeddings, axis=0)
        curr_mean = np.mean(current_embeddings, axis=0)

        mmd = np.linalg.norm(ref_mean - curr_mean)

        collector.set("mmd_drift", mmd)

        return mmd

    def compute_ks_drift(self, current_confidences: np.ndarray) -> float:
        """
        Compute Kolmogorov-Smirnov test for confidence drift.

        Returns:
            KS statistic
        """
        if self.reference_confidences is None or len(current_confidences) == 0:
            return 0.0

        if len(self.reference_confidences) < 2 or len(current_confidences) < 2:
            return 0.0

        ks_stat, _ = ks_2samp(self.reference_confidences, current_confidences)

        collector.set("ks_drift", ks_stat)

        return ks_stat

    def compute_chi_sq_drift(self, current_cluster_counts: np.ndarray) -> float:
        """
        Compute chi-squared test for cluster distribution drift.

        Returns:
            Chi-squared statistic
        """
        if self.reference_cluster_counts is None or len(current_cluster_counts) == 0:
            return 0.0

        if len(self.reference_cluster_counts) != len(current_cluster_counts):
            max_len = max(len(self.reference_cluster_counts), len(current_cluster_counts))
            ref = np.pad(self.reference_cluster_counts, (0, max_len - len(self.reference_cluster_counts)))
            curr = np.pad(current_cluster_counts, (0, max_len - len(current_cluster_counts)))
        else:
            ref = self.reference_cluster_counts
            curr = current_cluster_counts

        contingency = np.array([ref, curr])
        chi2, _, _, _ = chi2_contingency(contingency)

        normalized_chi2 = chi2 / (np.sum(ref) + np.sum(curr) + 1e-10)

        collector.set("chi_sq_drift", normalized_chi2)

        return normalized_chi2

    def detect_all_drifts(
        self,
        current_embeddings: Optional[np.ndarray] = None,
        current_confidences: Optional[np.ndarray] = None,
        current_cluster_counts: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Run all drift detection methods.

        Returns:
            Dictionary with drift detection results
        """
        results = {
            "embedding_drift": False,
            "confidence_drift": False,
            "cluster_drift": False,
            "overall_drift": False,
        }

        if current_embeddings is not None:
            mmd = self.compute_mmd(current_embeddings)
            results["mmd_score"] = mmd
            results["embedding_drift"] = mmd > self.mmd_threshold

        if current_confidences is not None:
            ks = self.compute_ks_drift(current_confidences)
            results["ks_score"] = ks
            results["confidence_drift"] = ks > self.ks_threshold

        if current_cluster_counts is not None:
            chi_sq = self.compute_chi_sq_drift(current_cluster_counts)
            results["chi_sq_score"] = chi_sq
            results["cluster_drift"] = chi_sq > self.chi_sq_threshold

        results["overall_drift"] = (
            results["embedding_drift"] or results["confidence_drift"] or results["cluster_drift"]
        )

        return results

    def get_status(self) -> Dict[str, Any]:
        """Get drift detector status."""
        return {
            "mmd_threshold": self.mmd_threshold,
            "ks_threshold": self.ks_threshold,
            "chi_sq_threshold": self.chi_sq_threshold,
            "reference_embeddings_count": len(self.reference_embeddings) if self.reference_embeddings is not None else 0,
            "reference_confidences_count": len(self.reference_confidences) if self.reference_confidences is not None else 0,
        }

    def reset(self):
        """Reset reference data."""
        self.reference_embeddings = None
        self.reference_confidences = None
        self.reference_cluster_counts = None


drift_detector = DriftDetector()