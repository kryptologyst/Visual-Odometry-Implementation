"""Evaluation metrics for Visual Odometry algorithms."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from .core import Pose

logger = logging.getLogger(__name__)


class VisualOdometryMetrics:
    """Comprehensive evaluation metrics for Visual Odometry."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.estimated_trajectory: List[Pose] = []
        self.ground_truth_trajectory: List[Pose] = []
        self.timestamps: List[float] = []
    
    def add_pose(self, estimated: Pose, ground_truth: Optional[Pose] = None, timestamp: Optional[float] = None):
        """Add a pose to the trajectory.
        
        Args:
            estimated: Estimated pose
            ground_truth: Ground truth pose (optional)
            timestamp: Timestamp (optional)
        """
        self.estimated_trajectory.append(estimated)
        if ground_truth is not None:
            self.ground_truth_trajectory.append(ground_truth)
        if timestamp is not None:
            self.timestamps.append(timestamp)
    
    def set_trajectories(
        self, 
        estimated: List[Pose], 
        ground_truth: Optional[List[Pose]] = None,
        timestamps: Optional[List[float]] = None
    ):
        """Set complete trajectories.
        
        Args:
            estimated: List of estimated poses
            ground_truth: List of ground truth poses (optional)
            timestamps: List of timestamps (optional)
        """
        self.estimated_trajectory = estimated.copy()
        if ground_truth is not None:
            self.ground_truth_trajectory = ground_truth.copy()
        if timestamps is not None:
            self.timestamps = timestamps.copy()
    
    def compute_ate(self, align_trajectories: bool = True) -> Dict[str, float]:
        """Compute Absolute Trajectory Error (ATE).
        
        Args:
            align_trajectories: Whether to align trajectories using Umeyama algorithm
            
        Returns:
            Dictionary containing ATE metrics
        """
        if len(self.ground_truth_trajectory) == 0:
            logger.warning("No ground truth trajectory available for ATE computation")
            return {}
        
        # Extract positions
        est_positions = np.array([pose.position for pose in self.estimated_trajectory])
        gt_positions = np.array([pose.position for pose in self.ground_truth_trajectory])
        
        # Align trajectories if requested
        if align_trajectories:
            est_positions = self._align_trajectories(est_positions, gt_positions)
        
        # Compute ATE
        errors = np.linalg.norm(est_positions - gt_positions, axis=1)
        
        return {
            'ate_mean': float(np.mean(errors)),
            'ate_std': float(np.std(errors)),
            'ate_rmse': float(np.sqrt(np.mean(errors**2))),
            'ate_max': float(np.max(errors)),
            'ate_median': float(np.median(errors)),
        }
    
    def compute_rpe(self, delta: float = 1.0) -> Dict[str, float]:
        """Compute Relative Pose Error (RPE).
        
        Args:
            delta: Time interval for RPE computation
            
        Returns:
            Dictionary containing RPE metrics
        """
        if len(self.ground_truth_trajectory) == 0:
            logger.warning("No ground truth trajectory available for RPE computation")
            return {}
        
        # Compute RPE for translation
        trans_errors = []
        rot_errors = []
        
        for i in range(len(self.estimated_trajectory) - int(delta)):
            j = i + int(delta)
            
            # Estimated relative pose
            est_rel = self.estimated_trajectory[i].inverse().compose(self.estimated_trajectory[j])
            
            # Ground truth relative pose
            gt_rel = self.ground_truth_trajectory[i].inverse().compose(self.ground_truth_trajectory[j])
            
            # Translation error
            trans_error = np.linalg.norm(est_rel.position - gt_rel.position)
            trans_errors.append(trans_error)
            
            # Rotation error (angle between rotation matrices)
            rot_error = self._rotation_error(est_rel.rotation_matrix, gt_rel.rotation_matrix)
            rot_errors.append(rot_error)
        
        trans_errors = np.array(trans_errors)
        rot_errors = np.array(rot_errors)
        
        return {
            'rpe_trans_mean': float(np.mean(trans_errors)),
            'rpe_trans_std': float(np.std(trans_errors)),
            'rpe_trans_rmse': float(np.sqrt(np.mean(trans_errors**2))),
            'rpe_rot_mean': float(np.mean(rot_errors)),
            'rpe_rot_std': float(np.std(rot_errors)),
            'rpe_rot_rmse': float(np.sqrt(np.mean(rot_errors**2))),
        }
    
    def compute_drift(self) -> Dict[str, float]:
        """Compute trajectory drift metrics.
        
        Returns:
            Dictionary containing drift metrics
        """
        if len(self.ground_truth_trajectory) == 0:
            logger.warning("No ground truth trajectory available for drift computation")
            return {}
        
        # Extract positions
        est_positions = np.array([pose.position for pose in self.estimated_trajectory])
        gt_positions = np.array([pose.position for pose in self.ground_truth_trajectory])
        
        # Align trajectories
        est_positions = self._align_trajectories(est_positions, gt_positions)
        
        # Compute drift over time
        drift_rates = []
        for i in range(1, len(est_positions)):
            est_drift = np.linalg.norm(est_positions[i] - est_positions[0])
            gt_drift = np.linalg.norm(gt_positions[i] - gt_positions[0])
            drift_rate = abs(est_drift - gt_drift) / gt_drift if gt_drift > 0 else 0
            drift_rates.append(drift_rate)
        
        drift_rates = np.array(drift_rates)
        
        return {
            'drift_mean': float(np.mean(drift_rates)),
            'drift_std': float(np.std(drift_rates)),
            'drift_max': float(np.max(drift_rates)),
            'drift_final': float(drift_rates[-1]) if len(drift_rates) > 0 else 0.0,
        }
    
    def compute_trajectory_length(self) -> Dict[str, float]:
        """Compute trajectory length metrics.
        
        Returns:
            Dictionary containing trajectory length metrics
        """
        if len(self.estimated_trajectory) < 2:
            return {'length': 0.0}
        
        # Compute estimated trajectory length
        est_positions = np.array([pose.position for pose in self.estimated_trajectory])
        est_length = np.sum(np.linalg.norm(np.diff(est_positions, axis=0), axis=1))
        
        metrics = {'estimated_length': float(est_length)}
        
        if len(self.ground_truth_trajectory) >= 2:
            # Compute ground truth trajectory length
            gt_positions = np.array([pose.position for pose in self.ground_truth_trajectory])
            gt_length = np.sum(np.linalg.norm(np.diff(gt_positions, axis=0), axis=1))
            
            metrics.update({
                'ground_truth_length': float(gt_length),
                'length_ratio': float(est_length / gt_length) if gt_length > 0 else 0.0,
            })
        
        return metrics
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """Compute all available metrics.
        
        Returns:
            Dictionary containing all computed metrics
        """
        metrics = {}
        
        # ATE metrics
        ate_metrics = self.compute_ate()
        metrics.update(ate_metrics)
        
        # RPE metrics
        rpe_metrics = self.compute_rpe()
        metrics.update(rpe_metrics)
        
        # Drift metrics
        drift_metrics = self.compute_drift()
        metrics.update(drift_metrics)
        
        # Trajectory length metrics
        length_metrics = self.compute_trajectory_length()
        metrics.update(length_metrics)
        
        return metrics
    
    def _align_trajectories(self, est_positions: np.ndarray, gt_positions: np.ndarray) -> np.ndarray:
        """Align estimated trajectory to ground truth using Umeyama algorithm.
        
        Args:
            est_positions: Estimated trajectory positions
            gt_positions: Ground truth trajectory positions
            
        Returns:
            Aligned estimated positions
        """
        if len(est_positions) != len(gt_positions):
            min_len = min(len(est_positions), len(gt_positions))
            est_positions = est_positions[:min_len]
            gt_positions = gt_positions[:min_len]
        
        # Compute centroids
        est_centroid = np.mean(est_positions, axis=0)
        gt_centroid = np.mean(gt_positions, axis=0)
        
        # Center trajectories
        est_centered = est_positions - est_centroid
        gt_centered = gt_positions - gt_centroid
        
        # Compute rotation matrix using SVD
        H = est_centered.T @ gt_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure proper rotation matrix (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Apply transformation
        aligned_positions = (R @ est_centered.T).T + gt_centroid
        
        return aligned_positions
    
    def _rotation_error(self, R1: np.ndarray, R2: np.ndarray) -> float:
        """Compute rotation error between two rotation matrices.
        
        Args:
            R1: First rotation matrix
            R2: Second rotation matrix
            
        Returns:
            Rotation error in radians
        """
        # Compute relative rotation
        R_rel = R1.T @ R2
        
        # Convert to axis-angle representation
        trace = np.trace(R_rel)
        trace = np.clip(trace, -1.0, 3.0)
        
        angle = np.arccos((trace - 1.0) / 2.0)
        
        return angle


class VisualOdometryLeaderboard:
    """Leaderboard for comparing Visual Odometry algorithms."""
    
    def __init__(self):
        """Initialize leaderboard."""
        self.results: Dict[str, Dict[str, float]] = {}
    
    def add_result(self, algorithm: str, metrics: Dict[str, float]):
        """Add algorithm results to leaderboard.
        
        Args:
            algorithm: Algorithm name
            metrics: Computed metrics
        """
        self.results[algorithm] = metrics.copy()
    
    def get_leaderboard(self, metric: str = 'ate_rmse', ascending: bool = True) -> List[Tuple[str, float]]:
        """Get leaderboard sorted by specified metric.
        
        Args:
            metric: Metric to sort by
            ascending: Whether to sort in ascending order
            
        Returns:
            List of (algorithm, metric_value) tuples
        """
        if not self.results:
            return []
        
        # Filter algorithms that have the requested metric
        valid_results = {
            alg: metrics[metric] 
            for alg, metrics in self.results.items() 
            if metric in metrics
        }
        
        # Sort by metric value
        sorted_results = sorted(valid_results.items(), key=lambda x: x[1], reverse=not ascending)
        
        return sorted_results
    
    def print_leaderboard(self, metric: str = 'ate_rmse', ascending: bool = True):
        """Print formatted leaderboard.
        
        Args:
            metric: Metric to sort by
            ascending: Whether to sort in ascending order
        """
        leaderboard = self.get_leaderboard(metric, ascending)
        
        if not leaderboard:
            print(f"No results available for metric: {metric}")
            return
        
        print(f"\nVisual Odometry Leaderboard ({metric})")
        print("=" * 50)
        print(f"{'Rank':<4} {'Algorithm':<20} {'Value':<10}")
        print("-" * 50)
        
        for i, (algorithm, value) in enumerate(leaderboard, 1):
            print(f"{i:<4} {algorithm:<20} {value:<10.6f}")
    
    def export_results(self, filename: str):
        """Export results to CSV file.
        
        Args:
            filename: Output filename
        """
        import pandas as pd
        
        if not self.results:
            logger.warning("No results to export")
            return
        
        df = pd.DataFrame(self.results).T
        df.to_csv(filename)
        logger.info(f"Results exported to {filename}")
