"""Core Visual Odometry algorithms and utilities."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


class VisualOdometryConfig:
    """Configuration class for Visual Odometry algorithms."""
    
    def __init__(
        self,
        feature_detector: str = "ORB",
        max_features: int = 1000,
        match_ratio: float = 0.7,
        ransac_threshold: float = 1.0,
        ransac_prob: float = 0.999,
        min_matches: int = 50,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
    ):
        """Initialize Visual Odometry configuration.
        
        Args:
            feature_detector: Type of feature detector ('ORB', 'SIFT', 'SURF')
            max_features: Maximum number of features to detect
            match_ratio: Ratio for feature matching (Lowe's ratio test)
            ransac_threshold: RANSAC threshold for essential matrix estimation
            ransac_prob: RANSAC probability parameter
            min_matches: Minimum number of matches required for pose estimation
            camera_matrix: Camera intrinsic matrix (3x3)
            dist_coeffs: Camera distortion coefficients
        """
        self.feature_detector = feature_detector
        self.max_features = max_features
        self.match_ratio = match_ratio
        self.ransac_threshold = ransac_threshold
        self.ransac_prob = ransac_prob
        self.min_matches = min_matches
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        # Default camera matrix (will be overridden with real calibration)
        if self.camera_matrix is None:
            self.camera_matrix = np.array([
                [525.0, 0.0, 320.0],
                [0.0, 525.0, 240.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
        
        if self.dist_coeffs is None:
            self.dist_coeffs = np.zeros(5, dtype=np.float32)


class Pose:
    """Represents a 6DOF pose (position + orientation)."""
    
    def __init__(self, position: np.ndarray, rotation: Union[np.ndarray, Rotation]):
        """Initialize pose.
        
        Args:
            position: 3D position vector (x, y, z)
            rotation: Rotation matrix (3x3) or scipy Rotation object
        """
        self.position = np.array(position, dtype=np.float64)
        
        if isinstance(rotation, Rotation):
            self.rotation = rotation
            self.rotation_matrix = rotation.as_matrix()
        else:
            self.rotation_matrix = np.array(rotation, dtype=np.float64)
            self.rotation = Rotation.from_matrix(self.rotation_matrix)
    
    @property
    def translation(self) -> np.ndarray:
        """Get translation vector."""
        return self.position
    
    @property
    def quaternion(self) -> np.ndarray:
        """Get quaternion representation (w, x, y, z)."""
        return self.rotation.as_quat()
    
    @property
    def euler_angles(self) -> np.ndarray:
        """Get Euler angles (roll, pitch, yaw) in radians."""
        return self.rotation.as_euler('xyz')
    
    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """Transform a 3D point by this pose."""
        return self.rotation_matrix @ point + self.position
    
    def inverse(self) -> Pose:
        """Get inverse pose."""
        inv_rotation = self.rotation.inv()
        inv_position = -inv_rotation.as_matrix() @ self.position
        return Pose(inv_position, inv_rotation)
    
    def compose(self, other: Pose) -> Pose:
        """Compose this pose with another pose."""
        new_rotation = self.rotation * other.rotation
        new_position = self.position + self.rotation_matrix @ other.position
        return Pose(new_position, new_rotation)
    
    def __repr__(self) -> str:
        return f"Pose(pos={self.position}, euler={self.euler_angles})"


class VisualOdometryAlgorithm(ABC):
    """Abstract base class for Visual Odometry algorithms."""
    
    def __init__(self, config: VisualOdometryConfig):
        """Initialize the algorithm with configuration."""
        self.config = config
        self.feature_detector = self._create_feature_detector()
        self.matcher = self._create_matcher()
        self.current_pose = Pose(np.zeros(3), np.eye(3))
        self.trajectory: List[Pose] = [self.current_pose]
        
    @abstractmethod
    def _create_feature_detector(self):
        """Create the feature detector."""
        pass
    
    @abstractmethod
    def _create_matcher(self):
        """Create the feature matcher."""
        pass
    
    @abstractmethod
    def _estimate_pose(
        self, 
        kp1: List[cv2.KeyPoint], 
        kp2: List[cv2.KeyPoint], 
        matches: List[cv2.DMatch]
    ) -> Tuple[Optional[Pose], int]:
        """Estimate pose from feature matches."""
        pass
    
    def process_frame(self, frame: np.ndarray) -> Optional[Pose]:
        """Process a new frame and estimate pose.
        
        Args:
            frame: Input image frame
            
        Returns:
            Estimated pose relative to previous frame, or None if estimation failed
        """
        if not hasattr(self, 'prev_frame'):
            self.prev_frame = frame
            self.prev_kp, self.prev_des = self.feature_detector.detectAndCompute(frame, None)
            return None
        
        # Detect features in current frame
        curr_kp, curr_des = self.feature_detector.detectAndCompute(frame, None)
        
        if curr_des is None or self.prev_des is None:
            logger.warning("No features detected in frame")
            self.prev_frame = frame
            self.prev_kp, self.prev_des = curr_kp, curr_des
            return None
        
        # Match features
        matches = self.matcher.knnMatch(self.prev_des, curr_des, k=2)
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.config.match_ratio * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < self.config.min_matches:
            logger.warning(f"Insufficient matches: {len(good_matches)} < {self.config.min_matches}")
            self.prev_frame = frame
            self.prev_kp, self.prev_des = curr_kp, curr_des
            return None
        
        # Estimate pose
        pose_delta, num_inliers = self._estimate_pose(self.prev_kp, curr_kp, good_matches)
        
        if pose_delta is None:
            logger.warning("Pose estimation failed")
            self.prev_frame = frame
            self.prev_kp, self.prev_des = curr_kp, curr_des
            return None
        
        # Update trajectory
        self.current_pose = self.current_pose.compose(pose_delta)
        self.trajectory.append(self.current_pose)
        
        # Update previous frame data
        self.prev_frame = frame
        self.prev_kp, self.prev_des = curr_kp, curr_des
        
        logger.info(f"Pose estimated with {num_inliers} inliers")
        return pose_delta
    
    def get_trajectory(self) -> List[Pose]:
        """Get the estimated trajectory."""
        return self.trajectory.copy()
    
    def reset(self):
        """Reset the algorithm state."""
        self.current_pose = Pose(np.zeros(3), np.eye(3))
        self.trajectory = [self.current_pose]
        if hasattr(self, 'prev_frame'):
            delattr(self, 'prev_frame')
            delattr(self, 'prev_kp')
            delattr(self, 'prev_des')


class ORBVisualOdometry(VisualOdometryAlgorithm):
    """ORB-based Visual Odometry implementation."""
    
    def _create_feature_detector(self):
        """Create ORB feature detector."""
        return cv2.ORB_create(nfeatures=self.config.max_features)
    
    def _create_matcher(self):
        """Create brute force matcher for ORB."""
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def _estimate_pose(
        self, 
        kp1: List[cv2.KeyPoint], 
        kp2: List[cv2.KeyPoint], 
        matches: List[cv2.DMatch]
    ) -> Tuple[Optional[Pose], int]:
        """Estimate pose using ORB features and essential matrix."""
        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2, 
            cameraMatrix=self.config.camera_matrix,
            method=cv2.RANSAC,
            prob=self.config.ransac_prob,
            threshold=self.config.ransac_threshold
        )
        
        if E is None:
            return None, 0
        
        # Recover pose from essential matrix
        _, R, t, mask = cv2.recoverPose(
            E, pts1, pts2, 
            cameraMatrix=self.config.camera_matrix
        )
        
        num_inliers = np.sum(mask)
        
        if num_inliers < self.config.min_matches:
            return None, num_inliers
        
        pose = Pose(t.flatten(), R)
        return pose, num_inliers


class SIFTVisualOdometry(VisualOdometryAlgorithm):
    """SIFT-based Visual Odometry implementation."""
    
    def _create_feature_detector(self):
        """Create SIFT feature detector."""
        return cv2.SIFT_create(nfeatures=self.config.max_features)
    
    def _create_matcher(self):
        """Create FLANN matcher for SIFT."""
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        return cv2.FlannBasedMatcher(index_params, search_params)
    
    def _estimate_pose(
        self, 
        kp1: List[cv2.KeyPoint], 
        kp2: List[cv2.KeyPoint], 
        matches: List[cv2.DMatch]
    ) -> Tuple[Optional[Pose], int]:
        """Estimate pose using SIFT features and essential matrix."""
        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2, 
            cameraMatrix=self.config.camera_matrix,
            method=cv2.RANSAC,
            prob=self.config.ransac_prob,
            threshold=self.config.ransac_threshold
        )
        
        if E is None:
            return None, 0
        
        # Recover pose from essential matrix
        _, R, t, mask = cv2.recoverPose(
            E, pts1, pts2, 
            cameraMatrix=self.config.camera_matrix
        )
        
        num_inliers = np.sum(mask)
        
        if num_inliers < self.config.min_matches:
            return None, num_inliers
        
        pose = Pose(t.flatten(), R)
        return pose, num_inliers


class OpticalFlowVisualOdometry(VisualOdometryAlgorithm):
    """Optical Flow-based Visual Odometry implementation."""
    
    def __init__(self, config: VisualOdometryConfig):
        """Initialize optical flow VO."""
        super().__init__(config)
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.feature_params = dict(
            maxCorners=self.config.max_features,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=3
        )
    
    def _create_feature_detector(self):
        """Optical flow doesn't use traditional feature detector."""
        return None
    
    def _create_matcher(self):
        """Optical flow doesn't use traditional matcher."""
        return None
    
    def _estimate_pose(
        self, 
        kp1: List[cv2.KeyPoint], 
        kp2: List[cv2.KeyPoint], 
        matches: List[cv2.DMatch]
    ) -> Tuple[Optional[Pose], int]:
        """Estimate pose using optical flow."""
        # Convert keypoints to points
        pts1 = np.float32([kp.pt for kp in kp1]).reshape(-1, 1, 2)
        
        # Calculate optical flow
        pts2, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, 
            self.current_frame, 
            pts1, 
            None, 
            **self.lk_params
        )
        
        # Filter good points
        good_pts1 = pts1[status == 1]
        good_pts2 = pts2[status == 1]
        
        if len(good_pts1) < self.config.min_matches:
            return None, len(good_pts1)
        
        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(
            good_pts1, good_pts2, 
            cameraMatrix=self.config.camera_matrix,
            method=cv2.RANSAC,
            prob=self.config.ransac_prob,
            threshold=self.config.ransac_threshold
        )
        
        if E is None:
            return None, 0
        
        # Recover pose from essential matrix
        _, R, t, mask = cv2.recoverPose(
            E, good_pts1, good_pts2, 
            cameraMatrix=self.config.camera_matrix
        )
        
        num_inliers = np.sum(mask)
        
        if num_inliers < self.config.min_matches:
            return None, num_inliers
        
        pose = Pose(t.flatten(), R)
        return pose, num_inliers
    
    def process_frame(self, frame: np.ndarray) -> Optional[Pose]:
        """Process frame with optical flow."""
        if not hasattr(self, 'prev_frame'):
            self.prev_frame = frame
            # Detect initial features
            corners = cv2.goodFeaturesToTrack(frame, **self.feature_params)
            if corners is not None:
                self.prev_kp = [cv2.KeyPoint(x[0][0], x[0][1], 1) for x in corners]
            else:
                self.prev_kp = []
            return None
        
        self.current_frame = frame
        
        if len(self.prev_kp) < self.config.min_matches:
            # Detect new features
            corners = cv2.goodFeaturesToTrack(frame, **self.feature_params)
            if corners is not None:
                self.prev_kp = [cv2.KeyPoint(x[0][0], x[0][1], 1) for x in corners]
            else:
                self.prev_kp = []
            self.prev_frame = frame
            return None
        
        # Estimate pose using optical flow
        pose_delta, num_inliers = self._estimate_pose(self.prev_kp, [], [])
        
        if pose_delta is None:
            logger.warning("Pose estimation failed")
            self.prev_frame = frame
            return None
        
        # Update trajectory
        self.current_pose = self.current_pose.compose(pose_delta)
        self.trajectory.append(self.current_pose)
        
        # Update previous frame data
        self.prev_frame = frame
        
        logger.info(f"Pose estimated with {num_inliers} inliers")
        return pose_delta


def create_visual_odometry(algorithm: str, config: VisualOdometryConfig) -> VisualOdometryAlgorithm:
    """Factory function to create Visual Odometry algorithm.
    
    Args:
        algorithm: Algorithm type ('ORB', 'SIFT', 'OpticalFlow')
        config: Configuration object
        
    Returns:
        Visual Odometry algorithm instance
        
    Raises:
        ValueError: If algorithm type is not supported
    """
    algorithms = {
        'ORB': ORBVisualOdometry,
        'SIFT': SIFTVisualOdometry,
        'OpticalFlow': OpticalFlowVisualOdometry,
    }
    
    if algorithm not in algorithms:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Supported: {list(algorithms.keys())}")
    
    return algorithms[algorithm](config)
