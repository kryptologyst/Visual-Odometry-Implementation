"""Utility functions for Visual Odometry."""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from .core import Pose, VisualOdometryConfig

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )


def load_config(config_path: str) -> VisualOdometryConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Visual Odometry configuration object
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert camera matrix from list to numpy array
    if 'camera_matrix' in config_dict:
        config_dict['camera_matrix'] = np.array(config_dict['camera_matrix'])
    
    # Convert distortion coefficients from list to numpy array
    if 'dist_coeffs' in config_dict:
        config_dict['dist_coeffs'] = np.array(config_dict['dist_coeffs'])
    
    return VisualOdometryConfig(**config_dict)


def save_config(config: VisualOdometryConfig, config_path: str):
    """Save configuration to YAML file.
    
    Args:
        config: Visual Odometry configuration object
        config_path: Path to save configuration file
    """
    config_dict = {
        'feature_detector': config.feature_detector,
        'max_features': config.max_features,
        'match_ratio': config.match_ratio,
        'ransac_threshold': config.ransac_threshold,
        'ransac_prob': config.ransac_prob,
        'min_matches': config.min_matches,
        'camera_matrix': config.camera_matrix.tolist(),
        'dist_coeffs': config.dist_coeffs.tolist(),
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)


def load_images_from_directory(
    directory: str, 
    pattern: str = "*.png",
    sort: bool = True
) -> List[np.ndarray]:
    """Load images from directory.
    
    Args:
        directory: Directory containing images
        pattern: File pattern to match
        sort: Whether to sort files
        
    Returns:
        List of loaded images
    """
    import glob
    
    image_files = glob.glob(os.path.join(directory, pattern))
    
    if sort:
        image_files = sorted(image_files)
    
    images = []
    for image_file in image_files:
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            images.append(image)
        else:
            logger.warning(f"Failed to load image: {image_file}")
    
    logger.info(f"Loaded {len(images)} images from {directory}")
    return images


def load_trajectory_from_file(file_path: str) -> List[Pose]:
    """Load trajectory from file.
    
    Args:
        file_path: Path to trajectory file
        
    Returns:
        List of poses
    """
    import pickle
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    poses = []
    for pos, rot in zip(data['positions'], data['rotations']):
        pose = Pose(pos, rot)
        poses.append(pose)
    
    logger.info(f"Loaded {len(poses)} poses from {file_path}")
    return poses


def save_trajectory_to_file(poses: List[Pose], file_path: str):
    """Save trajectory to file.
    
    Args:
        poses: List of poses
        file_path: Path to save trajectory file
    """
    import pickle
    
    data = {
        'positions': [pose.position for pose in poses],
        'rotations': [pose.rotation_matrix for pose in poses],
    }
    
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    
    logger.info(f"Saved {len(poses)} poses to {file_path}")


def visualize_trajectory(
    poses: List[Pose], 
    ground_truth: Optional[List[Pose]] = None,
    title: str = "Camera Trajectory",
    save_path: Optional[str] = None
):
    """Visualize camera trajectory.
    
    Args:
        poses: Estimated trajectory
        ground_truth: Ground truth trajectory (optional)
        title: Plot title
        save_path: Path to save plot (optional)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract positions
    est_positions = np.array([pose.position for pose in poses])
    
    # Plot estimated trajectory
    ax.plot(est_positions[:, 0], est_positions[:, 1], est_positions[:, 2], 
            'b-', label='Estimated', linewidth=2)
    ax.scatter(est_positions[0, 0], est_positions[0, 1], est_positions[0, 2], 
               'go', s=100, label='Start')
    ax.scatter(est_positions[-1, 0], est_positions[-1, 1], est_positions[-1, 2], 
               'ro', s=100, label='End')
    
    # Plot ground truth if available
    if ground_truth is not None:
        gt_positions = np.array([pose.position for pose in ground_truth])
        ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 
                'r--', label='Ground Truth', linewidth=2)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Trajectory plot saved to {save_path}")
    
    plt.show()


def visualize_feature_matches(
    image1: np.ndarray, 
    image2: np.ndarray, 
    kp1: List[cv2.KeyPoint], 
    kp2: List[cv2.KeyPoint], 
    matches: List[cv2.DMatch],
    max_matches: int = 50,
    save_path: Optional[str] = None
):
    """Visualize feature matches between two images.
    
    Args:
        image1: First image
        image2: Second image
        kp1: Keypoints in first image
        kp2: Keypoints in second image
        matches: Feature matches
        max_matches: Maximum number of matches to display
        save_path: Path to save plot (optional)
    """
    import matplotlib.pyplot as plt
    
    # Draw matches
    img_matches = cv2.drawMatches(
        image1, kp1, image2, kp2, matches[:max_matches], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    plt.figure(figsize=(15, 10))
    plt.imshow(img_matches)
    plt.title(f"Feature Matches ({len(matches)} total)")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature matches plot saved to {save_path}")
    
    plt.show()


def compute_trajectory_statistics(poses: List[Pose]) -> dict:
    """Compute trajectory statistics.
    
    Args:
        poses: List of poses
        
    Returns:
        Dictionary containing trajectory statistics
    """
    if len(poses) < 2:
        return {}
    
    positions = np.array([pose.position for pose in poses])
    
    # Compute trajectory length
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    total_length = np.sum(distances)
    
    # Compute velocity statistics
    velocities = distances  # Assuming unit time between frames
    
    # Compute acceleration statistics
    accelerations = np.diff(velocities)
    
    # Compute orientation statistics
    euler_angles = np.array([pose.euler_angles for pose in poses])
    
    stats = {
        'num_poses': len(poses),
        'total_length': float(total_length),
        'mean_velocity': float(np.mean(velocities)),
        'std_velocity': float(np.std(velocities)),
        'max_velocity': float(np.max(velocities)),
        'mean_acceleration': float(np.mean(accelerations)) if len(accelerations) > 0 else 0.0,
        'std_acceleration': float(np.std(accelerations)) if len(accelerations) > 0 else 0.0,
        'max_acceleration': float(np.max(np.abs(accelerations))) if len(accelerations) > 0 else 0.0,
        'position_range': {
            'x': [float(np.min(positions[:, 0])), float(np.max(positions[:, 0]))],
            'y': [float(np.min(positions[:, 1])), float(np.max(positions[:, 1]))],
            'z': [float(np.min(positions[:, 2])), float(np.max(positions[:, 2]))],
        },
        'orientation_range': {
            'roll': [float(np.min(euler_angles[:, 0])), float(np.max(euler_angles[:, 0]))],
            'pitch': [float(np.min(euler_angles[:, 1])), float(np.max(euler_angles[:, 1]))],
            'yaw': [float(np.min(euler_angles[:, 2])), float(np.max(euler_angles[:, 2]))],
        },
    }
    
    return stats


def create_camera_calibration_matrix(
    width: int, 
    height: int, 
    fov_degrees: float = 60.0
) -> np.ndarray:
    """Create a camera calibration matrix from image dimensions and field of view.
    
    Args:
        width: Image width
        height: Image height
        fov_degrees: Field of view in degrees
        
    Returns:
        Camera calibration matrix
    """
    fov_radians = np.radians(fov_degrees)
    focal_length = width / (2 * np.tan(fov_radians / 2))
    
    camera_matrix = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return camera_matrix


def undistort_image(image: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    """Undistort an image using camera calibration parameters.
    
    Args:
        image: Input image
        camera_matrix: Camera calibration matrix
        dist_coeffs: Distortion coefficients
        
    Returns:
        Undistorted image
    """
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    return undistorted


def set_random_seed(seed: int):
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Set OpenCV random seed if available
    try:
        cv2.setRNGSeed(seed)
    except AttributeError:
        pass
    
    logger.info(f"Random seed set to {seed}")


def get_device_info() -> dict:
    """Get information about available compute devices.
    
    Returns:
        Dictionary containing device information
    """
    import platform
    import psutil
    
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'opencv_version': cv2.__version__,
        'numpy_version': np.__version__,
    }
    
    # Check for CUDA
    try:
        import torch
        info['torch_version'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['cuda_device_count'] = torch.cuda.device_count()
            info['cuda_device_name'] = torch.cuda.get_device_name(0)
    except ImportError:
        info['torch_available'] = False
    
    return info
