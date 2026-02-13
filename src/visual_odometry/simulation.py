"""Simulation data generation for Visual Odometry testing."""

from __future__ import annotations

import logging
import math
from typing import List, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .core import Pose, VisualOdometryConfig

logger = logging.getLogger(__name__)


class SyntheticTrajectoryGenerator:
    """Generate synthetic camera trajectories for testing Visual Odometry."""
    
    def __init__(self, config: VisualOdometryConfig):
        """Initialize trajectory generator.
        
        Args:
            config: Visual Odometry configuration
        """
        self.config = config
        self.camera_matrix = config.camera_matrix
        self.dist_coeffs = config.dist_coeffs
    
    def generate_circular_trajectory(
        self, 
        radius: float = 2.0, 
        height: float = 1.5, 
        num_frames: int = 100,
        look_at_center: bool = True
    ) -> List[Pose]:
        """Generate a circular camera trajectory.
        
        Args:
            radius: Circle radius
            height: Camera height
            num_frames: Number of poses
            look_at_center: Whether to look at the center
            
        Returns:
            List of camera poses
        """
        poses = []
        
        for i in range(num_frames):
            angle = 2 * math.pi * i / num_frames
            
            # Position
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = height
            position = np.array([x, y, z])
            
            # Orientation
            if look_at_center:
                # Look at the center
                forward = -position / np.linalg.norm(position)
                up = np.array([0, 0, 1])
                right = np.cross(forward, up)
                right = right / np.linalg.norm(right)
                up = np.cross(right, forward)
                
                rotation_matrix = np.column_stack([right, up, -forward])
            else:
                # Tangent to circle
                tangent = np.array([-math.sin(angle), math.cos(angle), 0])
                up = np.array([0, 0, 1])
                forward = np.cross(up, tangent)
                
                rotation_matrix = np.column_stack([tangent, up, forward])
            
            pose = Pose(position, rotation_matrix)
            poses.append(pose)
        
        return poses
    
    def generate_straight_line_trajectory(
        self, 
        start: np.ndarray, 
        end: np.ndarray, 
        num_frames: int = 100,
        look_forward: bool = True
    ) -> List[Pose]:
        """Generate a straight line camera trajectory.
        
        Args:
            start: Starting position
            end: Ending position
            num_frames: Number of poses
            look_forward: Whether to look in the direction of motion
            
        Returns:
            List of camera poses
        """
        poses = []
        
        for i in range(num_frames):
            t = i / (num_frames - 1)
            position = start + t * (end - start)
            
            if look_forward:
                # Look in the direction of motion
                forward = (end - start) / np.linalg.norm(end - start)
                up = np.array([0, 0, 1])
                right = np.cross(forward, up)
                right = right / np.linalg.norm(right)
                up = np.cross(right, forward)
                
                rotation_matrix = np.column_stack([right, up, forward])
            else:
                # Default orientation
                rotation_matrix = np.eye(3)
            
            pose = Pose(position, rotation_matrix)
            poses.append(pose)
        
        return poses
    
    def generate_figure_eight_trajectory(
        self, 
        radius: float = 2.0, 
        height: float = 1.5, 
        num_frames: int = 200
    ) -> List[Pose]:
        """Generate a figure-eight camera trajectory.
        
        Args:
            radius: Trajectory radius
            height: Camera height
            num_frames: Number of poses
            
        Returns:
            List of camera poses
        """
        poses = []
        
        for i in range(num_frames):
            t = 2 * math.pi * i / num_frames
            
            # Figure-eight parametric equations
            x = radius * math.sin(t)
            y = radius * math.sin(2 * t) / 2
            z = height
            position = np.array([x, y, z])
            
            # Compute velocity for orientation
            vx = radius * math.cos(t)
            vy = radius * math.cos(2 * t)
            vz = 0
            
            if np.linalg.norm([vx, vy]) > 0:
                # Look in the direction of motion
                forward = np.array([vx, vy, vz]) / np.linalg.norm([vx, vy, vz])
                up = np.array([0, 0, 1])
                right = np.cross(forward, up)
                right = right / np.linalg.norm(right)
                up = np.cross(right, forward)
                
                rotation_matrix = np.column_stack([right, up, forward])
            else:
                rotation_matrix = np.eye(3)
            
            pose = Pose(position, rotation_matrix)
            poses.append(pose)
        
        return poses


class SyntheticImageGenerator:
    """Generate synthetic images for Visual Odometry testing."""
    
    def __init__(self, config: VisualOdometryConfig):
        """Initialize image generator.
        
        Args:
            config: Visual Odometry configuration
        """
        self.config = config
        self.camera_matrix = config.camera_matrix
        self.dist_coeffs = config.dist_coeffs
        self.image_size = (640, 480)  # Default image size
    
    def generate_scene_points(self, num_points: int = 1000) -> np.ndarray:
        """Generate random 3D points in the scene.
        
        Args:
            num_points: Number of points to generate
            
        Returns:
            Array of 3D points
        """
        # Generate points in a cube around the origin
        points = np.random.uniform(-10, 10, (num_points, 3))
        
        # Filter points that are in front of typical camera positions
        points = points[points[:, 2] > 0]
        
        return points
    
    def project_points(self, points_3d: np.ndarray, pose: Pose) -> Tuple[np.ndarray, np.ndarray]:
        """Project 3D points to image coordinates.
        
        Args:
            points_3d: 3D points in world coordinates
            pose: Camera pose
            
        Returns:
            Tuple of (image_points, valid_mask)
        """
        # Transform points to camera coordinates
        points_cam = []
        for point in points_3d:
            point_cam = pose.inverse().transform_point(point)
            points_cam.append(point_cam)
        
        points_cam = np.array(points_cam)
        
        # Filter points in front of camera
        valid_mask = points_cam[:, 2] > 0
        points_cam = points_cam[valid_mask]
        
        if len(points_cam) == 0:
            return np.array([]), np.array([])
        
        # Project to image plane
        image_points, _ = cv2.projectPoints(
            points_cam,
            np.zeros(3),  # No rotation
            np.zeros(3),  # No translation
            self.camera_matrix,
            self.dist_coeffs
        )
        
        image_points = image_points.reshape(-1, 2)
        
        # Filter points within image bounds
        valid_mask = (
            (image_points[:, 0] >= 0) & 
            (image_points[:, 0] < self.image_size[0]) &
            (image_points[:, 1] >= 0) & 
            (image_points[:, 1] < self.image_size[1])
        )
        
        return image_points, valid_mask
    
    def generate_image(self, pose: Pose, scene_points: np.ndarray) -> np.ndarray:
        """Generate a synthetic image for a given pose.
        
        Args:
            pose: Camera pose
            scene_points: 3D points in the scene
            
        Returns:
            Generated image
        """
        # Create blank image
        image = np.zeros((self.image_size[1], self.image_size[0]), dtype=np.uint8)
        
        # Project points to image
        image_points, valid_mask = self.project_points(scene_points, pose)
        
        if len(image_points) == 0:
            return image
        
        # Draw points as circles
        for point in image_points[valid_mask]:
            cv2.circle(image, (int(point[0]), int(point[1])), 2, 255, -1)
        
        # Add some noise
        noise = np.random.normal(0, 10, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def generate_image_sequence(
        self, 
        poses: List[Pose], 
        scene_points: Optional[np.ndarray] = None
    ) -> List[np.ndarray]:
        """Generate a sequence of images for a trajectory.
        
        Args:
            poses: List of camera poses
            scene_points: 3D points in the scene (optional)
            
        Returns:
            List of generated images
        """
        if scene_points is None:
            scene_points = self.generate_scene_points()
        
        images = []
        for pose in poses:
            image = self.generate_image(pose, scene_points)
            images.append(image)
        
        return images


class VisualOdometryDataset:
    """Dataset class for Visual Odometry with synthetic data."""
    
    def __init__(self, config: VisualOdometryConfig):
        """Initialize dataset.
        
        Args:
            config: Visual Odometry configuration
        """
        self.config = config
        self.trajectory_generator = SyntheticTrajectoryGenerator(config)
        self.image_generator = SyntheticImageGenerator(config)
        
        self.poses: List[Pose] = []
        self.images: List[np.ndarray] = []
        self.scene_points: np.ndarray = np.array([])
    
    def generate_circular_dataset(
        self, 
        radius: float = 2.0, 
        height: float = 1.5, 
        num_frames: int = 100
    ) -> Tuple[List[np.ndarray], List[Pose]]:
        """Generate a circular trajectory dataset.
        
        Args:
            radius: Circle radius
            height: Camera height
            num_frames: Number of frames
            
        Returns:
            Tuple of (images, poses)
        """
        # Generate trajectory
        self.poses = self.trajectory_generator.generate_circular_trajectory(
            radius, height, num_frames
        )
        
        # Generate scene points
        self.scene_points = self.image_generator.generate_scene_points()
        
        # Generate images
        self.images = self.image_generator.generate_image_sequence(
            self.poses, self.scene_points
        )
        
        return self.images, self.poses
    
    def generate_straight_line_dataset(
        self, 
        start: np.ndarray, 
        end: np.ndarray, 
        num_frames: int = 100
    ) -> Tuple[List[np.ndarray], List[Pose]]:
        """Generate a straight line trajectory dataset.
        
        Args:
            start: Starting position
            end: Ending position
            num_frames: Number of frames
            
        Returns:
            Tuple of (images, poses)
        """
        # Generate trajectory
        self.poses = self.trajectory_generator.generate_straight_line_trajectory(
            start, end, num_frames
        )
        
        # Generate scene points
        self.scene_points = self.image_generator.generate_scene_points()
        
        # Generate images
        self.images = self.image_generator.generate_image_sequence(
            self.poses, self.scene_points
        )
        
        return self.images, self.poses
    
    def generate_figure_eight_dataset(
        self, 
        radius: float = 2.0, 
        height: float = 1.5, 
        num_frames: int = 200
    ) -> Tuple[List[np.ndarray], List[Pose]]:
        """Generate a figure-eight trajectory dataset.
        
        Args:
            radius: Trajectory radius
            height: Camera height
            num_frames: Number of frames
            
        Returns:
            Tuple of (images, poses)
        """
        # Generate trajectory
        self.poses = self.trajectory_generator.generate_figure_eight_trajectory(
            radius, height, num_frames
        )
        
        # Generate scene points
        self.scene_points = self.image_generator.generate_scene_points()
        
        # Generate images
        self.images = self.image_generator.generate_image_sequence(
            self.poses, self.scene_points
        )
        
        return self.images, self.poses
    
    def save_dataset(self, output_dir: str):
        """Save dataset to files.
        
        Args:
            output_dir: Output directory
        """
        import os
        import pickle
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save images
        for i, image in enumerate(self.images):
            cv2.imwrite(os.path.join(output_dir, f"frame_{i:06d}.png"), image)
        
        # Save poses
        poses_data = {
            'positions': [pose.position for pose in self.poses],
            'rotations': [pose.rotation_matrix for pose in self.poses],
        }
        
        with open(os.path.join(output_dir, "poses.pkl"), 'wb') as f:
            pickle.dump(poses_data, f)
        
        # Save scene points
        np.save(os.path.join(output_dir, "scene_points.npy"), self.scene_points)
        
        logger.info(f"Dataset saved to {output_dir}")
    
    def load_dataset(self, input_dir: str):
        """Load dataset from files.
        
        Args:
            input_dir: Input directory
        """
        import os
        import pickle
        import glob
        
        # Load images
        image_files = sorted(glob.glob(os.path.join(input_dir, "frame_*.png")))
        self.images = []
        for image_file in image_files:
            image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            self.images.append(image)
        
        # Load poses
        with open(os.path.join(input_dir, "poses.pkl"), 'rb') as f:
            poses_data = pickle.load(f)
        
        self.poses = []
        for pos, rot in zip(poses_data['positions'], poses_data['rotations']):
            pose = Pose(pos, rot)
            self.poses.append(pose)
        
        # Load scene points
        self.scene_points = np.load(os.path.join(input_dir, "scene_points.npy"))
        
        logger.info(f"Dataset loaded from {input_dir}")
