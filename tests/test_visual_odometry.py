"""Tests for Visual Odometry implementation."""

import unittest
from unittest.mock import Mock, patch

import numpy as np
import cv2

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from visual_odometry import (
    VisualOdometryConfig,
    Pose,
    create_visual_odometry,
    ORBVisualOdometry,
    SIFTVisualOdometry,
    OpticalFlowVisualOdometry,
)
from visual_odometry.metrics import VisualOdometryMetrics
from visual_odometry.simulation import (
    SyntheticTrajectoryGenerator,
    SyntheticImageGenerator,
    VisualOdometryDataset,
)


class TestPose(unittest.TestCase):
    """Test Pose class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.position = np.array([1.0, 2.0, 3.0])
        self.rotation_matrix = np.eye(3)
        self.pose = Pose(self.position, self.rotation_matrix)
    
    def test_pose_creation(self):
        """Test pose creation."""
        self.assertTrue(np.allclose(self.pose.position, self.position))
        self.assertTrue(np.allclose(self.pose.rotation_matrix, self.rotation_matrix))
    
    def test_pose_inverse(self):
        """Test pose inverse."""
        inverse_pose = self.pose.inverse()
        expected_position = -self.position
        self.assertTrue(np.allclose(inverse_pose.position, expected_position))
    
    def test_pose_compose(self):
        """Test pose composition."""
        other_pose = Pose(np.array([1.0, 0.0, 0.0]), np.eye(3))
        composed_pose = self.pose.compose(other_pose)
        expected_position = self.position + other_pose.position
        self.assertTrue(np.allclose(composed_pose.position, expected_position))
    
    def test_transform_point(self):
        """Test point transformation."""
        point = np.array([1.0, 0.0, 0.0])
        transformed_point = self.pose.transform_point(point)
        expected_point = point + self.position
        self.assertTrue(np.allclose(transformed_point, expected_point))


class TestVisualOdometryConfig(unittest.TestCase):
    """Test VisualOdometryConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = VisualOdometryConfig()
        self.assertEqual(config.feature_detector, "ORB")
        self.assertEqual(config.max_features, 1000)
        self.assertEqual(config.match_ratio, 0.7)
        self.assertIsNotNone(config.camera_matrix)
        self.assertIsNotNone(config.dist_coeffs)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = VisualOdometryConfig(
            feature_detector="SIFT",
            max_features=500,
            match_ratio=0.8
        )
        self.assertEqual(config.feature_detector, "SIFT")
        self.assertEqual(config.max_features, 500)
        self.assertEqual(config.match_ratio, 0.8)


class TestVisualOdometryAlgorithms(unittest.TestCase):
    """Test Visual Odometry algorithms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = VisualOdometryConfig()
        self.images = self._create_test_images()
    
    def _create_test_images(self):
        """Create test images."""
        images = []
        for i in range(5):
            # Create a simple test image with some features
            image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
            # Add some structured features
            cv2.circle(image, (100 + i*10, 100 + i*5), 5, 255, -1)
            cv2.circle(image, (200 + i*8, 200 + i*3), 3, 255, -1)
            images.append(image)
        return images
    
    def test_orb_visual_odometry(self):
        """Test ORB Visual Odometry."""
        vo = ORBVisualOdometry(self.config)
        
        # Process first frame
        pose_delta = vo.process_frame(self.images[0])
        self.assertIsNone(pose_delta)  # First frame should return None
        
        # Process second frame
        pose_delta = vo.process_frame(self.images[1])
        # Should return a pose or None depending on feature detection
        if pose_delta is not None:
            self.assertIsInstance(pose_delta, Pose)
    
    def test_sift_visual_odometry(self):
        """Test SIFT Visual Odometry."""
        vo = SIFTVisualOdometry(self.config)
        
        # Process first frame
        pose_delta = vo.process_frame(self.images[0])
        self.assertIsNone(pose_delta)  # First frame should return None
        
        # Process second frame
        pose_delta = vo.process_frame(self.images[1])
        # Should return a pose or None depending on feature detection
        if pose_delta is not None:
            self.assertIsInstance(pose_delta, Pose)
    
    def test_optical_flow_visual_odometry(self):
        """Test Optical Flow Visual Odometry."""
        vo = OpticalFlowVisualOdometry(self.config)
        
        # Process first frame
        pose_delta = vo.process_frame(self.images[0])
        self.assertIsNone(pose_delta)  # First frame should return None
        
        # Process second frame
        pose_delta = vo.process_frame(self.images[1])
        # Should return a pose or None depending on feature detection
        if pose_delta is not None:
            self.assertIsInstance(pose_delta, Pose)
    
    def test_create_visual_odometry(self):
        """Test factory function."""
        vo = create_visual_odometry("ORB", self.config)
        self.assertIsInstance(vo, ORBVisualOdometry)
        
        vo = create_visual_odometry("SIFT", self.config)
        self.assertIsInstance(vo, SIFTVisualOdometry)
        
        vo = create_visual_odometry("OpticalFlow", self.config)
        self.assertIsInstance(vo, OpticalFlowVisualOdometry)
        
        with self.assertRaises(ValueError):
            create_visual_odometry("INVALID", self.config)


class TestVisualOdometryMetrics(unittest.TestCase):
    """Test Visual Odometry metrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics = VisualOdometryMetrics()
        self.poses = self._create_test_poses()
    
    def _create_test_poses(self):
        """Create test poses."""
        poses = []
        for i in range(10):
            position = np.array([i * 0.1, 0.0, 1.0])
            rotation = np.eye(3)
            pose = Pose(position, rotation)
            poses.append(pose)
        return poses
    
    def test_add_pose(self):
        """Test adding poses."""
        pose = self.poses[0]
        self.metrics.add_pose(pose)
        self.assertEqual(len(self.metrics.estimated_trajectory), 1)
    
    def test_set_trajectories(self):
        """Test setting trajectories."""
        self.metrics.set_trajectories(self.poses)
        self.assertEqual(len(self.metrics.estimated_trajectory), len(self.poses))
    
    def test_compute_trajectory_length(self):
        """Test trajectory length computation."""
        self.metrics.set_trajectories(self.poses)
        length_metrics = self.metrics.compute_trajectory_length()
        self.assertIn('estimated_length', length_metrics)
        self.assertGreater(length_metrics['estimated_length'], 0)
    
    def test_reset(self):
        """Test metrics reset."""
        self.metrics.add_pose(self.poses[0])
        self.metrics.reset()
        self.assertEqual(len(self.metrics.estimated_trajectory), 0)


class TestSyntheticTrajectoryGenerator(unittest.TestCase):
    """Test synthetic trajectory generator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = VisualOdometryConfig()
        self.generator = SyntheticTrajectoryGenerator(self.config)
    
    def test_circular_trajectory(self):
        """Test circular trajectory generation."""
        poses = self.generator.generate_circular_trajectory(num_frames=10)
        self.assertEqual(len(poses), 10)
        
        # Check that poses are valid
        for pose in poses:
            self.assertIsInstance(pose, Pose)
            self.assertEqual(len(pose.position), 3)
            self.assertEqual(pose.rotation_matrix.shape, (3, 3))
    
    def test_straight_line_trajectory(self):
        """Test straight line trajectory generation."""
        start = np.array([0, 0, 1])
        end = np.array([5, 0, 1])
        poses = self.generator.generate_straight_line_trajectory(start, end, num_frames=10)
        self.assertEqual(len(poses), 10)
        
        # Check that poses are valid
        for pose in poses:
            self.assertIsInstance(pose, Pose)
            self.assertEqual(len(pose.position), 3)
            self.assertEqual(pose.rotation_matrix.shape, (3, 3))
    
    def test_figure_eight_trajectory(self):
        """Test figure-eight trajectory generation."""
        poses = self.generator.generate_figure_eight_trajectory(num_frames=20)
        self.assertEqual(len(poses), 20)
        
        # Check that poses are valid
        for pose in poses:
            self.assertIsInstance(pose, Pose)
            self.assertEqual(len(pose.position), 3)
            self.assertEqual(pose.rotation_matrix.shape, (3, 3))


class TestSyntheticImageGenerator(unittest.TestCase):
    """Test synthetic image generator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = VisualOdometryConfig()
        self.generator = SyntheticImageGenerator(self.config)
        self.pose = Pose(np.array([0, 0, 1]), np.eye(3))
    
    def test_generate_scene_points(self):
        """Test scene point generation."""
        points = self.generator.generate_scene_points(num_points=100)
        self.assertEqual(len(points), 100)
        self.assertEqual(points.shape[1], 3)
    
    def test_project_points(self):
        """Test point projection."""
        points_3d = np.array([[1, 0, 2], [0, 1, 2], [0, 0, 2]])
        image_points, valid_mask = self.generator.project_points(points_3d, self.pose)
        
        self.assertIsInstance(image_points, np.ndarray)
        self.assertIsInstance(valid_mask, np.ndarray)
    
    def test_generate_image(self):
        """Test image generation."""
        scene_points = self.generator.generate_scene_points(num_points=50)
        image = self.generator.generate_image(self.pose, scene_points)
        
        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(len(image.shape), 2)  # Grayscale image


class TestVisualOdometryDataset(unittest.TestCase):
    """Test Visual Odometry dataset."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = VisualOdometryConfig()
        self.dataset = VisualOdometryDataset(self.config)
    
    def test_generate_circular_dataset(self):
        """Test circular dataset generation."""
        images, poses = self.dataset.generate_circular_dataset(num_frames=10)
        
        self.assertEqual(len(images), 10)
        self.assertEqual(len(poses), 10)
        
        # Check that images are valid
        for image in images:
            self.assertIsInstance(image, np.ndarray)
            self.assertEqual(len(image.shape), 2)  # Grayscale image
        
        # Check that poses are valid
        for pose in poses:
            self.assertIsInstance(pose, Pose)
    
    def test_generate_straight_line_dataset(self):
        """Test straight line dataset generation."""
        start = np.array([0, 0, 1])
        end = np.array([5, 0, 1])
        images, poses = self.dataset.generate_straight_line_dataset(start, end, num_frames=10)
        
        self.assertEqual(len(images), 10)
        self.assertEqual(len(poses), 10)
    
    def test_generate_figure_eight_dataset(self):
        """Test figure-eight dataset generation."""
        images, poses = self.dataset.generate_figure_eight_dataset(num_frames=20)
        
        self.assertEqual(len(images), 20)
        self.assertEqual(len(poses), 20)


if __name__ == '__main__':
    unittest.main()
