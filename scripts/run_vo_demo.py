#!/usr/bin/env python3
"""Main script for Visual Odometry demonstration."""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from visual_odometry import (
    VisualOdometryConfig,
    create_visual_odometry,
    VisualOdometryMetrics,
    VisualOdometryLeaderboard,
)
from visual_odometry.simulation import VisualOdometryDataset
from utils import (
    setup_logging,
    set_random_seed,
    visualize_trajectory,
    compute_trajectory_statistics,
    get_device_info,
)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Visual Odometry Implementation")
    parser.add_argument(
        "--algorithm", 
        choices=["ORB", "SIFT", "OpticalFlow"], 
        default="ORB",
        help="Visual Odometry algorithm to use"
    )
    parser.add_argument(
        "--trajectory", 
        choices=["circular", "straight", "figure_eight"], 
        default="circular",
        help="Trajectory type to generate"
    )
    parser.add_argument(
        "--num-frames", 
        type=int, 
        default=100,
        help="Number of frames to generate"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="assets",
        help="Output directory for results"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--save-results", 
        action="store_true",
        help="Save results to files"
    )
    parser.add_argument(
        "--compare-algorithms", 
        action="store_true",
        help="Compare all algorithms"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print system info
    device_info = get_device_info()
    logger.info(f"System info: {device_info}")
    
    # Create configuration
    config = VisualOdometryConfig(
        feature_detector=args.algorithm,
        max_features=1000,
        match_ratio=0.7,
        ransac_threshold=1.0,
        min_matches=50,
    )
    
    # Create dataset
    dataset = VisualOdometryDataset(config)
    
    # Generate dataset based on trajectory type
    if args.trajectory == "circular":
        images, ground_truth_poses = dataset.generate_circular_dataset(
            radius=2.0, height=1.5, num_frames=args.num_frames
        )
    elif args.trajectory == "straight":
        start = np.array([0, 0, 1.5])
        end = np.array([5, 0, 1.5])
        images, ground_truth_poses = dataset.generate_straight_line_dataset(
            start, end, num_frames=args.num_frames
        )
    elif args.trajectory == "figure_eight":
        images, ground_truth_poses = dataset.generate_figure_eight_dataset(
            radius=2.0, height=1.5, num_frames=args.num_frames
        )
    else:
        raise ValueError(f"Unknown trajectory type: {args.trajectory}")
    
    logger.info(f"Generated {len(images)} images and {len(ground_truth_poses)} poses")
    
    if args.compare_algorithms:
        # Compare all algorithms
        algorithms = ["ORB", "SIFT", "OpticalFlow"]
        leaderboard = VisualOdometryLeaderboard()
        
        for algorithm in algorithms:
            logger.info(f"Running {algorithm} algorithm...")
            
            # Create VO algorithm
            config.feature_detector = algorithm
            vo = create_visual_odometry(algorithm, config)
            
            # Process images
            estimated_poses = []
            metrics = VisualOdometryMetrics()
            
            for i, image in enumerate(images):
                pose_delta = vo.process_frame(image)
                if pose_delta is not None:
                    estimated_poses.append(vo.current_pose)
                    metrics.add_pose(vo.current_pose, ground_truth_poses[i])
                else:
                    logger.warning(f"Failed to estimate pose for frame {i}")
            
            # Compute metrics
            algorithm_metrics = metrics.compute_all_metrics()
            leaderboard.add_result(algorithm, algorithm_metrics)
            
            # Save results
            if args.save_results:
                algorithm_dir = os.path.join(args.output_dir, algorithm.lower())
                os.makedirs(algorithm_dir, exist_ok=True)
                
                # Save trajectory
                trajectory_file = os.path.join(algorithm_dir, "trajectory.pkl")
                from utils import save_trajectory_to_file
                save_trajectory_to_file(estimated_poses, trajectory_file)
                
                # Save metrics
                import json
                metrics_file = os.path.join(algorithm_dir, "metrics.json")
                with open(metrics_file, 'w') as f:
                    json.dump(algorithm_metrics, f, indent=2)
                
                # Save trajectory plot
                plot_file = os.path.join(algorithm_dir, "trajectory.png")
                visualize_trajectory(
                    estimated_poses, 
                    ground_truth_poses,
                    title=f"{algorithm} Visual Odometry Trajectory",
                    save_path=plot_file
                )
        
        # Print leaderboard
        leaderboard.print_leaderboard('ate_rmse')
        leaderboard.print_leaderboard('rpe_trans_rmse')
        
        # Save leaderboard
        if args.save_results:
            leaderboard.export_results(os.path.join(args.output_dir, "leaderboard.csv"))
    
    else:
        # Run single algorithm
        logger.info(f"Running {args.algorithm} algorithm...")
        
        # Create VO algorithm
        vo = create_visual_odometry(args.algorithm, config)
        
        # Process images
        estimated_poses = []
        metrics = VisualOdometryMetrics()
        
        for i, image in enumerate(images):
            pose_delta = vo.process_frame(image)
            if pose_delta is not None:
                estimated_poses.append(vo.current_pose)
                metrics.add_pose(vo.current_pose, ground_truth_poses[i])
            else:
                logger.warning(f"Failed to estimate pose for frame {i}")
        
        # Compute metrics
        algorithm_metrics = metrics.compute_all_metrics()
        
        # Print results
        logger.info("Visual Odometry Results:")
        for metric, value in algorithm_metrics.items():
            logger.info(f"  {metric}: {value:.6f}")
        
        # Compute trajectory statistics
        stats = compute_trajectory_statistics(estimated_poses)
        logger.info("Trajectory Statistics:")
        for stat, value in stats.items():
            logger.info(f"  {stat}: {value}")
        
        # Visualize results
        visualize_trajectory(
            estimated_poses, 
            ground_truth_poses,
            title=f"{args.algorithm} Visual Odometry Trajectory"
        )
        
        # Save results
        if args.save_results:
            # Save trajectory
            trajectory_file = os.path.join(args.output_dir, "trajectory.pkl")
            from utils import save_trajectory_to_file
            save_trajectory_to_file(estimated_poses, trajectory_file)
            
            # Save metrics
            import json
            metrics_file = os.path.join(args.output_dir, "metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(algorithm_metrics, f, indent=2)
            
            # Save trajectory plot
            plot_file = os.path.join(args.output_dir, "trajectory.png")
            visualize_trajectory(
                estimated_poses, 
                ground_truth_poses,
                title=f"{args.algorithm} Visual Odometry Trajectory",
                save_path=plot_file
            )
            
            # Save dataset
            dataset.save_dataset(os.path.join(args.output_dir, "dataset"))
            
            logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
