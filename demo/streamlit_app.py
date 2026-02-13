"""Streamlit demo for Visual Odometry."""

import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    set_random_seed,
    compute_trajectory_statistics,
    get_device_info,
)


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Visual Odometry Demo",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("Visual Odometry Implementation Demo")
    st.markdown("""
    This demo showcases different Visual Odometry algorithms on synthetic data.
    
    **DISCLAIMER**: This is for educational and research purposes only. 
    DO NOT use on real robots without proper safety measures and expert review.
    """)
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    
    algorithm = st.sidebar.selectbox(
        "Algorithm",
        ["ORB", "SIFT", "OpticalFlow"],
        help="Visual Odometry algorithm to use"
    )
    
    trajectory_type = st.sidebar.selectbox(
        "Trajectory Type",
        ["circular", "straight", "figure_eight"],
        help="Type of trajectory to generate"
    )
    
    num_frames = st.sidebar.slider(
        "Number of Frames",
        min_value=50,
        max_value=500,
        value=100,
        step=10,
        help="Number of frames to generate"
    )
    
    max_features = st.sidebar.slider(
        "Max Features",
        min_value=100,
        max_value=2000,
        value=1000,
        step=100,
        help="Maximum number of features to detect"
    )
    
    match_ratio = st.sidebar.slider(
        "Match Ratio",
        min_value=0.5,
        max_value=0.9,
        value=0.7,
        step=0.05,
        help="Ratio for feature matching (Lowe's ratio test)"
    )
    
    ransac_threshold = st.sidebar.slider(
        "RANSAC Threshold",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="RANSAC threshold for essential matrix estimation"
    )
    
    min_matches = st.sidebar.slider(
        "Min Matches",
        min_value=10,
        max_value=100,
        value=50,
        step=5,
        help="Minimum number of matches required for pose estimation"
    )
    
    seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=10000,
        value=42,
        help="Random seed for reproducibility"
    )
    
    compare_algorithms = st.sidebar.checkbox(
        "Compare All Algorithms",
        help="Run all algorithms and compare results"
    )
    
    # Main content
    if st.button("Run Visual Odometry", type="primary"):
        with st.spinner("Running Visual Odometry..."):
            # Set random seed
            set_random_seed(seed)
            
            # Create configuration
            config = VisualOdometryConfig(
                feature_detector=algorithm,
                max_features=max_features,
                match_ratio=match_ratio,
                ransac_threshold=ransac_threshold,
                min_matches=min_matches,
            )
            
            # Create dataset
            dataset = VisualOdometryDataset(config)
            
            # Generate dataset based on trajectory type
            if trajectory_type == "circular":
                images, ground_truth_poses = dataset.generate_circular_dataset(
                    radius=2.0, height=1.5, num_frames=num_frames
                )
            elif trajectory_type == "straight":
                start = np.array([0, 0, 1.5])
                end = np.array([5, 0, 1.5])
                images, ground_truth_poses = dataset.generate_straight_line_dataset(
                    start, end, num_frames=num_frames
                )
            elif trajectory_type == "figure_eight":
                images, ground_truth_poses = dataset.generate_figure_eight_dataset(
                    radius=2.0, height=1.5, num_frames=num_frames
                )
            
            if compare_algorithms:
                # Compare all algorithms
                algorithms = ["ORB", "SIFT", "OpticalFlow"]
                leaderboard = VisualOdometryLeaderboard()
                
                results = {}
                
                for alg in algorithms:
                    # Create VO algorithm
                    config.feature_detector = alg
                    vo = create_visual_odometry(alg, config)
                    
                    # Process images
                    estimated_poses = []
                    metrics = VisualOdometryMetrics()
                    
                    for i, image in enumerate(images):
                        pose_delta = vo.process_frame(image)
                        if pose_delta is not None:
                            estimated_poses.append(vo.current_pose)
                            metrics.add_pose(vo.current_pose, ground_truth_poses[i])
                    
                    # Compute metrics
                    algorithm_metrics = metrics.compute_all_metrics()
                    leaderboard.add_result(alg, algorithm_metrics)
                    results[alg] = {
                        'poses': estimated_poses,
                        'metrics': algorithm_metrics
                    }
                
                # Display results
                st.header("Algorithm Comparison Results")
                
                # Create tabs for each algorithm
                tabs = st.tabs(["Results", "ORB", "SIFT", "OpticalFlow"])
                
                with tabs[0]:
                    # Leaderboard
                    st.subheader("Leaderboard (ATE RMSE)")
                    leaderboard_data = leaderboard.get_leaderboard('ate_rmse')
                    
                    if leaderboard_data:
                        leaderboard_df = {
                            'Algorithm': [alg for alg, _ in leaderboard_data],
                            'ATE RMSE': [val for _, val in leaderboard_data]
                        }
                        st.dataframe(leaderboard_df, use_container_width=True)
                    
                    # Metrics comparison
                    st.subheader("Metrics Comparison")
                    
                    metrics_to_compare = ['ate_rmse', 'rpe_trans_rmse', 'rpe_rot_rmse', 'drift_mean']
                    
                    for metric in metrics_to_compare:
                        st.write(f"**{metric.replace('_', ' ').title()}**")
                        metric_data = {
                            'Algorithm': [],
                            'Value': []
                        }
                        
                        for alg in algorithms:
                            if metric in results[alg]['metrics']:
                                metric_data['Algorithm'].append(alg)
                                metric_data['Value'].append(results[alg]['metrics'][metric])
                        
                        if metric_data['Algorithm']:
                            st.bar_chart({alg: val for alg, val in zip(metric_data['Algorithm'], metric_data['Value'])})
                
                # Individual algorithm results
                for i, alg in enumerate(algorithms):
                    with tabs[i + 1]:
                        st.subheader(f"{alg} Results")
                        
                        # Metrics
                        metrics = results[alg]['metrics']
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("ATE RMSE", f"{metrics.get('ate_rmse', 0):.6f}")
                            st.metric("RPE Trans RMSE", f"{metrics.get('rpe_trans_rmse', 0):.6f}")
                        
                        with col2:
                            st.metric("RPE Rot RMSE", f"{metrics.get('rpe_rot_rmse', 0):.6f}")
                            st.metric("Drift Mean", f"{metrics.get('drift_mean', 0):.6f}")
                        
                        # Trajectory plot
                        st.subheader("Trajectory")
                        
                        fig = plt.figure(figsize=(10, 8))
                        ax = fig.add_subplot(111, projection='3d')
                        
                        # Extract positions
                        est_positions = np.array([pose.position for pose in results[alg]['poses']])
                        gt_positions = np.array([pose.position for pose in ground_truth_poses])
                        
                        # Plot trajectories
                        ax.plot(est_positions[:, 0], est_positions[:, 1], est_positions[:, 2], 
                                'b-', label='Estimated', linewidth=2)
                        ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 
                                'r--', label='Ground Truth', linewidth=2)
                        
                        ax.set_xlabel('X (m)')
                        ax.set_ylabel('Y (m)')
                        ax.set_zlabel('Z (m)')
                        ax.set_title(f'{alg} Visual Odometry Trajectory')
                        ax.legend()
                        ax.grid(True)
                        
                        st.pyplot(fig)
            
            else:
                # Single algorithm
                st.header(f"{algorithm} Visual Odometry Results")
                
                # Create VO algorithm
                vo = create_visual_odometry(algorithm, config)
                
                # Process images
                estimated_poses = []
                metrics = VisualOdometryMetrics()
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, image in enumerate(images):
                    pose_delta = vo.process_frame(image)
                    if pose_delta is not None:
                        estimated_poses.append(vo.current_pose)
                        metrics.add_pose(vo.current_pose, ground_truth_poses[i])
                    
                    progress_bar.progress((i + 1) / len(images))
                    status_text.text(f"Processing frame {i + 1}/{len(images)}")
                
                progress_bar.empty()
                status_text.empty()
                
                # Compute metrics
                algorithm_metrics = metrics.compute_all_metrics()
                
                # Display metrics
                st.subheader("Evaluation Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("ATE RMSE", f"{algorithm_metrics.get('ate_rmse', 0):.6f}")
                    st.metric("ATE Mean", f"{algorithm_metrics.get('ate_mean', 0):.6f}")
                
                with col2:
                    st.metric("RPE Trans RMSE", f"{algorithm_metrics.get('rpe_trans_rmse', 0):.6f}")
                    st.metric("RPE Rot RMSE", f"{algorithm_metrics.get('rpe_rot_rmse', 0):.6f}")
                
                with col3:
                    st.metric("Drift Mean", f"{algorithm_metrics.get('drift_mean', 0):.6f}")
                    st.metric("Drift Max", f"{algorithm_metrics.get('drift_max', 0):.6f}")
                
                with col4:
                    st.metric("Trajectory Length", f"{algorithm_metrics.get('estimated_length', 0):.3f}")
                    st.metric("Length Ratio", f"{algorithm_metrics.get('length_ratio', 0):.3f}")
                
                # Trajectory visualization
                st.subheader("Trajectory Visualization")
                
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Extract positions
                est_positions = np.array([pose.position for pose in estimated_poses])
                gt_positions = np.array([pose.position for pose in ground_truth_poses])
                
                # Plot trajectories
                ax.plot(est_positions[:, 0], est_positions[:, 1], est_positions[:, 2], 
                        'b-', label='Estimated', linewidth=2)
                ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 
                        'r--', label='Ground Truth', linewidth=2)
                
                # Mark start and end points
                ax.scatter(est_positions[0, 0], est_positions[0, 1], est_positions[0, 2], 
                           'go', s=100, label='Start')
                ax.scatter(est_positions[-1, 0], est_positions[-1, 1], est_positions[-1, 2], 
                           'ro', s=100, label='End')
                
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_zlabel('Z (m)')
                ax.set_title(f'{algorithm} Visual Odometry Trajectory')
                ax.legend()
                ax.grid(True)
                
                st.pyplot(fig)
                
                # Trajectory statistics
                st.subheader("Trajectory Statistics")
                stats = compute_trajectory_statistics(estimated_poses)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Position Statistics**")
                    for key, value in stats.items():
                        if isinstance(value, dict) and 'position' in key:
                            st.write(f"- {key}: {value}")
                
                with col2:
                    st.write("**Motion Statistics**")
                    for key, value in stats.items():
                        if not isinstance(value, dict):
                            st.write(f"- {key}: {value}")
    
    # System information
    st.sidebar.header("System Information")
    device_info = get_device_info()
    
    st.sidebar.write("**Platform:**", device_info.get('platform', 'Unknown'))
    st.sidebar.write("**Python:**", device_info.get('python_version', 'Unknown'))
    st.sidebar.write("**OpenCV:**", device_info.get('opencv_version', 'Unknown'))
    st.sidebar.write("**NumPy:**", device_info.get('numpy_version', 'Unknown'))
    
    if device_info.get('cuda_available'):
        st.sidebar.write("**CUDA:**", "Available")
        st.sidebar.write("**GPU:**", device_info.get('cuda_device_name', 'Unknown'))
    else:
        st.sidebar.write("**CUDA:**", "Not Available")


if __name__ == "__main__":
    main()
