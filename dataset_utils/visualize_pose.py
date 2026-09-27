import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
from pathlib import Path
import argparse

def find_pose_files(root_folder):
    """Find all poses.npy files in the folder structure."""
    pose_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file == 'poses.npy':
                pose_files.append(os.path.join(root, file))
    return pose_files

def load_poses(pose_file):
    """Load pose matrices from a .npy file."""
    try:
        poses = np.load(pose_file)
        if poses.shape[-2:] != (4, 4):
            print(f"Warning: {pose_file} doesn't contain 4x4 matrices. Shape: {poses.shape}")
            return None
        return poses
    except Exception as e:
        print(f"Error loading {pose_file}: {e}")
        return None

def extract_positions_and_orientations(poses):
    """Extract camera positions and orientations from pose matrices."""
    positions = poses[:, :3, 3]  # Translation part (x, y, z)
    
    # Extract orientation (forward direction from rotation matrix)
    # Using the negative z-axis (third column) as the camera forward direction
    forward_dirs = -poses[:, :3, 2]
    
    return positions, forward_dirs

def plot_poses_3d(all_poses_data, output_path=None, show_orientations=True, arrow_length=0.1, equal_aspect=False):
    """Create a 3D plot of all camera poses."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_poses_data)))
    
    for i, (folder_name, poses) in enumerate(all_poses_data.items()):
        positions, forward_dirs = extract_positions_and_orientations(poses)
        
        # Plot camera positions
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                  c=[colors[i]], label=f'{folder_name} ({len(positions)} poses)', 
                  s=30, alpha=0.7)
        
        # Plot trajectory
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
               color=colors[i], alpha=0.5, linewidth=1)
        
        # Plot orientation arrows (optional)
        if show_orientations and len(positions) <= 100:  # Limit arrows for readability
            step = max(1, len(positions) // 20)  # Show max 20 arrows per trajectory
            for j in range(0, len(positions), step):
                ax.quiver(positions[j, 0], positions[j, 1], positions[j, 2],
                         forward_dirs[j, 0], forward_dirs[j, 1], forward_dirs[j, 2],
                         length=arrow_length, color=colors[i], alpha=0.6, arrow_length_ratio=0.3)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Camera Poses Visualization')
    ax.legend()
    
    # Better axis scaling - use actual data bounds with small padding
    all_positions = []
    for poses in all_poses_data.values():
        positions, _ = extract_positions_and_orientations(poses)
        all_positions.append(positions)
    
    if all_positions:
        all_positions = np.vstack(all_positions)
        
        # Get actual data bounds
        x_min, x_max = np.min(all_positions[:, 0]), np.max(all_positions[:, 0])
        y_min, y_max = np.min(all_positions[:, 1]), np.max(all_positions[:, 1])
        z_min, z_max = np.min(all_positions[:, 2]), np.max(all_positions[:, 2])
        
        # Add small padding (5% of range)
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        padding_x = x_range * 0.05 if x_range > 0 else 0.1
        padding_y = y_range * 0.05 if y_range > 0 else 0.1
        padding_z = z_range * 0.05 if z_range > 0 else 0.1
        
        ax.set_xlim([x_min - padding_x, x_max + padding_x])
        ax.set_ylim([y_min - padding_y, y_max + padding_y])
        ax.set_zlim([z_min - padding_z, z_max + padding_z])
        
        # For equal aspect ratio (optional), use the largest range
        if equal_aspect:
            max_range = max(x_range, y_range, z_range) / 2
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            center_z = (z_min + z_max) / 2
            
            ax.set_xlim([center_x - max_range, center_x + max_range])
            ax.set_ylim([center_y - max_range, center_y + max_range])
            ax.set_zlim([center_z - max_range, center_z + max_range])
    
    # Save the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    return fig, ax

def plot_poses_2d_projections(all_poses_data, output_path=None):
    """Create 2D projection plots (XY, XZ, YZ) of all camera poses."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_poses_data)))
    
    # Define projection axes and labels
    projections = [
        (0, 1, 'X', 'Y', 'XY Projection (Top View)'),
        (0, 2, 'X', 'Z', 'XZ Projection (Front View)'),
        (1, 2, 'Y', 'Z', 'YZ Projection (Side View)')
    ]
    
    for proj_idx, (axis1, axis2, label1, label2, title) in enumerate(projections):
        ax = axes[proj_idx]
        
        for i, (folder_name, poses) in enumerate(all_poses_data.items()):
            positions, _ = extract_positions_and_orientations(poses)
            
            # Plot camera positions
            ax.scatter(positions[:, axis1], positions[:, axis2], 
                      c=[colors[i]], label=f'{folder_name}' if proj_idx == 0 else None,
                      s=30, alpha=0.7)
            
            # Plot trajectory
            ax.plot(positions[:, axis1], positions[:, axis2], 
                   color=colors[i], alpha=0.5, linewidth=1)
            
            # Mark start and end points
            ax.scatter(positions[0, axis1], positions[0, axis2], 
                      c=[colors[i]], marker='o', s=100, edgecolors='black', linewidths=2, alpha=0.9)
            ax.scatter(positions[-1, axis1], positions[-1, axis2], 
                      c=[colors[i]], marker='s', s=100, edgecolors='black', linewidths=2, alpha=0.9)
        
        ax.set_xlabel(label1, fontsize=12)
        ax.set_ylabel(label2, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        if proj_idx == 0:
            ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"2D projections plot saved to: {output_path}")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize 3D poses from poses.npy files')
    parser.add_argument('--folder', help='Root folder containing subfolders with poses.npy files')
    parser.add_argument('--output', '-o', default='poses_3d_visualization.png', 
                       help='Output image filename (default: poses_3d_visualization.png)')
    parser.add_argument('--no-orientations', action='store_true', 
                       help='Don\'t show orientation arrows')
    parser.add_argument('--arrow-length', type=float, default=0.1,
                       help='Length of orientation arrows (default: 0.1)')
    parser.add_argument('--equal-aspect', action='store_true',
                       help='Use equal aspect ratio for all axes')
    parser.add_argument('--show', action='store_true',
                       help='Show the plot interactively')
    parser.add_argument('--volumes', type=str, default=None,
                       help='Comma-separated list of volume numbers to visualize (e.g., "0,20,110")')
    parser.add_argument('--volume-prefix', type=str, default='volume',
                       help='Prefix for volume directories (default: "volume")')
    
    args = parser.parse_args()
    
    # Parse volume filter if provided
    volume_filter = None
    if args.volumes:
        print(f"Volume filter provided: {args.volumes}")
        try:
            volume_filter = set([int(v.strip()) for v in args.volumes.split(',')])
            print(f"Filtering to volumes: {sorted(volume_filter)}")
        except ValueError:    
            volume_filter = None
            print(f"Filtering to volumes: all (no specific filter applied)")
    
    # Find all poses.npy files
    print(f"Searching for poses.npy files in: {args.folder}")
    pose_files = find_pose_files(args.folder)
    
    if not pose_files:
        print("No poses.npy files found!")
        return
    
    # Filter by volume numbers if specified
    if volume_filter is not None:
        filtered_pose_files = []
        for pose_file in pose_files:
            folder_name = os.path.basename(os.path.dirname(pose_file))
            # Extract volume number
            if folder_name.startswith(args.volume_prefix):
                try:
                    vol_num = int(folder_name.replace(args.volume_prefix, ''))
                    if vol_num in volume_filter:
                        filtered_pose_files.append(pose_file)
                except ValueError:
                    continue
        pose_files = filtered_pose_files
        
        if not pose_files:
            print(f"No volumes found matching filter: {sorted(volume_filter)}")
            return
    
    print(f"Found {len(pose_files)} poses.npy files:")
    for file in pose_files:
        print(f"  - {file}")
    
    # Load all pose data
    all_poses_data = {}
    total_poses = 0
    
    for pose_file in pose_files:
        poses = load_poses(pose_file)
        if poses is not None:
            folder_name = os.path.basename(os.path.dirname(pose_file))
            if folder_name in all_poses_data:
                # If folder name exists, append a number
                counter = 1
                original_name = folder_name
                while folder_name in all_poses_data:
                    folder_name = f"{original_name}_{counter}"
                    counter += 1
            
            all_poses_data[folder_name] = poses
            total_poses += len(poses)
            print(f"Loaded {len(poses)} poses from {folder_name}")
    
    if not all_poses_data:
        print("No valid pose data found!")
        return
    
    print(f"\nTotal poses loaded: {total_poses}")
    
    # Create 3D visualization
    print("Creating 3D visualization...")
    fig, ax = plot_poses_3d(all_poses_data, 
                           output_path=args.output,
                           show_orientations=not args.no_orientations,
                           arrow_length=args.arrow_length,
                           equal_aspect=args.equal_aspect)
    
    # Create 2D projections figure
    print("Creating 2D projections...")
    output_2d = args.output.replace('.png', '_2d_projections.png')
    fig_2d = plot_poses_2d_projections(all_poses_data, output_path=output_2d)
    
    # Show plot if requested
    if args.show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(fig_2d)
    
    print("Done!")

if __name__ == "__main__":
    main()