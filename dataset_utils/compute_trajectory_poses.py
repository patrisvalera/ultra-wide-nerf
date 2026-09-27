"""
Compute Trajectory Poses from Multiple Volumes

This script creates a non-overlapping trajectory from poses across multiple volumes.
Volumes are sorted along a specified axis (x, y, or z), and overlapping poses are removed.

For each volume, only poses beyond the previous volume's extent are kept, creating
a continuous non-overlapping trajectory along the sorted axis.

Usage:
    python compute_trajectory_poses.py --data_folder ./data --axis z --output_path trajectory_poses.npy
    python compute_trajectory_poses.py --data_folder ./data --axis x --volumes 0,5,10,15
"""

import numpy as np
from typing import Tuple, Optional, List, Union
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse


def load_poses_from_volumes(root_folder: str, 
                            volume_filter: Optional[set] = None, 
                            volume_prefix: str = 'volume') -> Tuple[List[np.ndarray], List[str], List[int]]:
    """
    Load poses.npy files from volume subfolders.
    
    Args:
        root_folder: Path to the root folder containing volume subfolders
        volume_filter: Optional set of volume numbers to include
        volume_prefix: Prefix for volume directories (default: "volume")
        
    Returns:
        poses_list: List of pose arrays, one per volume
        volume_names: List of volume folder names
        volume_numbers: List of volume numbers (extracted from folder names)
    """
    root_path = Path(root_folder)
    
    if not root_path.exists():
        raise FileNotFoundError(f"Folder not found: {root_folder}")
    
    poses_list = []
    volume_names = []
    volume_numbers = []
    
    # Find all volume subfolders
    #volume_folders = sorted([f for f in root_path.iterdir() 
    #                        if f.is_dir() and f.name.startswith(volume_prefix)])
    # Sort based on what comes after "volume"
    volume_folders = sorted([f for f in root_path.iterdir() 
                        if f.is_dir() and f.name.startswith(volume_prefix)],
                        key=lambda x: int(x.name.replace(volume_prefix, '')))
    
    if volume_filter is not None:
        print(f"Filtering volumes: {sorted(volume_filter)}")
        volume_folders = [f for f in volume_folders 
                         if int(f.name.replace(volume_prefix, '')) in volume_filter]
        print(f"Volumes after filtering: {[f.name for f in volume_folders]}")
    
    print(f"\nSearching for poses.npy in {len(volume_folders)} volume folders...")
    print(f"{'='*70}")
    
    for volume_folder in volume_folders:
        pose_file = volume_folder / 'poses.npy'
        
        if pose_file.exists():
            try:
                poses = np.load(str(pose_file))
                original_shape = poses.shape
                
                # Validate shape but keep original format
                if poses.ndim == 3 and poses.shape[-2:] == (3, 4):
                    # Shape [N, 3, 4] - valid
                    pass
                elif poses.ndim == 4 and poses.shape[-2:] == (3, 4):
                    # Shape [N, D, 3, 4] - valid
                    pass
                elif poses.ndim == 2 and poses.shape == (3, 4):
                    # Single pose [3, 4] - valid
                    pass
                elif poses.ndim == 3 and poses.shape[-2:] == (4, 4):
                    # Shape [N, 4, 4] - valid
                    pass
                elif poses.ndim == 4 and poses.shape[-2:] == (4, 4):
                    # Shape [N, D, 4, 4] - valid
                    pass
                else:
                    print(f"  WARNING: Unexpected pose shape {poses.shape} in {volume_folder.name}, skipping")
                    continue
                
                volume_num = int(volume_folder.name.replace(volume_prefix, ''))
                poses_list.append(poses)
                volume_names.append(volume_folder.name)
                volume_numbers.append(volume_num)
                
                print(f"  ✓ {volume_folder.name}: shape {original_shape}, {poses.shape[0]} poses loaded")
                
            except Exception as e:
                print(f"  ✗ {volume_folder.name}: Failed to load - {e}")
        else:
            print(f"  ✗ {volume_folder.name}: poses.npy not found")
    
    if not poses_list:
        raise ValueError(f"No valid poses found in {root_folder}")
    
    print(f"{'='*70}")
    print(f"Total volumes loaded: {len(poses_list)}")
    
    return poses_list, volume_names, volume_numbers


def compute_volume_center_positions(poses_list: List[np.ndarray],
                                   volume_numbers: List[int]) -> Tuple[np.ndarray, List[int]]:
    """
    Compute the center position of each volume for sorting.
    
    Args:
        poses_list: List of pose arrays, one per volume (various shapes supported)
        volume_numbers: List of volume numbers
        
    Returns:
        center_positions: Array of shape [M, 3] with center position per volume
        volume_numbers: List of volume numbers
    """
    M = len(poses_list)
    center_positions = np.zeros((M, 3))
    
    print(f"\n{'='*70}")
    print(f"Computing volume center positions")
    print(f"{'='*70}")
    
    for i, (poses, vol_num) in enumerate(zip(poses_list, volume_numbers)):
        # Extract translation vectors regardless of shape
        if poses.ndim == 2:
            # Single pose [3, 4] or [4, 4]
            center = poses[:3, 3]
        elif poses.ndim == 3:
            # [N, 3, 4] or [N, 4, 4]
            center = poses[:, :3, 3].mean(axis=0)
        elif poses.ndim == 4:
            # [N, D, 3, 4] or [N, D, 4, 4]
            positions = poses[:, :, :3, 3].reshape(-1, 3)
            center = positions.mean(axis=0)
        else:
            raise ValueError(f"Unexpected pose shape: {poses.shape}")
        
        center_positions[i] = center
        
        print(f"  Volume {vol_num:3d}: center = [{center[0]:8.3f}, {center[1]:8.3f}, {center[2]:8.3f}], shape {poses.shape}")
    
    return center_positions, volume_numbers


def concatenate_poses_bidirectional(poses_list: List[np.ndarray],
                                    volume_numbers: List[int],
                                    center_positions: np.ndarray,
                                    sort_indices: np.ndarray,
                                    axis_idx: int,
                                    sort_axis: str,
                                    reverse: bool) -> Tuple[np.ndarray, dict]:
    """
    Concatenate poses using bidirectional strategy: next_to_prior from start, last_to_first from end.
    
    This strategy splits volumes at the middle:
    - First half: processes forward using next_to_prior logic (each keeps poses beyond previous)
    - Second half: processes backward using last_to_first logic (each keeps poses before next)
    - Balances pose distribution from both trajectory ends toward the center
    
    Args:
        poses_list: List of pose arrays
        volume_numbers: List of volume numbers
        center_positions: Center positions of volumes
        sort_indices: Sorted indices for volumes
        axis_idx: Index of sorting axis (0=x, 1=y, 2=z)
        sort_axis: Name of sorting axis
        reverse: Whether sorting in reverse order
        
    Returns:
        trajectory_poses: Concatenated trajectory poses
        trajectory_info: Metadata dictionary
    """
    n_volumes = len(sort_indices)
    middle_idx = n_volumes // 2
    
    # Split into two groups - keep them completely separate
    # First half includes middle, second half excludes middle completely
    first_half_indices = sort_indices[:middle_idx+1]  # Include middle in first half
    second_half_indices = sort_indices[middle_idx+1:]  # Start AFTER middle (no overlap)
    
    print(f"\n  Split point: volume index {middle_idx} (middle volume included in first half)")
    print(f"  First half: {len(first_half_indices)} volumes")
    print(f"  Second half: {len(second_half_indices)} volumes")
    
    print(f"\n  Processing FIRST HALF ({len(first_half_indices)} volumes, including middle): next_to_prior (forward)")
    print(f"  {'='*68}")
    
    # Process first half: next_to_prior (forward)
    first_half_results = []
    first_half_counts = []
    first_half_counts_original = []
    first_half_volume_numbers = []
    last_axis_value = None
    
    for i, idx in enumerate(first_half_indices):
        poses = poses_list[idx]
        vol_num = volume_numbers[idx]
        center = center_positions[idx]
        
        # Extract axis coordinates
        if poses.ndim == 2:
            axis_coords = np.array([poses[axis_idx, 3]])
            original_count = 1
        elif poses.ndim == 3:
            axis_coords = poses[:, axis_idx, 3]
            original_count = poses.shape[0]
        elif poses.ndim == 4:
            axis_coords = poses[:, :, axis_idx, 3].flatten()
            original_count = poses.shape[0] * poses.shape[1]
        else:
            raise ValueError(f"Unexpected pose shape: {poses.shape}")
        
        if i == 0:
            # First volume: keep all
            filtered_poses = poses
            filtered_count = original_count
        else:
            # next_to_prior: keep poses beyond last_axis_value
            if reverse:
                mask = axis_coords < last_axis_value
            else:
                mask = axis_coords > last_axis_value
            
            # Apply mask
            if poses.ndim == 2:
                filtered_poses = poses if mask[0] else None
                filtered_count = 1 if mask[0] else 0
            elif poses.ndim == 3:
                filtered_poses = poses[mask]
                filtered_count = filtered_poses.shape[0]
            elif poses.ndim == 4:
                N, D = poses.shape[:2]
                filtered_flat_indices = np.where(mask)[0]
                if len(filtered_flat_indices) > 0:
                    n_indices = filtered_flat_indices // D
                    unique_n = np.unique(n_indices)
                    filtered_poses = poses[unique_n]
                    filtered_count = len(unique_n) * D
                else:
                    filtered_poses = None
                    filtered_count = 0
        
        # Update threshold
        if filtered_poses is not None and filtered_count > 0:
            if poses.ndim == 2:
                last_axis_value = filtered_poses[axis_idx, 3]
            elif poses.ndim == 3:
                last_axis_value = filtered_poses[:, axis_idx, 3].max() if not reverse else filtered_poses[:, axis_idx, 3].min()
            elif poses.ndim == 4:
                coords = filtered_poses[:, :, axis_idx, 3].flatten()
                last_axis_value = coords.max() if not reverse else coords.min()
            
            first_half_results.append(filtered_poses)
            first_half_counts.append(filtered_count)
            first_half_counts_original.append(original_count)
            first_half_volume_numbers.append(vol_num)
            
            print(f"    Volume {vol_num:3d}: {filtered_count:4d}/{original_count:4d} poses kept (forward)")
        else:
            print(f"    Volume {vol_num:3d}: {0:4d}/{original_count:4d} poses kept (all filtered)")
    
    # Get the boundary from first half to use as starting threshold for second half
    # This prevents duplicates at the middle
    first_half_boundary = last_axis_value if last_axis_value is not None else None
    
    print(f"\n  Processing SECOND HALF ({len(second_half_indices)} volumes): last_to_first (backward)")
    if first_half_boundary is not None:
        print(f"  Starting from boundary: {first_half_boundary:.3f} (no overlap with first half)")
    print(f"  {'='*68}")
    
    # Process second half: last_to_first (backward from end)
    second_half_results = []
    second_half_counts = []
    second_half_counts_original = []
    second_half_volume_numbers = []
    last_axis_value = None
    
    # Process in reverse order for second half
    reversed_second_half = list(reversed(second_half_indices))
    
    for i, idx in enumerate(reversed_second_half):
        poses = poses_list[idx]
        vol_num = volume_numbers[idx]
        center = center_positions[idx]
        
        # Extract axis coordinates
        if poses.ndim == 2:
            axis_coords = np.array([poses[axis_idx, 3]])
            original_count = 1
        elif poses.ndim == 3:
            axis_coords = poses[:, axis_idx, 3]
            original_count = poses.shape[0]
        elif poses.ndim == 4:
            axis_coords = poses[:, :, axis_idx, 3].flatten()
            original_count = poses.shape[0] * poses.shape[1]
        else:
            raise ValueError(f"Unexpected pose shape: {poses.shape}")
        
        if i == 0:
            # Last volume of second half: keep all poses
            filtered_poses = poses
            filtered_count = original_count
        else:
            # last_to_first: keep poses before last_axis_value (going backward)
            if reverse:
                mask = axis_coords > last_axis_value
            else:
                mask = axis_coords < last_axis_value
            
            # Apply mask
            if poses.ndim == 2:
                filtered_poses = poses if mask[0] else None
                filtered_count = 1 if mask[0] else 0
            elif poses.ndim == 3:
                filtered_poses = poses[mask]
                filtered_count = filtered_poses.shape[0]
            elif poses.ndim == 4:
                N, D = poses.shape[:2]
                filtered_flat_indices = np.where(mask)[0]
                if len(filtered_flat_indices) > 0:
                    n_indices = filtered_flat_indices // D
                    unique_n = np.unique(n_indices)
                    filtered_poses = poses[unique_n]
                    filtered_count = len(unique_n) * D
                else:
                    filtered_poses = None
                    filtered_count = 0
        
        # Additional check: filter out overlap with first half boundary
        if filtered_poses is not None and filtered_count > 0 and first_half_boundary is not None:
            # Get axis coords of filtered poses
            if filtered_poses.ndim == 2:
                filtered_axis_coords = np.array([filtered_poses[axis_idx, 3]])
            elif filtered_poses.ndim == 3:
                filtered_axis_coords = filtered_poses[:, axis_idx, 3]
            elif filtered_poses.ndim == 4:
                filtered_axis_coords = filtered_poses[:, :, axis_idx, 3].flatten()
            
            # Check for overlap with first half
            if reverse:
                # In reverse: keep only poses < first_half_boundary (strictly less than)
                boundary_mask = filtered_axis_coords < first_half_boundary
            else:
                # In forward: keep only poses > first_half_boundary (strictly greater than)
                boundary_mask = filtered_axis_coords > first_half_boundary
            
            # Apply boundary mask if there's overlap
            if not boundary_mask.all():
                if filtered_poses.ndim == 2:
                    filtered_poses = filtered_poses if boundary_mask[0] else None
                    filtered_count = 1 if boundary_mask[0] else 0
                elif filtered_poses.ndim == 3:
                    filtered_poses = filtered_poses[boundary_mask]
                    filtered_count = filtered_poses.shape[0]
                elif filtered_poses.ndim == 4:
                    N, D = filtered_poses.shape[:2]
                    filtered_flat_indices = np.where(boundary_mask)[0]
                    if len(filtered_flat_indices) > 0:
                        n_indices = filtered_flat_indices // D
                        unique_n = np.unique(n_indices)
                        filtered_poses = filtered_poses[unique_n]
                        filtered_count = len(unique_n) * D
                    else:
                        filtered_poses = None
                        filtered_count = 0
        
        # Update threshold for next iteration (for last_to_first logic within second half)
        if filtered_poses is not None and filtered_count > 0:
            if poses.ndim == 2:
                last_axis_value = filtered_poses[axis_idx, 3]
            elif poses.ndim == 3:
                last_axis_value = filtered_poses[:, axis_idx, 3].min() if not reverse else filtered_poses[:, axis_idx, 3].max()
            elif poses.ndim == 4:
                coords = filtered_poses[:, :, axis_idx, 3].flatten()
                last_axis_value = coords.min() if not reverse else coords.max()
            
            second_half_results.append(filtered_poses)
            second_half_counts.append(filtered_count)
            second_half_counts_original.append(original_count)
            second_half_volume_numbers.append(vol_num)
            
            print(f"    Volume {vol_num:3d}: {filtered_count:4d}/{original_count:4d} poses kept (backward)")
        else:
            print(f"    Volume {vol_num:3d}: {0:4d}/{original_count:4d} poses kept (all filtered)")
    
    # Reverse second half results to maintain sort order
    second_half_results = list(reversed(second_half_results))
    second_half_counts = list(reversed(second_half_counts))
    second_half_counts_original = list(reversed(second_half_counts_original))
    second_half_volume_numbers = list(reversed(second_half_volume_numbers))
    
    # Combine both halves
    trajectory_poses_list = first_half_results + second_half_results
    volume_pose_counts = first_half_counts + second_half_counts
    volume_pose_counts_original = first_half_counts_original + second_half_counts_original
    sorted_volume_numbers = first_half_volume_numbers + second_half_volume_numbers
    
    # Compute start indices
    volume_start_indices = []
    current_idx = 0
    for count in volume_pose_counts:
        volume_start_indices.append(current_idx)
        current_idx += count
    
    if not trajectory_poses_list:
        raise ValueError("No poses remain after bidirectional filtering.")
    
    # Concatenate all poses
    first_shape = trajectory_poses_list[0].shape
    
    if first_shape[-2:] == (3, 4):
        trajectory_poses = np.concatenate(trajectory_poses_list, axis=0)
    elif first_shape[-2:] == (4, 4):
        trajectory_poses = np.concatenate(trajectory_poses_list, axis=0)
    else:
        raise ValueError(f"Unexpected pose shape: {first_shape}")
    
    # Compute trajectory statistics
    if trajectory_poses.ndim == 2:
        all_positions = trajectory_poses[:3, 3].reshape(1, 3)
    elif trajectory_poses.ndim == 3:
        all_positions = trajectory_poses[:, :3, 3]
    elif trajectory_poses.ndim == 4:
        all_positions = trajectory_poses[:, :, :3, 3].reshape(-1, 3)
    
    trajectory_info = {
        'num_volumes': len(trajectory_poses_list),
        'total_poses': sum(volume_pose_counts),
        'total_poses_original': sum(volume_pose_counts_original),
        'overlap_removed_count': sum(volume_pose_counts_original) - sum(volume_pose_counts),
        'sort_axis': sort_axis,
        'sort_reverse': reverse,
        'overlap_strategy': 'bidirectional',
        'middle_split_index': middle_idx,
        'first_half_volumes': len(first_half_results),
        'second_half_volumes': len(second_half_results),
        'volume_numbers': sorted_volume_numbers,
        'volume_pose_counts': volume_pose_counts,
        'volume_pose_counts_original': volume_pose_counts_original,
        'volume_start_indices': volume_start_indices,
        'center_positions': center_positions[sort_indices[:len(sorted_volume_numbers)]],
        'output_shape': trajectory_poses.shape,
        'x_range': (all_positions[:, 0].min(), all_positions[:, 0].max()),
        'y_range': (all_positions[:, 1].min(), all_positions[:, 1].max()),
        'z_range': (all_positions[:, 2].min(), all_positions[:, 2].max()),
        'x_span': all_positions[:, 0].max() - all_positions[:, 0].min(),
        'y_span': all_positions[:, 1].max() - all_positions[:, 1].min(),
        'z_span': all_positions[:, 2].max() - all_positions[:, 2].min(),
    }
    
    print(f"\n{'='*70}")
    print(f"Trajectory Summary (Bidirectional):")
    print(f"  Volumes contributing: {trajectory_info['num_volumes']}")
    print(f"  First half (next_to_prior): {trajectory_info['first_half_volumes']} volumes")
    print(f"  Second half (last_to_first): {trajectory_info['second_half_volumes']} volumes")
    print(f"  Overlap strategy: bidirectional")
    print(f"  Output shape: {trajectory_info['output_shape']}")
    print(f"  Total poses (after overlap removal): {trajectory_info['total_poses']}")
    print(f"  Total poses (before filtering): {trajectory_info['total_poses_original']}")
    print(f"  Overlapping poses removed: {trajectory_info['overlap_removed_count']}")
    print(f"  X range: [{trajectory_info['x_range'][0]:.3f}, {trajectory_info['x_range'][1]:.3f}] (span: {trajectory_info['x_span']:.3f})")
    print(f"  Y range: [{trajectory_info['y_range'][0]:.3f}, {trajectory_info['y_range'][1]:.3f}] (span: {trajectory_info['y_span']:.3f})")
    print(f"  Z range: [{trajectory_info['z_range'][0]:.3f}, {trajectory_info['z_range'][1]:.3f}] (span: {trajectory_info['z_span']:.3f})")
    print(f"{'='*70}")
    
    return trajectory_poses, trajectory_info


def concatenate_poses_along_trajectory(poses_list: List[np.ndarray],
                                       volume_numbers: List[int],
                                       center_positions: np.ndarray,
                                       sort_axis: str = 'z',
                                       reverse: bool = False,
                                       overlap_strategy: str = 'next_to_prior') -> Tuple[np.ndarray, dict]:
    """
    Concatenate poses from volumes ordered along the trajectory, removing overlaps.
    
    Volumes are sorted by their center position along the specified axis.
    Overlapping poses between volumes are removed using the specified strategy.
    
    Overlap strategies:
    - 'next_to_prior': Keep only poses beyond the previous volume's extent (default)
                       Results in poses adjacent to prior volume, minimal overlap
    - 'last_to_first': Process volumes from last to first, keeping poses from larger to smaller axis values
                       Results in maximum poses from the last volume in sorted order
    - 'bidirectional': Combines both strategies, splitting at the middle volume
                       First half uses next_to_prior (forward), second half uses last_to_first (backward)
                       Balances pose distribution from both ends toward the center
    
    This creates a non-overlapping trajectory where each volume contributes
    only its non-overlapping portion. The original shape of poses is preserved.
    
    Args:
        poses_list: List of pose arrays, one per volume (various shapes)
        volume_numbers: List of volume numbers
        center_positions: Center position of each volume [M, 3]
        sort_axis: Axis to sort along ('x', 'y', or 'z')
        reverse: If True, sort in descending order
        overlap_strategy: Strategy for handling overlaps ('next_to_prior', 'last_to_first', or 'bidirectional')
        
    Returns:
        trajectory_poses: Array with non-overlapping poses (shape preserved from input)
        trajectory_info: Dictionary with trajectory metadata
    """
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if sort_axis.lower() not in axis_map:
        raise ValueError(f"Invalid axis: {sort_axis}. Must be 'x', 'y', or 'z'")
    
    if overlap_strategy not in ['next_to_prior', 'last_to_first', 'bidirectional']:
        raise ValueError(f"Invalid overlap_strategy: {overlap_strategy}. Must be 'next_to_prior', 'last_to_first', or 'bidirectional'")
    
    axis_idx = axis_map[sort_axis.lower()]
    
    # Sort volumes by their center position along the specified axis
    sort_indices = np.argsort(center_positions[:, axis_idx])
    if reverse:
        sort_indices = sort_indices[::-1]
    
    print(f"\n{'='*70}")
    print(f"Sorting volumes along {sort_axis.upper()}-axis ({'descending' if reverse else 'ascending'})")
    print(f"Removing overlapping poses between volumes")
    print(f"Overlap strategy: {overlap_strategy}")
    if overlap_strategy == 'last_to_first':
        print(f"  Processing from last to first volume (larger to smaller {sort_axis.upper()} values)")
    elif overlap_strategy == 'bidirectional':
        middle_idx = len(sort_indices) // 2
        print(f"  Processing bidirectionally: first half forward (next_to_prior), second half backward (last_to_first)")
        print(f"  Split at middle volume (index {middle_idx}/{len(sort_indices)})")
    print(f"{'='*70}")
    
    # Handle bidirectional strategy separately
    if overlap_strategy == 'bidirectional':
        return concatenate_poses_bidirectional(
            poses_list, volume_numbers, center_positions,
            sort_indices, axis_idx, sort_axis, reverse
        )
    
    # Concatenate poses in sorted order, removing overlaps
    trajectory_poses_list = []
    volume_pose_counts = []
    volume_pose_counts_original = []
    volume_pose_counts_filtered = []
    sorted_volume_numbers = []
    volume_start_indices = []
    current_index = 0
    
    last_axis_value = None  # Track the threshold axis value
    
    # For last_to_first strategy, process volumes in reverse order
    if overlap_strategy == 'last_to_first':
        process_indices = list(reversed(sort_indices))
    else:
        process_indices = sort_indices
    # For last_to_first strategy, process volumes in reverse order
    if overlap_strategy == 'last_to_first':
        process_indices = list(reversed(sort_indices))
    else:
        process_indices = sort_indices
    
    for i, idx in enumerate(process_indices):
        poses = poses_list[idx]
        vol_num = volume_numbers[idx]
        center = center_positions[idx]
        
        # Extract axis coordinates based on shape
        if poses.ndim == 2:
            # Single pose [3, 4] or [4, 4]
            axis_coords = np.array([poses[axis_idx, 3]])
            original_count = 1
        elif poses.ndim == 3:
            # [N, 3, 4] or [N, 4, 4]
            axis_coords = poses[:, axis_idx, 3]
            original_count = poses.shape[0]
        elif poses.ndim == 4:
            # [N, D, 3, 4] or [N, D, 4, 4]
            # Flatten to [N*D] for filtering
            axis_coords = poses[:, :, axis_idx, 3].flatten()
            original_count = poses.shape[0] * poses.shape[1]
        else:
            raise ValueError(f"Unexpected pose shape: {poses.shape}")
        
        if last_axis_value is None:
            # First volume (or last in last_to_first): keep all poses
            filtered_poses = poses
            filtered_count = original_count
        else:
            # Filter poses based on overlap strategy
            if overlap_strategy == 'next_to_prior':
                # Keep only poses beyond the last volume's extent (strict no overlap)
                if reverse:
                    # In reverse order, keep poses with axis value < last_axis_value
                    mask = axis_coords < last_axis_value
                else:
                    # In forward order, keep poses with axis value > last_axis_value
                    mask = axis_coords > last_axis_value
            
            elif overlap_strategy == 'last_to_first':
                # Processing from last to first, so keep poses before (smaller than) the current threshold
                # This works opposite to next_to_prior
                if reverse:
                    # In reverse sort order, going backwards means keeping larger values
                    mask = axis_coords > last_axis_value
                else:
                    # In forward sort order, going backwards means keeping smaller values
                    mask = axis_coords < last_axis_value
            
            # Apply mask based on shape
            if poses.ndim == 2:
                # Single pose
                if mask[0]:
                    filtered_poses = poses
                    filtered_count = 1
                else:
                    filtered_poses = None
                    filtered_count = 0
            elif poses.ndim == 3:
                # [N, 3, 4] or [N, 4, 4]
                filtered_poses = poses[mask]
                filtered_count = filtered_poses.shape[0]
            elif poses.ndim == 4:
                # [N, D, 3, 4] or [N, D, 4, 4]
                # Reshape mask to [N, D]
                N, D = poses.shape[:2]
                mask_2d = mask.reshape(N, D)
                # Keep rows where at least one element in D dimension passes
                # OR filter per N (first dimension)
                # For now, filter in flattened space then reshape
                filtered_flat_indices = np.where(mask)[0]
                if len(filtered_flat_indices) > 0:
                    # Convert flat indices back to [N, D] indices
                    n_indices = filtered_flat_indices // D
                    d_indices = filtered_flat_indices % D
                    # Create new array with filtered poses
                    unique_n = np.unique(n_indices)
                    # Keep full volumes (N dimension) if any D slice passes
                    filtered_poses = poses[unique_n]
                    filtered_count = len(unique_n) * D
                else:
                    filtered_poses = None
                    filtered_count = 0
        
        # Update last_axis_value for next iteration
        if filtered_poses is not None and filtered_count > 0:
            if poses.ndim == 2:
                last_axis_value = filtered_poses[axis_idx, 3]
            elif poses.ndim == 3:
                if overlap_strategy == 'last_to_first':
                    # For last_to_first, track the minimum (we're going backwards)
                    if reverse:
                        last_axis_value = filtered_poses[:, axis_idx, 3].max()
                    else:
                        last_axis_value = filtered_poses[:, axis_idx, 3].min()
                else:
                    # For next_to_prior, track based on sort direction
                    if reverse:
                        last_axis_value = filtered_poses[:, axis_idx, 3].min()
                    else:
                        last_axis_value = filtered_poses[:, axis_idx, 3].max()
            elif poses.ndim == 4:
                coords = filtered_poses[:, :, axis_idx, 3].flatten()
                if overlap_strategy == 'last_to_first':
                    # For last_to_first, track the minimum (we're going backwards)
                    if reverse:
                        last_axis_value = coords.max()
                    else:
                        last_axis_value = coords.min()
                else:
                    # For next_to_prior, track based on sort direction
                    if reverse:
                        last_axis_value = coords.min()
                    else:
                        last_axis_value = coords.max()
        
        # Store results
        if filtered_poses is not None and filtered_count > 0:
            trajectory_poses_list.append(filtered_poses)
            volume_pose_counts.append(filtered_count)
            volume_pose_counts_original.append(original_count)
            volume_pose_counts_filtered.append(filtered_count)
            sorted_volume_numbers.append(vol_num)
            volume_start_indices.append(current_index)
            
            # Get axis range for display
            if poses.ndim == 2:
                axis_min = axis_max = filtered_poses[axis_idx, 3]
            elif poses.ndim == 3:
                axis_min = filtered_poses[:, axis_idx, 3].min()
                axis_max = filtered_poses[:, axis_idx, 3].max()
            elif poses.ndim == 4:
                coords = filtered_poses[:, :, axis_idx, 3].flatten()
                axis_min = coords.min()
                axis_max = coords.max()
            
            axis_range_str = f"[{axis_min:.3f}, {axis_max:.3f}]"
            print(f"  Volume {vol_num:3d}: {filtered_count:4d}/{original_count:4d} poses kept, "
                  f"shape {filtered_poses.shape}, "
                  f"center {sort_axis.upper()}={center[axis_idx]:8.3f}, "
                  f"range={axis_range_str}, indices [{current_index}:{current_index + filtered_count})")
            
            current_index += filtered_count
        else:
            print(f"  Volume {vol_num:3d}: {0:4d}/{original_count:4d} poses kept (all filtered - complete overlap)")
    
    if not trajectory_poses_list:
        raise ValueError("No poses remain after filtering overlaps. Check volume ordering or axis selection.")
    
    # For last_to_first, we processed in reverse, so reverse the results back to maintain sort order
    if overlap_strategy == 'last_to_first':
        trajectory_poses_list = list(reversed(trajectory_poses_list))
        volume_pose_counts = list(reversed(volume_pose_counts))
        volume_pose_counts_original = list(reversed(volume_pose_counts_original))
        sorted_volume_numbers = list(reversed(sorted_volume_numbers))
        # Recompute start indices after reversing
        volume_start_indices = []
        current_idx = 0
        for count in volume_pose_counts:
            volume_start_indices.append(current_idx)
            current_idx += count
    
    # Concatenate all filtered poses - handle different shapes
    first_shape = trajectory_poses_list[0].shape
    
    if first_shape[-2:] == (3, 4):
        # Concatenate along first dimension, preserve [N, 3, 4] or [N, D, 3, 4] format
        trajectory_poses = np.concatenate(trajectory_poses_list, axis=0)
    elif first_shape[-2:] == (4, 4):
        # Concatenate along first dimension, preserve [N, 4, 4] or [N, D, 4, 4] format
        trajectory_poses = np.concatenate(trajectory_poses_list, axis=0)
    else:
        raise ValueError(f"Unexpected pose shape: {first_shape}")
    
    # Compute trajectory statistics
    if trajectory_poses.ndim == 2:
        all_positions = trajectory_poses[:3, 3].reshape(1, 3)
    elif trajectory_poses.ndim == 3:
        all_positions = trajectory_poses[:, :3, 3]
    elif trajectory_poses.ndim == 4:
        all_positions = trajectory_poses[:, :, :3, 3].reshape(-1, 3)
    
    trajectory_info = {
        'num_volumes': len(trajectory_poses_list),  # Only volumes that contributed poses
        'total_poses': sum(volume_pose_counts),
        'total_poses_original': sum(volume_pose_counts_original),
        'overlap_removed_count': sum(volume_pose_counts_original) - sum(volume_pose_counts),
        'sort_axis': sort_axis,
        'sort_reverse': reverse,
        'overlap_strategy': overlap_strategy,
        'volume_numbers': sorted_volume_numbers,
        'volume_pose_counts': volume_pose_counts,
        'volume_pose_counts_original': volume_pose_counts_original,
        'volume_start_indices': volume_start_indices,
        'center_positions': center_positions[sort_indices[:len(sorted_volume_numbers)]],
        'output_shape': trajectory_poses.shape,
        'x_range': (all_positions[:, 0].min(), all_positions[:, 0].max()),
        'y_range': (all_positions[:, 1].min(), all_positions[:, 1].max()),
        'z_range': (all_positions[:, 2].min(), all_positions[:, 2].max()),
        'x_span': all_positions[:, 0].max() - all_positions[:, 0].min(),
        'y_span': all_positions[:, 1].max() - all_positions[:, 1].min(),
        'z_span': all_positions[:, 2].max() - all_positions[:, 2].min(),
    }
    
    print(f"\n{'='*70}")
    print(f"Trajectory Summary:")
    print(f"  Volumes contributing: {trajectory_info['num_volumes']}")
    print(f"  Overlap strategy: {overlap_strategy}")
    print(f"  Output shape: {trajectory_info['output_shape']}")
    print(f"  Total poses (after overlap removal): {trajectory_info['total_poses']}")
    print(f"  Total poses (before filtering): {trajectory_info['total_poses_original']}")
    print(f"  Overlapping poses removed: {trajectory_info['overlap_removed_count']}")
    print(f"  X range: [{trajectory_info['x_range'][0]:.3f}, {trajectory_info['x_range'][1]:.3f}] (span: {trajectory_info['x_span']:.3f})")
    print(f"  Y range: [{trajectory_info['y_range'][0]:.3f}, {trajectory_info['y_range'][1]:.3f}] (span: {trajectory_info['y_span']:.3f})")
    print(f"  Z range: [{trajectory_info['z_range'][0]:.3f}, {trajectory_info['z_range'][1]:.3f}] (span: {trajectory_info['z_span']:.3f})")
    print(f"{'='*70}")
    
    return trajectory_poses, trajectory_info


def compute_middle_transformed_poses(trajectory_poses: np.ndarray,
                                    trajectory_info: dict,
                                    num_middle_poses: int = 1,
                                    translation_axis: str = 'x',
                                    translation_scale: float = 10.0,
                                    flip_axes: Tuple[str, str] = ('x', 'y')) -> np.ndarray:
    """
    Find middle pose(s) along trajectory axis, translate along perpendicular axis, and flip two axes.
    
    This creates new poses by:
    1. Finding the middle pose(s) along the sorting axis
    2. Translating them along a perpendicular axis by translation_scale
    3. Rotating 180 degrees by flipping two axes
    
    Args:
        trajectory_poses: Array with trajectory poses
        trajectory_info: Trajectory metadata with sort axis info
        num_middle_poses: Number of poses to extract from middle (default: 1)
        translation_axis: Axis to translate along ('x', 'y', or 'z')
        translation_scale: Translation distance in mm
        flip_axes: Tuple of two axes to flip for 180-degree rotation
        
    Returns:
        transformed_poses: New poses with translation and rotation applied
    """
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    sort_axis = trajectory_info['sort_axis']
    sort_axis_idx = axis_map[sort_axis.lower()]
    trans_axis_idx = axis_map[translation_axis.lower()]
    flip_idx_1 = axis_map[flip_axes[0].lower()]
    flip_idx_2 = axis_map[flip_axes[1].lower()]
    
    # Extract all positions to find middle along sort axis
    if trajectory_poses.ndim == 2:
        positions = trajectory_poses[:3, 3].reshape(1, 3)
        all_poses = trajectory_poses.reshape(1, *trajectory_poses.shape)
    elif trajectory_poses.ndim == 3:
        positions = trajectory_poses[:, :3, 3]
        all_poses = trajectory_poses
    elif trajectory_poses.ndim == 4:
        # Flatten to [N*D, 3, 4]
        N, D = trajectory_poses.shape[:2]
        positions = trajectory_poses[:, :, :3, 3].reshape(-1, 3)
        all_poses = trajectory_poses.reshape(-1, 3, 4)
    else:
        raise ValueError(f"Unexpected pose shape: {trajectory_poses.shape}")
    
    # Find middle pose(s) along the sort axis
    sort_axis_values = positions[:, sort_axis_idx]
    middle_value = (sort_axis_values.min() + sort_axis_values.max()) / 2.0
    
    # Find indices closest to middle
    distances_from_middle = np.abs(sort_axis_values - middle_value)
    middle_indices = np.argsort(distances_from_middle)[:num_middle_poses]
    
    print(f"\n{'='*70}")
    print(f"COMPUTING TRANSFORMED POSES FROM MIDDLE")
    print(f"{'='*70}")
    print(f"Sort axis: {sort_axis.upper()}")
    print(f"Middle value along {sort_axis.upper()}: {middle_value:.3f}")
    print(f"Number of middle poses: {num_middle_poses}")
    print(f"Translation axis: {translation_axis.upper()}")
    print(f"Translation scale: {translation_scale:.3f} mm")
    print(f"Flip axes: {flip_axes[0].upper()} and {flip_axes[1].upper()}")
    print(f"{'='*70}")
    
    # Extract middle poses
    middle_poses = all_poses[middle_indices].copy()
    
    # Apply transformation to each middle pose
    transformed_poses = []
    
    for i, pose_idx in enumerate(middle_indices):
        pose = middle_poses[i].copy()
        
        original_pos = pose[:3, 3].copy()
        
        # Step 1: Translate along translation_axis
        pose[:3, 3][trans_axis_idx] -= translation_scale
        
        # Step 2: Flip two axes (180-degree rotation)
        # This is done by negating the corresponding columns of the rotation matrix
        # and negating the translation components
        pose[:3, flip_idx_1] = -pose[:3, flip_idx_1]
        pose[:3, flip_idx_2] = -pose[:3, flip_idx_2]
        pose[:3, 3][flip_idx_1] = -pose[:3, 3][flip_idx_1]
        pose[:3, 3][flip_idx_2] = -pose[:3, 3][flip_idx_2]
        
        transformed_poses.append(pose)
        
        new_pos = pose[:3, 3]
        print(f"  Pose {i+1}/{num_middle_poses}:")
        print(f"    Original position: [{original_pos[0]:8.3f}, {original_pos[1]:8.3f}, {original_pos[2]:8.3f}]")
        print(f"    New position:      [{new_pos[0]:8.3f}, {new_pos[1]:8.3f}, {new_pos[2]:8.3f}]")
    
    transformed_poses = np.array(transformed_poses)
    
    print(f"{'='*70}")
    print(f"Transformed poses shape: {transformed_poses.shape}")
    print(f"{'='*70}\n")
    
    return transformed_poses


def visualize_trajectory_with_transformed(original_poses: np.ndarray,
                                         transformed_poses: np.ndarray,
                                         trajectory_info: dict,
                                         save_path: Optional[str] = None):
    """
    Visualize original trajectory and transformed poses in 3D with orientation arrows.
    Shows both sets of poses with direction vectors.
    
    Args:
        original_poses: Original trajectory poses
        transformed_poses: Transformed poses
        trajectory_info: Trajectory metadata
        save_path: Optional path to save the figure
    """
    # Extract positions
    if original_poses.ndim == 2:
        orig_positions = original_poses[:3, 3].reshape(1, 3)
        orig_poses_arr = original_poses.reshape(1, *original_poses.shape)
    elif original_poses.ndim == 3:
        orig_positions = original_poses[:, :3, 3]
        orig_poses_arr = original_poses
    elif original_poses.ndim == 4:
        orig_positions = original_poses[:, :, :3, 3].reshape(-1, 3)
        orig_poses_arr = original_poses.reshape(-1, 3, 4)
    
    if transformed_poses.ndim == 2:
        trans_positions = transformed_poses[:3, 3].reshape(1, 3)
        trans_poses_arr = transformed_poses.reshape(1, *transformed_poses.shape)
    elif transformed_poses.ndim == 3:
        trans_positions = transformed_poses[:, :3, 3]
        trans_poses_arr = transformed_poses
    
    fig = plt.figure(figsize=(20, 10))
    
    # 3D plot with orientation
    ax1 = fig.add_subplot(231, projection='3d')
    
    # Plot original trajectory
    ax1.scatter(orig_positions[:, 0], orig_positions[:, 1], orig_positions[:, 2],
               c='blue', s=10, alpha=0.4, label='Original trajectory')
    ax1.plot(orig_positions[:, 0], orig_positions[:, 1], orig_positions[:, 2],
            c='blue', alpha=0.2, linewidth=0.5)
    
    # Plot transformed poses
    ax1.scatter(trans_positions[:, 0], trans_positions[:, 1], trans_positions[:, 2],
               c='red', s=100, marker='*', edgecolors='black', linewidth=2,
               label='Transformed poses')
    
    # Draw orientation arrows for transformed poses
    arrow_length = 5.0
    for i in range(len(trans_poses_arr)):
        pos = trans_poses_arr[i, :3, 3]
        # Z-axis direction (forward)
        z_dir = trans_poses_arr[i, :3, 2]
        ax1.quiver(pos[0], pos[1], pos[2],
                  z_dir[0], z_dir[1], z_dir[2],
                  length=arrow_length, color='red', arrow_length_ratio=0.3, linewidth=2)
    
    ax1.set_xlabel('X', fontweight='bold')
    ax1.set_ylabel('Y', fontweight='bold')
    ax1.set_zlabel('Z', fontweight='bold')
    ax1.set_title('3D View with Orientations', fontweight='bold', fontsize=14)
    ax1.legend()
    
    # XY projection
    ax2 = fig.add_subplot(232)
    ax2.scatter(orig_positions[:, 0], orig_positions[:, 1],
               c='blue', s=10, alpha=0.4, label='Original')
    ax2.plot(orig_positions[:, 0], orig_positions[:, 1],
            c='blue', alpha=0.2, linewidth=0.5)
    ax2.scatter(trans_positions[:, 0], trans_positions[:, 1],
               c='red', s=100, marker='*', edgecolors='black', linewidth=2,
               label='Transformed')
    
    # Draw orientation arrows in XY
    for i in range(len(trans_poses_arr)):
        pos = trans_poses_arr[i, :3, 3]
        z_dir = trans_poses_arr[i, :3, 2]
        ax2.arrow(pos[0], pos[1], z_dir[0]*arrow_length, z_dir[1]*arrow_length,
                 head_width=1.0, head_length=1.5, fc='red', ec='red', linewidth=2)
    
    ax2.set_xlabel('X', fontweight='bold')
    ax2.set_ylabel('Y', fontweight='bold')
    ax2.set_title('XY Projection (Top View)', fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    ax2.legend()
    
    # XZ projection
    ax3 = fig.add_subplot(233)
    ax3.scatter(orig_positions[:, 0], orig_positions[:, 2],
               c='blue', s=10, alpha=0.4, label='Original')
    ax3.plot(orig_positions[:, 0], orig_positions[:, 2],
            c='blue', alpha=0.2, linewidth=0.5)
    ax3.scatter(trans_positions[:, 0], trans_positions[:, 2],
               c='red', s=100, marker='*', edgecolors='black', linewidth=2,
               label='Transformed')
    
    # Draw orientation arrows in XZ
    for i in range(len(trans_poses_arr)):
        pos = trans_poses_arr[i, :3, 3]
        z_dir = trans_poses_arr[i, :3, 2]
        ax3.arrow(pos[0], pos[2], z_dir[0]*arrow_length, z_dir[2]*arrow_length,
                 head_width=1.0, head_length=1.5, fc='red', ec='red', linewidth=2)
    
    ax3.set_xlabel('X', fontweight='bold')
    ax3.set_ylabel('Z', fontweight='bold')
    ax3.set_title('XZ Projection (Front View)', fontweight='bold', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')
    ax3.legend()
    
    # YZ projection
    ax4 = fig.add_subplot(234)
    ax4.scatter(orig_positions[:, 1], orig_positions[:, 2],
               c='blue', s=10, alpha=0.4, label='Original')
    ax4.plot(orig_positions[:, 1], orig_positions[:, 2],
            c='blue', alpha=0.2, linewidth=0.5)
    ax4.scatter(trans_positions[:, 1], trans_positions[:, 2],
               c='red', s=100, marker='*', edgecolors='black', linewidth=2,
               label='Transformed')
    
    # Draw orientation arrows in YZ
    for i in range(len(trans_poses_arr)):
        pos = trans_poses_arr[i, :3, 3]
        z_dir = trans_poses_arr[i, :3, 2]
        ax4.arrow(pos[1], pos[2], z_dir[1]*arrow_length, z_dir[2]*arrow_length,
                 head_width=1.0, head_length=1.5, fc='red', ec='red', linewidth=2)
    
    ax4.set_xlabel('Y', fontweight='bold')
    ax4.set_ylabel('Z', fontweight='bold')
    ax4.set_title('YZ Projection (Side View)', fontweight='bold', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal', adjustable='box')
    ax4.legend()
    
    # Add zoomed view around transformed poses
    ax5 = fig.add_subplot(235, projection='3d')
    
    # Get bounds around transformed poses
    margin = 20.0
    x_min, x_max = trans_positions[:, 0].min() - margin, trans_positions[:, 0].max() + margin
    y_min, y_max = trans_positions[:, 1].min() - margin, trans_positions[:, 1].max() + margin
    z_min, z_max = trans_positions[:, 2].min() - margin, trans_positions[:, 2].max() + margin
    
    # Filter original poses within bounds
    mask = ((orig_positions[:, 0] >= x_min) & (orig_positions[:, 0] <= x_max) &
            (orig_positions[:, 1] >= y_min) & (orig_positions[:, 1] <= y_max) &
            (orig_positions[:, 2] >= z_min) & (orig_positions[:, 2] <= z_max))
    
    nearby_positions = orig_positions[mask]
    
    if len(nearby_positions) > 0:
        ax5.scatter(nearby_positions[:, 0], nearby_positions[:, 1], nearby_positions[:, 2],
                   c='blue', s=20, alpha=0.6, label='Original (nearby)')
        ax5.plot(nearby_positions[:, 0], nearby_positions[:, 1], nearby_positions[:, 2],
                c='blue', alpha=0.3, linewidth=1)
    
    ax5.scatter(trans_positions[:, 0], trans_positions[:, 1], trans_positions[:, 2],
               c='red', s=150, marker='*', edgecolors='black', linewidth=2,
               label='Transformed')
    
    # Draw orientation arrows
    for i in range(len(trans_poses_arr)):
        pos = trans_poses_arr[i, :3, 3]
        z_dir = trans_poses_arr[i, :3, 2]
        ax5.quiver(pos[0], pos[1], pos[2],
                  z_dir[0], z_dir[1], z_dir[2],
                  length=arrow_length, color='red', arrow_length_ratio=0.3, linewidth=2)
    
    ax5.set_xlabel('X', fontweight='bold')
    ax5.set_ylabel('Y', fontweight='bold')
    ax5.set_zlabel('Z', fontweight='bold')
    ax5.set_title('Zoomed View Around Transformed Poses', fontweight='bold', fontsize=14)
    ax5.legend()
    
    # Add text summary
    ax6 = fig.add_subplot(236)
    ax6.axis('off')
    
    summary_text = f"""
TRANSFORMATION SUMMARY

Original trajectory:
  Total poses: {len(orig_positions)}
  X range: [{orig_positions[:, 0].min():.2f}, {orig_positions[:, 0].max():.2f}]
  Y range: [{orig_positions[:, 1].min():.2f}, {orig_positions[:, 1].max():.2f}]
  Z range: [{orig_positions[:, 2].min():.2f}, {orig_positions[:, 2].max():.2f}]

Transformed poses:
  Count: {len(trans_positions)}
  X range: [{trans_positions[:, 0].min():.2f}, {trans_positions[:, 0].max():.2f}]
  Y range: [{trans_positions[:, 1].min():.2f}, {trans_positions[:, 1].max():.2f}]
  Z range: [{trans_positions[:, 2].min():.2f}, {trans_positions[:, 2].max():.2f}]

Sort axis: {trajectory_info['sort_axis'].upper()}
"""
    
    ax6.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\nSaved transformed trajectory visualization to: {save_path}")
    
    plt.show()


def visualize_trajectory(trajectory_poses: np.ndarray,
                         trajectory_info: dict,
                         save_path: Optional[str] = None):
    """
    Visualize the trajectory in 3D projections.
    Shows volume boundaries with different colors.
    
    Args:
        trajectory_poses: Array with trajectory poses (various shapes supported)
        trajectory_info: Trajectory metadata
        save_path: Optional path to save the figure
    """
    # Extract positions based on shape
    if trajectory_poses.ndim == 2:
        positions = trajectory_poses[:3, 3].reshape(1, 3)
    elif trajectory_poses.ndim == 3:
        positions = trajectory_poses[:, :3, 3]
    elif trajectory_poses.ndim == 4:
        positions = trajectory_poses[:, :, :3, 3].reshape(-1, 3)
    else:
        raise ValueError(f"Unexpected pose shape: {trajectory_poses.shape}")
    
    volume_numbers = trajectory_info['volume_numbers']
    volume_start_indices = trajectory_info['volume_start_indices']
    center_positions = trajectory_info['center_positions']
    
    fig = plt.figure(figsize=(18, 5))
    
    # Create colors for each volume
    n_volumes = len(volume_numbers)
    colors = plt.cm.tab20(np.linspace(0, 1, n_volumes))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Plot each volume with a different color
    for i, (vol_num, start_idx) in enumerate(zip(volume_numbers, volume_start_indices)):
        if i < len(volume_start_indices) - 1:
            end_idx = volume_start_indices[i + 1]
        else:
            end_idx = len(positions)
        
        vol_positions = positions[start_idx:end_idx]
        
        ax1.scatter(vol_positions[:, 0], vol_positions[:, 1], vol_positions[:, 2],
                   c=[colors[i]], s=20, alpha=0.6, label=f'Vol {vol_num}')
        
        # Draw line through volume
        ax1.plot(vol_positions[:, 0], vol_positions[:, 1], vol_positions[:, 2],
                c=colors[i], alpha=0.3, linewidth=1)
        
        # Mark center
        center = center_positions[i]
        ax1.scatter([center[0]], [center[1]], [center[2]],
                   c=[colors[i]], s=200, marker='*', edgecolors='black', linewidth=2)
    
    ax1.set_xlabel('X', fontweight='bold')
    ax1.set_ylabel('Y', fontweight='bold')
    ax1.set_zlabel('Z', fontweight='bold')
    ax1.set_title('3D Trajectory (All Poses)', fontweight='bold', fontsize=14)
    if n_volumes <= 10:
        ax1.legend(fontsize=8, loc='upper left')
    
    # XY projection
    ax2 = fig.add_subplot(132)
    for i, (vol_num, start_idx) in enumerate(zip(volume_numbers, volume_start_indices)):
        if i < len(volume_start_indices) - 1:
            end_idx = volume_start_indices[i + 1]
        else:
            end_idx = len(positions)
        
        vol_positions = positions[start_idx:end_idx]
        
        ax2.scatter(vol_positions[:, 0], vol_positions[:, 1],
                   c=[colors[i]], s=20, alpha=0.6, label=f'Vol {vol_num}')
        ax2.plot(vol_positions[:, 0], vol_positions[:, 1],
                c=colors[i], alpha=0.3, linewidth=1)
        
        # Mark center
        center = center_positions[i]
        ax2.scatter([center[0]], [center[1]],
                   c=[colors[i]], s=200, marker='*', edgecolors='black', linewidth=2)
    
    ax2.set_xlabel('X', fontweight='bold')
    ax2.set_ylabel('Y', fontweight='bold')
    ax2.set_title('XY Projection (Top View)', fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    # XZ projection (or YZ depending on sort axis)
    ax3 = fig.add_subplot(133)
    sort_axis = trajectory_info['sort_axis']
    
    for i, (vol_num, start_idx) in enumerate(zip(volume_numbers, volume_start_indices)):
        if i < len(volume_start_indices) - 1:
            end_idx = volume_start_indices[i + 1]
        else:
            end_idx = len(positions)
        
        vol_positions = positions[start_idx:end_idx]
        
        if sort_axis == 'z':
            ax3.scatter(vol_positions[:, 0], vol_positions[:, 2],
                       c=[colors[i]], s=20, alpha=0.6, label=f'Vol {vol_num}')
            ax3.plot(vol_positions[:, 0], vol_positions[:, 2],
                    c=colors[i], alpha=0.3, linewidth=1)
            center = center_positions[i]
            ax3.scatter([center[0]], [center[2]],
                       c=[colors[i]], s=200, marker='*', edgecolors='black', linewidth=2)
            ax3.set_xlabel('X', fontweight='bold')
            ax3.set_ylabel('Z', fontweight='bold')
            ax3.set_title('XZ Projection (Front View)', fontweight='bold', fontsize=14)
        elif sort_axis == 'y':
            ax3.scatter(vol_positions[:, 0], vol_positions[:, 1],
                       c=[colors[i]], s=20, alpha=0.6, label=f'Vol {vol_num}')
            ax3.plot(vol_positions[:, 0], vol_positions[:, 1],
                    c=colors[i], alpha=0.3, linewidth=1)
            center = center_positions[i]
            ax3.scatter([center[0]], [center[1]],
                       c=[colors[i]], s=200, marker='*', edgecolors='black', linewidth=2)
            ax3.set_xlabel('X', fontweight='bold')
            ax3.set_ylabel('Y', fontweight='bold')
            ax3.set_title('XY Projection', fontweight='bold', fontsize=14)
        else:  # x
            ax3.scatter(vol_positions[:, 1], vol_positions[:, 2],
                       c=[colors[i]], s=20, alpha=0.6, label=f'Vol {vol_num}')
            ax3.plot(vol_positions[:, 1], vol_positions[:, 2],
                    c=colors[i], alpha=0.3, linewidth=1)
            center = center_positions[i]
            ax3.scatter([center[1]], [center[2]],
                       c=[colors[i]], s=200, marker='*', edgecolors='black', linewidth=2)
            ax3.set_xlabel('Y', fontweight='bold')
            ax3.set_ylabel('Z', fontweight='bold')
            ax3.set_title('YZ Projection (Side View)', fontweight='bold', fontsize=14)
    
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\nSaved trajectory visualization to: {save_path}")
    
    plt.show()


def main():
    """
    Main function: Load poses from volume folders and compute trajectory.
    """
    parser = argparse.ArgumentParser(
        description='Compute trajectory poses from multiple volume folders by concatenating all poses in sorted order.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract non-overlapping trajectory sorted by Z-axis (default: next_to_prior)
  python compute_trajectory_poses.py --data_folder ./data --axis z
  
  # Use last_to_first strategy to maximize poses from last volume in sorted order
  python compute_trajectory_poses.py --data_folder ./data --axis z --overlap_strategy last_to_first
  
  # Use bidirectional strategy to balance poses from both ends
  python compute_trajectory_poses.py --data_folder ./data --axis z --overlap_strategy bidirectional
  
  # Extract trajectory from specific volumes, sorted by X-axis
  python compute_trajectory_poses.py --data_folder ./data --axis x --volumes 0,5,10,15,20
  
  # Sort in descending order with next_to_prior strategy
  python compute_trajectory_poses.py --data_folder ./data --axis z --reverse
  
  # Create transformed poses from middle (translate + 180° rotation)
  python compute_trajectory_poses.py --data_folder ./data --axis z --transform_middle \
      --num_middle_poses 5 --translation_axis x --translation_scale 15.0 --flip_axes x,y

How it works:
  1. Loads poses.npy from each volumeX folder
  2. Computes center position for each volume
  3. Sorts volumes along the specified axis (x, y, or z)
  4. Removes overlaps using the specified strategy:
     - next_to_prior: Keep only poses beyond previous volume (minimal overlap, first→last)
     - last_to_first: Keep all from last volume, extend backward (larger→smaller values)
     - bidirectional: Combine both - first half uses next_to_prior, second half uses last_to_first
  5. Concatenates non-overlapping poses into trajectory_poses.npy
  
Overlap strategies:
  - next_to_prior (default): Each volume contributes poses adjacent to the prior volume.
    Results in even distribution of poses across volumes. Processes from first to last.
  
  - last_to_first: Processes volumes from last to first in sorted order, keeping poses 
    from larger to smaller axis values. Prioritizes maximum poses from the last volume 
    in the sorted order. Results in more poses from the end of the trajectory.
  
  - bidirectional: Combines both strategies, splitting at the middle volume.
    First half uses next_to_prior (forward processing), second half uses last_to_first
    (backward processing). Balances pose distribution from both trajectory ends toward
    the center. Best when both ends of trajectory are equally important.

Transformation feature:
  The --transform_middle option finds the middle pose(s) along the trajectory axis and:
  1. Translates them along a perpendicular axis (specified by --translation_axis)
  2. Applies a 180-degree rotation by flipping two axes (specified by --flip_axes)
  
  This creates new camera poses that view the anatomy from a different perspective,
  useful for testing reconstruction quality from alternative viewpoints.
        """
    )
    
    parser.add_argument('--data_folder', type=str, required=True,
                       help='Path to the data folder containing volume subfolders')
    parser.add_argument('--axis', type=str, default='z', choices=['x', 'y', 'z'],
                       help='Axis to sort trajectory along (default: z)')
    parser.add_argument('--reverse', action='store_true',
                       help='Sort in descending order (default: ascending)')
    parser.add_argument('--overlap_strategy', type=str, default='next_to_prior',
                       choices=['next_to_prior', 'last_to_first', 'bidirectional'],
                       help='Strategy for removing overlaps: '
                            'next_to_prior (process first→last, keep poses beyond prior), '
                            'last_to_first (process last→first, keep poses from larger to smaller), '
                            'bidirectional (combine both, split at middle) (default: next_to_prior)')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Output path for trajectory poses (default: <data_folder>/trajectory_poses.npy)')
    parser.add_argument('--volumes', type=str, default=None,
                       help='Comma-separated list of volume numbers to include (e.g., "0,5,10,15")')
    parser.add_argument('--volume_prefix', type=str, default='volume',
                       help='Prefix for volume directories (default: "volume")')
    parser.add_argument('--no_visualize', action='store_true',
                       help='Skip visualization')
    
    # Transformation options
    parser.add_argument('--transform_middle', action='store_true',
                       help='Compute transformed poses from middle of trajectory')
    parser.add_argument('--num_middle_poses', type=int, default=1,
                       help='Number of middle poses to transform (default: 1)')
    parser.add_argument('--translation_axis', type=str, default='x', choices=['x', 'y', 'z'],
                       help='Axis to translate along (default: x)')
    parser.add_argument('--translation_scale', type=float, default=10.0,
                       help='Translation distance in mm (default: 10.0)')
    parser.add_argument('--flip_axes', type=str, default='x,y',
                       help='Comma-separated pair of axes to flip for 180-degree rotation (default: "x,y")')
    
    args = parser.parse_args()
    
    # Parse volume filter
    volume_filter = None
    if args.volumes and args.volumes.strip() and args.volumes.strip().lower() != 'all':
        try:
            if '-' in args.volumes:
                start, end = map(int, args.volumes.split('-'))
                volume_filter = set(range(start, end + 1, 10))
            else:
                volume_filter = set([int(v.strip()) for v in args.volumes.split(',')])
            print(f"\nFiltering to volumes: {sorted(volume_filter)}")
        except ValueError:
            print(f"WARNING: Invalid volume list '{args.volumes}', using all volumes")
    
    # Check if path exists
    data_folder = args.data_folder
    if not os.path.exists(data_folder):
        data_folder = os.path.abspath(data_folder)
    
    if not os.path.exists(data_folder):
        print(f"ERROR: Data folder not found: {args.data_folder}")
        return
    
    print(f"\n{'='*70}")
    print(f"COMPUTING TRAJECTORY POSES")
    print(f"{'='*70}")
    print(f"Data folder: {data_folder}")
    print(f"Sort axis: {args.axis.upper()}")
    print(f"Sort order: {'Descending' if args.reverse else 'Ascending'}")
    print(f"Overlap strategy: {args.overlap_strategy}")
    print(f"{'='*70}")
    
    # Load poses from all volumes
    poses_list, volume_names, volume_numbers = load_poses_from_volumes(
        data_folder,
        volume_filter=volume_filter,
        volume_prefix=args.volume_prefix
    )
    
    # Compute center positions for sorting
    center_positions, volume_numbers = compute_volume_center_positions(
        poses_list,
        volume_numbers
    )
    
    # Concatenate poses along trajectory
    trajectory_poses, trajectory_info = concatenate_poses_along_trajectory(
        poses_list,
        volume_numbers,
        center_positions,
        sort_axis=args.axis,
        reverse=args.reverse,
        overlap_strategy=args.overlap_strategy
    )
    
    # Determine output path
    if args.output_path is None:
        output_path = os.path.join(data_folder, 'trajectory_poses.npy')
    else:
        output_path = args.output_path
    
    # Save trajectory poses
    # Keep original shape from concatenation
    np.save(output_path, trajectory_poses)
    
    print(f"\n{'='*70}")
    print(f"TRAJECTORY SAVED")
    print(f"{'='*70}")
    print(f"Output path: {output_path}")
    print(f"Shape: {trajectory_poses.shape}")
    print(f"Total poses: {trajectory_info['total_poses']}")
    print(f"Number of volumes: {trajectory_info['num_volumes']}")
    print(f"{'='*70}\n")
    
    # Save metadata
    metadata_path = output_path.replace('.npy', '_metadata.npz')
    np.savez(metadata_path,
             volume_numbers=trajectory_info['volume_numbers'],
             volume_pose_counts=trajectory_info['volume_pose_counts'],
             volume_pose_counts_original=trajectory_info['volume_pose_counts_original'],
             volume_start_indices=trajectory_info['volume_start_indices'],
             center_positions=trajectory_info['center_positions'],
             sort_axis=args.axis,
             sort_reverse=args.reverse,
             overlap_strategy=args.overlap_strategy,
             total_poses_original=trajectory_info['total_poses_original'],
             overlap_removed_count=trajectory_info['overlap_removed_count'],
             x_range=trajectory_info['x_range'],
             y_range=trajectory_info['y_range'],
             z_range=trajectory_info['z_range'])
    print(f"Metadata saved to: {metadata_path}")
    
    # Visualize trajectory
    if not args.no_visualize:
        vis_path = output_path.replace('.npy', '_visualization.png')
        visualize_trajectory(trajectory_poses, trajectory_info, save_path=vis_path)
    
    # Optional: Compute transformed poses from middle
    if args.transform_middle:
        # Parse flip_axes
        try:
            flip_axes_list = [ax.strip().lower() for ax in args.flip_axes.split(',')]
            if len(flip_axes_list) != 2 or any(ax not in ['x', 'y', 'z'] for ax in flip_axes_list):
                raise ValueError()
            flip_axes = tuple(flip_axes_list)
        except:
            print(f"WARNING: Invalid flip_axes '{args.flip_axes}', using default 'x,y'")
            flip_axes = ('x', 'y')
        
        # Compute transformed poses
        transformed_poses = compute_middle_transformed_poses(
            trajectory_poses,
            trajectory_info,
            num_middle_poses=args.num_middle_poses,
            translation_axis=args.translation_axis,
            translation_scale=args.translation_scale,
            flip_axes=flip_axes
        )
        
        # Save transformed poses
        transformed_output_path = output_path.replace('.npy', '_transformed.npy')
        np.save(transformed_output_path, transformed_poses)
        
        print(f"\n{'='*70}")
        print(f"TRANSFORMED POSES SAVED")
        print(f"{'='*70}")
        print(f"Output path: {transformed_output_path}")
        print(f"Shape: {transformed_poses.shape}")
        print(f"Number of poses: {len(transformed_poses)}")
        print(f"{'='*70}\n")
        
        # Visualize both original and transformed
        if not args.no_visualize:
            trans_vis_path = output_path.replace('.npy', '_transformed_visualization.png')
            visualize_trajectory_with_transformed(
                trajectory_poses,
                transformed_poses,
                trajectory_info,
                save_path=trans_vis_path
            )
    
    return trajectory_poses, trajectory_info


if __name__ == "__main__":
    np.random.seed(42)
    trajectory_poses, info = main()
