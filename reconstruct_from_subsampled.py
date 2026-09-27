"""
3D Interpolation Reconstruction from Subsampled UltraNeRF Outputs
=================================================================

This script reconstructs full-resolution 3D volumes from subsampled NeRF renderings
using 3D interpolation (linear, cubic, or nearest neighbor).

Usage:
    python reconstruct_from_subsampled.py \
        --eval_dir logs/experiment/eval_results \
        --method linear \
        --output_dir logs/experiment/reconstructed
"""

import os
import argparse
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from tqdm import tqdm


def reconstruct_volume_3d(subsampled_volume, kept_indices, target_depth, method='linear'):
    """
    Reconstruct full-resolution volume from subsampled slices using 3D interpolation.
    
    Args:
        subsampled_volume: Array of shape [D_sub, H, W] - subsampled volume
        kept_indices: Indices of kept slices in original volume
        target_depth: Target depth for full resolution
        method: Interpolation method ('linear', 'cubic', 'nearest')
    
    Returns:
        reconstructed_volume: Array of shape [target_depth, H, W]
    """
    print(f"Starting 3D reconstruction with method: {method}")
    
    D_sub, H, W = subsampled_volume.shape
    
    # Create coordinate grids
    # Source points (subsampled)
    z_sub = kept_indices
    y_sub = np.arange(H)
    x_sub = np.arange(W)
    
    # Create meshgrid for source points
    Z_sub, Y_sub, X_sub = np.meshgrid(z_sub, y_sub, x_sub, indexing='ij')
    
    # Flatten source coordinates and values
    source_points = np.column_stack([
        Z_sub.ravel(),
        Y_sub.ravel(),
        X_sub.ravel()
    ])
    source_values = subsampled_volume.ravel()
    
    # Target points (full resolution)
    z_full = np.arange(target_depth)
    y_full = np.arange(H)
    x_full = np.arange(W)
    
    Z_full, Y_full, X_full = np.meshgrid(z_full, y_full, x_full, indexing='ij')
    target_points = np.column_stack([
        Z_full.ravel(),
        Y_full.ravel(),
        X_full.ravel()
    ])
    
    print(f"Interpolating from {D_sub} slices to {target_depth} slices...")
    print(f"Source points: {source_points.shape}, Target points: {target_points.shape}")
    print(f"Method: {method}")
    
    # Perform 3D interpolation
    reconstructed_values = griddata(
        source_points,
        source_values,
        target_points,
        method=method,
        fill_value=0.0  # Fill any extrapolated points with 0
    )
    
    # Reshape to volume
    reconstructed_volume = reconstructed_values.reshape(target_depth, H, W)
    
    return reconstructed_volume


def reconstruct_volume_slice_by_slice(subsampled_volume, kept_indices, target_depth, method='linear'):
    """
    Alternative: Reconstruct by interpolating slice-by-slice in the depth dimension.
    This is faster but may be less accurate than full 3D interpolation.
    
    Args:
        subsampled_volume: Array of shape [D_sub, H, W]
        kept_indices: Indices of kept slices
        target_depth: Target depth
        method: Interpolation method
    
    Returns:
        reconstructed_volume: Array of shape [target_depth, H, W]
    """
    D_sub, H, W = subsampled_volume.shape
    reconstructed_volume = np.zeros((target_depth, H, W), dtype=subsampled_volume.dtype)
    
    print(f"Slice-by-slice interpolation from {D_sub} to {target_depth} slices...")
    
    # For each pixel, interpolate across depth
    for i in tqdm(range(H), desc="Height"):
        for j in range(W):
            # Get pixel values at subsampled depths
            pixel_values = subsampled_volume[:, i, j]
            
            # Interpolate to full depth
            reconstructed_volume[:, i, j] = np.interp(
                np.arange(target_depth),
                kept_indices,
                pixel_values
            )
    
    return reconstructed_volume


def visualize_reconstruction(original_sub, reconstructed, kept_indices, save_path):
    """Visualize reconstruction quality."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    D_full, H, W = reconstructed.shape
    
    # Show 3 slices: original positions
    for idx, slice_idx in enumerate([0, len(kept_indices)//2, len(kept_indices)-1]):
        original_depth = kept_indices[slice_idx]
        
        # Original subsampled
        axes[0, idx].imshow(original_sub[slice_idx], cmap='gray')
        axes[0, idx].set_title(f'Original (Depth {original_depth})')
        axes[0, idx].axis('off')
        
        # Reconstructed at same position
        axes[1, idx].imshow(reconstructed[original_depth], cmap='gray')
        axes[1, idx].set_title(f'Reconstructed (Depth {original_depth})')
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to: {save_path}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
    # Original subsampled
    axes[0, 0].imshow(original_sub[:, original_sub.shape[1]//2, :], cmap='gray')
    axes[0, 0].set_title(f'Original Coronal Middle Slice ')
    axes[0, 0].axis('off')
    
    # Reconstructed at same position
    axes[1, 0].imshow(reconstructed[:, reconstructed.shape[1]//2, :], cmap='gray')
    axes[1, 0].set_title(f'Reconstructed Coronal Middle Slice')
    axes[1, 0].axis('off')
    
    # Original subsampled
    axes[0, 1].imshow(original_sub[:, :, original_sub.shape[2]//2], cmap='gray')
    axes[0, 1].set_title(f'Original Axial Middle Slice ')
    axes[0, 1].axis('off')
    
    # Reconstructed at same position
    axes[1, 1].imshow(reconstructed[:, :, reconstructed.shape[2]//2], cmap='gray')
    axes[1, 1].set_title(f'Reconstructed Axial Middle Slice')
    axes[1, 1].axis('off')
    
    save_path = save_path.replace(".png", "_coronal_axial.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Reconstruct full volumes from subsampled NeRF outputs")
    parser.add_argument("--eval_dir", type=str, required=True,
                       help="Directory containing evaluation results with subsampling_info.npz")
    parser.add_argument("--method", type=str, default='linear',
                       choices=['linear', 'cubic', 'nearest'],
                       help="Interpolation method")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for reconstructed volumes (default: eval_dir/reconstructed)")
    parser.add_argument("--use_slice_by_slice", action='store_true',
                       help="Use faster slice-by-slice interpolation instead of full 3D")
    parser.add_argument("--visualize", action='store_true', default=True,
                       help="Create visualization comparing original and reconstructed")
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.eval_dir, "reconstructed")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load subsampling info
    subsample_info_path = os.path.join(args.eval_dir, "subsampling_info.npz")
    if not os.path.exists(subsample_info_path):
        raise FileNotFoundError(f"Subsampling info not found: {subsample_info_path}")
    
    print("\n" + "="*60)
    print("3D VOLUME RECONSTRUCTION FROM SUBSAMPLED DATA")
    print("="*60)
    
    subsample_info = np.load(subsample_info_path)
    kept_indices = subsample_info['kept_indices']
    target_depth = int(subsample_info['original_depth'])
    subsample_every_n = int(subsample_info['subsample_every_n'])
    
    print(f"\nSubsampling Info:")
    print(f"  Original depth: {target_depth}")
    print(f"  Subsampled depth: {len(kept_indices)}")
    print(f"  Subsample every: {subsample_every_n}")
    print(f"  Kept indices: {kept_indices}")
    print(f"  Compression ratio: {len(kept_indices)/target_depth:.2%}")
    print(f"\nInterpolation method: {args.method}")
    print("="*60 + "\n")
    
    # Find all volume files
    renderings_dir = os.path.join(args.eval_dir, "renderings")
    volume_files = sorted([f for f in os.listdir(renderings_dir) if f.startswith("volume_") and f.endswith(".npy")])
    
    if len(volume_files) == 0:
        raise FileNotFoundError(f"No volume files found in {renderings_dir}")
    
    print(f"Found {len(volume_files)} volumes to reconstruct\n")
    
    # Reconstruct each volume
    for vol_file in tqdm(volume_files, desc="Reconstructing volumes"):
        vol_path = os.path.join(renderings_dir, vol_file)
        
        # Load subsampled volume
        subsampled_volume = np.load(vol_path)
        print(f"\nProcessing {vol_file}: shape {subsampled_volume.shape}")
        
        # Validate depth matches
        if subsampled_volume.shape[0] != len(kept_indices):
            print(f"WARNING: Volume depth {subsampled_volume.shape[0]} doesn't match kept_indices {len(kept_indices)}")
            print(f"Skipping {vol_file}")
            continue
        
        # Reconstruct
        if args.use_slice_by_slice:
            reconstructed = reconstruct_volume_slice_by_slice(
                subsampled_volume, kept_indices, target_depth, method=args.method
            )
        else:
            reconstructed = reconstruct_volume_3d(
                subsampled_volume, kept_indices, target_depth, method=args.method
            )
        
        # Save reconstructed volume
        output_path = os.path.join(args.output_dir, vol_file.replace("volume_", "reconstructed_"))
        np.save(output_path, reconstructed)
        print(f"Saved reconstructed volume: {reconstructed.shape} -> {output_path}")
        
        # Visualize if requested
        if args.visualize:
            viz_path = os.path.join(args.output_dir, vol_file.replace("volume_", "viz_").replace(".npy", ".png"))
            visualize_reconstruction(subsampled_volume, reconstructed, kept_indices, viz_path)
    
    print("\n" + "="*60)
    print("RECONSTRUCTION COMPLETE")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Reconstructed {len(volume_files)} volumes")
    print(f"Original depth: {target_depth}, Interpolation: {args.method}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
