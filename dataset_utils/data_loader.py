"""
Flexible data loading system for ICE-NeRF with support for:
- Legacy single-file loading (images.npy, poses.npy in datadir) - Improved backward compatibility to previous versions
- Multi-volume loading (volume1/, volume2/, etc. subdirectories)
- Explicit train/val/test splits via config or auto-detection
- Metadata tracking for logging
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union


class VolumeDataset:
    """Represents a single volume with images and poses."""
    
    def __init__(self, volume_id: str, volume_path: str, images: np.ndarray, poses: np.ndarray):
        self.volume_id = volume_id
        self.volume_path = volume_path
        self.images = images
        self.poses = poses
        self.n_frames = images.shape[0]
        
    def __repr__(self):
        return f"VolumeDataset(id={self.volume_id}, frames={self.n_frames}, shape={self.images.shape})"


class DatasetSplit:
    """Holds train/val/test split with metadata."""
    
    def __init__(self, 
                 images: np.ndarray, 
                 poses: np.ndarray,
                 split_indices: Dict[str, List[int]],
                 volume_metadata: Optional[List[Dict]] = None,
                 full_volume_mode: bool = False):
        """
        Args:
            images: Combined images array
            poses: Combined poses array
            split_indices: Dict with 'train', 'val', 'test' keys
            volume_metadata: List of dicts with volume info for logging
            full_volume_mode: Whether we're in 3D volume mode
        """
        self.images = images
        self.poses = poses
        self.i_train = split_indices.get('train', [])
        self.i_val = split_indices.get('val', [])
        self.i_test = split_indices.get('test', [])
        self.volume_metadata = volume_metadata or []
        self.full_volume_mode = full_volume_mode
        
        # Create index to volume mapping for logging
        self.index_to_volume = {}
        if volume_metadata:
            current_idx = 0
            for vol_meta in volume_metadata:
                n_frames = vol_meta['n_frames']
                for i in range(n_frames):
                    self.index_to_volume[current_idx] = {
                        'volume_id': vol_meta['volume_id'],
                        'volume_path': vol_meta['volume_path'],
                        'frame_in_volume': i if not full_volume_mode else None,
                        'total_frames': n_frames
                    }
                    current_idx += 1
    
    def get_split_info(self) -> Dict:
        """Returns human-readable split information."""
        info = {
            'total_frames': len(self.images),
            'n_train': len(self.i_train),
            'n_val': len(self.i_val),
            'n_test': len(self.i_test),
            'train_indices': self.i_train,
            'val_indices': self.i_val,
            'test_indices': self.i_test,
            'n_volumes': len(self.volume_metadata),
            'full_volume_mode': self.full_volume_mode
        }
        return info
    
    def get_frame_info(self, index: int) -> Dict:
        """Get metadata for a specific frame index."""
        if index in self.index_to_volume:
            return self.index_to_volume[index]
        return {'volume_id': 'unknown', 'frame_in_volume': index}
    
    def save_split_info(self, save_path: str):
        """Save split information to JSON for reproducibility."""
        info = self.get_split_info()
        info['volume_metadata'] = self.volume_metadata
        
        with open(save_path, 'w') as f:
            json.dump(info, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        
        print(f"Saved split info to: {save_path}")


def discover_volumes(datadir: str, volume_prefix: str = "volume") -> Tuple[List[Path], List[int]]:
    """
    Discover volume directories in datadir.
    
    Args:
        datadir: Root data directory
        volume_prefix: Prefix for volume directories (e.g., 'volume' for volume1/, volume2/)
    
    Returns:
        Tuple of (sorted list of volume directory paths, list of volume numbers)
    """
    datadir = Path(datadir)
    
    # Find all subdirectories that start with volume_prefix
    volume_data = []
    for item in datadir.iterdir():
        if item.is_dir() and item.name.startswith(volume_prefix):
            try:
                # Extract number from volumeX
                num = int(item.name.replace(volume_prefix, ''))
                volume_data.append((item, num))
            except ValueError:
                # Skip directories without valid numbers
                continue
    
    # Sort by volume number
    volume_data.sort(key=lambda x: x[1])
    
    volume_dirs = [item[0] for item in volume_data]
    volume_numbers = [item[1] for item in volume_data]
    
    return volume_dirs, volume_numbers


def load_single_volume(volume_path: Union[str, Path], 
                       volume_id: str,
                       confmap: bool = False) -> VolumeDataset:
    """
    Load a single volume from a directory.
    
    Args:
        volume_path: Path to volume directory
        volume_id: Identifier for this volume
        confmap: Whether to load confidence maps instead of images
    
    Returns:
        VolumeDataset object
    """
    volume_path = Path(volume_path)
    
    # Load poses
    poses_file = volume_path / "poses.npy"
    if not poses_file.exists():
        raise FileNotFoundError(f"poses.npy not found in {volume_path}")
    poses = np.load(poses_file)
    
    # Load images or confidence maps
    if confmap:
        imgs_file = volume_path / "confidence_maps.npy"
    else:
        imgs_file = volume_path / "images.npy"
    
    if not imgs_file.exists():
        raise FileNotFoundError(f"{imgs_file.name} not found in {volume_path}")
    
    images = np.load(imgs_file)
    
    # Validate shapes
    if poses.shape[0] != images.shape[0]:
        raise ValueError(
            f"Mismatch in {volume_path}: images {images.shape[0]} vs poses {poses.shape[0]}"
        )
    
    return VolumeDataset(volume_id, str(volume_path), images, poses)


def parse_split_config(split_str: str, n_volumes: int, volume_numbers: Optional[List[int]] = None) -> List[int]:
    """
    Parse split configuration string into indices or volume numbers.
    
    Formats supported:
    - "0,1,2" -> [0, 1, 2] (volume numbers if volume_numbers provided, else indices)
    - "0,20,110" -> [0, 20, 110] (actual volume numbers)
    - "0-5" -> [0, 1, 2, 3, 4, 5] (range of indices)
    - "last" -> [n_volumes - 1] (last volume index)
    - "last-2" -> [n_volumes - 2] (second-to-last volume index)
    - "even" -> [0, 2, 4, ...] (even indices)
    - "odd" -> [1, 3, 5, ...] (odd indices)
    - "all" -> all volume indices
    
    Args:
        split_str: Configuration string
        n_volumes: Total number of volumes
        volume_numbers: Optional list of actual volume numbers (e.g., [0, 20, 30, 110])
                       If provided, returns indices into this list for numeric configs
    
    Returns:
        List of indices (into volume_numbers list if provided, else direct indices)
    """
    if not split_str or split_str.lower() == 'none':
        return []
    
    split_str = split_str.strip().lower()
    
    # Handle special keywords (always return indices)
    if split_str == 'all':
        return list(range(n_volumes))
    elif split_str == 'last':
        return [n_volumes - 1] if n_volumes > 0 else []
    elif split_str.startswith('last-'):
        offset = int(split_str.split('-')[1])
        idx = n_volumes - 1 - offset
        return [idx] if idx >= 0 else []
    elif split_str == 'even':
        return [i for i in range(n_volumes) if i % 2 == 0]
    elif split_str == 'odd':
        return [i for i in range(n_volumes) if i % 2 == 1]
    
    # Handle ranges: "0-5" (always treated as index ranges)
    if '-' in split_str and not split_str.startswith('last'):
        parts = split_str.split('-')
        start, end = int(parts[0]), int(parts[1])
        end = end // 10
        return list(range(start, end + 1))
    
    # Handle comma-separated: "0,1,2" or "0,20,110"
    if ',' in split_str:
        numbers = [int(x.strip()) for x in split_str.split(',')]
        
        # If volume_numbers provided, map volume numbers to indices
        if volume_numbers is not None:
            indices = []
            for num in numbers:
                try:
                    idx = volume_numbers.index(num)
                    indices.append(idx)
                except ValueError:
                    raise ValueError(f"Volume {num} not found in available volumes: {volume_numbers}")
            return indices
        else:
            return numbers
    
    # Single value
    try:
        num = int(split_str)
        # If volume_numbers provided, map to index
        if volume_numbers is not None:
            try:
                return [volume_numbers.index(num)]
            except ValueError:
                raise ValueError(f"Volume {num} not found in available volumes: {volume_numbers}")
        else:
            return [num]
    except ValueError:
        raise ValueError(f"Invalid split configuration: {split_str}")


def auto_split_indices(n_volumes: int, 
                       full_volume_mode: bool = False) -> Dict[str, List[int]]:
    """
    Automatically determine train/val/test split.
    
    Args:
        n_volumes: Number of volumes
        full_volume_mode: Whether in 3D volume mode
    
    Returns:
        Dict with 'train', 'val', 'test' keys
    """
    if full_volume_mode:
        # For 3D volumes, use last for test, second-last for val
        if n_volumes >= 3:
            return {
                'train': list(range(n_volumes - 2)),
                'val': [n_volumes - 2],
                'test': [n_volumes - 1]
            }
        elif n_volumes == 2:
            return {
                'train': [0],
                'val': [1],
                'test': [1]
            }
        else:
            return {
                'train': [0],
                'val': [0],
                'test': [0]
            }
    else:
        # For 2D slices, use last slice for both val and test
        return {
            'train': list(range(n_volumes - 1)) if n_volumes > 1 else [0],
            'val': [n_volumes - 1] if n_volumes > 0 else [],
            'test': [n_volumes - 1] if n_volumes > 0 else []
        }


def load_multi_volume_data(datadir: str,
                           confmap: bool = False,
                           full_volume_mode: bool = False,
                           train_volumes: Optional[str] = None,
                           val_volumes: Optional[str] = None,
                           test_volumes: Optional[str] = None,
                           volume_prefix: str = "volume") -> DatasetSplit:
    """
    Load data from multiple volume directories.
    
    Args:
        datadir: Root directory containing volume subdirectories
        confmap: Whether to load confidence maps
        full_volume_mode: Whether to treat each volume as a 3D volume
        train_volumes: Volume split config (e.g., "0-5", "even", "0,20,110")
        val_volumes: Validation split config
        test_volumes: Test split config
        volume_prefix: Prefix for volume directories
    
    Returns:
        DatasetSplit object with combined data and split information
    """
    print(f"\n{'='*60}")
    print("LOADING MULTI-VOLUME DATA")
    print(f"{'='*60}")
    print(f"Data directory: {datadir}")
    print(f"Full volume mode: {full_volume_mode}")
    
    # Discover volumes
    volume_dirs, volume_numbers = discover_volumes(datadir, volume_prefix)
    
    if not volume_dirs:
        raise FileNotFoundError(
            f"No {volume_prefix}* directories found in {datadir}. "
            f"Expected structure: {datadir}/{volume_prefix}0/, {volume_prefix}20/, etc."
        )
    
    print(f"Found {len(volume_dirs)} volumes:")
    for vdir, vnum in zip(volume_dirs, volume_numbers):
        print(f"  - {vdir.name} (volume number: {vnum})")
    
    # Load all volumes
    volumes = []
    for vdir in volume_dirs:
        vol = load_single_volume(vdir, vdir.name, confmap)
        volumes.append(vol)
        print(f"Loaded {vol}")
    
    # Determine splits
    n_volumes = len(volumes)
    
    if train_volumes or val_volumes or test_volumes:
        # Use explicit split configuration
        i_train = parse_split_config(train_volumes or "", n_volumes, volume_numbers)
        i_val = parse_split_config(val_volumes or "", n_volumes, volume_numbers)
        i_test = parse_split_config(test_volumes or "", n_volumes, volume_numbers)
        print(f"\nUsing explicit split configuration:")
        print(f"  Train volume indices: {i_train} -> volumes {[volume_numbers[i] for i in i_train]}")
        print(f"  Val volume indices: {i_val} -> volumes {[volume_numbers[i] for i in i_val]}")
        print(f"  Test volume indices: {i_test} -> volumes {[volume_numbers[i] for i in i_test]}")
    else:
        # Auto-determine splits
        splits = auto_split_indices(n_volumes, full_volume_mode)
        i_train = splits['train']
        i_val = splits['val']
        i_test = splits['test']
        print(f"\nUsing automatic split:")
        print(f"  Train volumes: {i_train}")
        print(f"  Val volumes: {i_val}")
        print(f"  Test volumes: {i_test}")
    
    # Combine data
    if full_volume_mode:
        # Stack volumes: (N_volumes, D, H, W) and (N_volumes, D, 3, 4)
        all_images = np.stack([vol.images for vol in volumes], axis=0)
        all_poses = np.stack([vol.poses for vol in volumes], axis=0)
        
        # Indices directly correspond to volumes
        split_indices = {
            'train': i_train,
            'val': i_val,
            'test': i_test
        }
    else:
        # Concatenate all frames from all volumes
        all_images = np.concatenate([vol.images for vol in volumes], axis=0)
        all_poses = np.concatenate([vol.poses for vol in volumes], axis=0)
        
        # Build frame-level indices
        frame_offset = 0
        train_frames = []
        val_frames = []
        test_frames = []
        
        for vol_idx, vol in enumerate(volumes):
            frame_indices = list(range(frame_offset, frame_offset + vol.n_frames))
            
            if vol_idx in i_train:
                train_frames.extend(frame_indices)
            if vol_idx in i_val:
                val_frames.extend(frame_indices)
            if vol_idx in i_test:
                test_frames.extend(frame_indices)
            
            frame_offset += vol.n_frames
        
        split_indices = {
            'train': train_frames,
            'val': val_frames,
            'test': test_frames
        }
    
    # Build metadata
    volume_metadata = []
    for vol, vnum in zip(volumes, volume_numbers):
        volume_metadata.append({
            'volume_id': vol.volume_id,
            'volume_number': vnum,
            'volume_path': vol.volume_path,
            'n_frames': vol.n_frames,
            'shape': list(vol.images.shape)
        })
    
    # Normalize and convert
    #all_images = all_images.astype(np.float32) / 255.0
    all_images = (all_images - all_images.min()) / (all_images.max() - all_images.min())
    all_poses = all_poses.astype(np.float32)
    all_poses[..., :3, 3] *= 0.001  # Convert to meters
    
    dataset = DatasetSplit(
        all_images,
        all_poses,
        split_indices,
        volume_metadata,
        full_volume_mode
    )
    
    print(f"\n{'='*60}")
    print("DATA LOADING COMPLETE")
    print(f"{'='*60}")
    print(f"Total images: {all_images.shape}")
    print(f"Total poses: {all_poses.shape}")
    print(f"Train frames: {len(dataset.i_train)}")
    print(f"Val frames: {len(dataset.i_val)}")
    print(f"Test frames: {len(dataset.i_test)}")
    print(f"{'='*60}\n")
    
    return dataset


def load_legacy_data(datadir: str,
                     confmap: bool = False,
                     pose_path: Optional[str] = None,
                     full_volume_mode: bool = False) -> DatasetSplit:
    """
    Load data using legacy method (single images.npy and poses.npy).
    
    This maintains backward compatibility with existing code.
    
    Args:
        datadir: Data directory
        confmap: Whether to load confidence maps
        pose_path: Optional explicit path to poses file
        full_volume_mode: Whether in 3D volume mode
    
    Returns:
        DatasetSplit object
    """
    print(f"\n{'='*60}")
    print("LOADING LEGACY DATA FORMAT")
    print(f"{'='*60}")
    print(f"Data directory: {datadir}")
    
    # Load poses
    if pose_path:
        poses = np.load(pose_path)
    else:
        poses = np.load(os.path.join(datadir, "poses.npy"))
    
    # Load images
    if confmap:
        imgs = np.load(os.path.join(datadir, "confidence_maps.npy"))
    else:
        imgs = np.load(os.path.join(datadir, "images.npy"))
    
    # Validate
    if not full_volume_mode and poses.shape[0] != imgs.shape[0]:
        raise ValueError(
            f"Mismatch: images {imgs.shape[0]} vs poses {poses.shape[0]}"
        )
    
    # Normalize
    #imgs = imgs.astype(np.float32) / 255.0
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
    poses = poses.astype(np.float32)
    poses[..., :3, 3] *= 0.001
    
    # Auto-determine splits
    n_items = imgs.shape[0]
    
    if full_volume_mode:
        splits = auto_split_indices(n_items, True)
    else:
        # Legacy: use last frame for test, same for val
        splits = {
            'train': list(range(n_items - 1)) if n_items > 1 else [0],
            'val': [n_items - 1] if n_items > 0 else [],
            'test': [n_items - 1] if n_items > 0 else []
        }
    
    volume_metadata = [{
        'volume_id': 'legacy',
        'volume_path': datadir,
        'n_frames': n_items,
        'shape': list(imgs.shape)
    }]
    
    dataset = DatasetSplit(imgs, poses, splits, volume_metadata, full_volume_mode)
    
    print(f"Loaded images: {imgs.shape}")
    print(f"Loaded poses: {poses.shape}")
    print(f"Train: {len(dataset.i_train)}, Val: {len(dataset.i_val)}, Test: {len(dataset.i_test)}")
    print(f"{'='*60}\n")
    
    return dataset


def load_us_data(datadir: str,
                 confmap: bool = False,
                 pose_path: Optional[str] = None,
                 full_volume_mode: bool = False,
                 use_multi_volume_data_loading: bool = False,
                 train_volumes: Optional[str] = None,
                 val_volumes: Optional[str] = None,
                 test_volumes: Optional[str] = None,
                 volume_prefix: str = "volume") -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Unified data loading function with automatic detection.
    
    Args:
        datadir: Data directory
        confmap: Load confidence maps
        pose_path: Explicit pose file path (legacy mode only)
        full_volume_mode: 3D volume mode
        use_multi_volume_data_loading: Force multi-volume loading
        train_volumes: Train split config
        val_volumes: Val split config
        test_volumes: Test split config
        volume_prefix: Volume directory prefix
    
    Returns:
        (images, poses, i_test) - for backward compatibility
        Note: Access full split info via dataset.i_train, dataset.i_val
    """
    # Auto-detect mode if not specified
    if not use_multi_volume_data_loading and not pose_path:
        # Check if multi-volume structure exists
        volume_dirs, volume_numbers = discover_volumes(datadir, volume_prefix)
        if volume_dirs:
            use_multi_volume_data_loading = True
            print(f"Auto-detected multi-volume structure ({len(volume_dirs)} volumes)")
    
    # Load data
    if use_multi_volume_data_loading:
        dataset = load_multi_volume_data(
            datadir, confmap, full_volume_mode,
            train_volumes, val_volumes, test_volumes,
            volume_prefix
        )
    else:
        dataset = load_legacy_data(datadir, confmap, pose_path, full_volume_mode)
    
    # Return in legacy format for backward compatibility
    # The calling code can access dataset.i_train, dataset.i_val separately
    i_test = dataset.i_test[0] if dataset.i_test else 0
    
    # Store dataset globally for access to splits
    # (Alternative: modify calling code to accept DatasetSplit directly)
    return dataset.images, dataset.poses, i_test, dataset
