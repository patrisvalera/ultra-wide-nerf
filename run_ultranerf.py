import os
import random
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai.losses.ssim_loss import SSIMLoss
from monai.losses import LocalNormalizedCrossCorrelationLoss
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from dataset_utils.load_us import load_us_data
from nerf_utils import create_nerf, img2mse, render_us, compute_loss, compute_regularization, set_seed
from unerf_config import config_parser

from evaluation_metrics import VolumeMetrics, evaluate_volume_with_slices, print_evaluation_results, save_metrics_to_file

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.95)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sampling import *

def visualize_2d_slice(rendering_output, target, pts_image, W, H, output_path, img_i, iteration, mode="train"):
    """
    Visualize 2D slice rendering outputs and target.
    
    Args:
        rendering_output: Dictionary of rendering outputs
        target: Target tensor
        pts_image: Sampling points for remapping
        W, H: Image dimensions
        output_path: Path to save visualization
        img_i: Image index
        iteration: Training iteration
        mode: "train" or "eval"
    """
    plt.figure(figsize=(16, 9))
    
    indexes = 11 if args.output_ch == 5 else 12
    for j, m in enumerate(list(rendering_output.keys())[:indexes]):
        plt.subplot(3, 4, j + 1)
        plt.title(m)
        array = rendering_output[m].detach().cpu().numpy()[0, 0].T
        if convex.config['use_convex_mode']:
            array = remap_output_to_original_image(array.T, pts_image, W, H)
        plt.imshow(array)

    plt.subplot(3, 4, 12)
    plt.title("Target")
    if convex.config['use_convex_mode']:
        array = target.detach().cpu().numpy()[0, 0]
        array = remap_output_to_original_image(array, pts_image, W, H)
        plt.imshow(array)
    else:
        plt.imshow(target.detach().cpu().numpy()[0, 0].T)

    plt.savefig(
        os.path.join(output_path, "{:08d}_{}_{}.png".format(iteration, mode, img_i)),
        bbox_inches="tight",
        dpi=200,
    )
    plt.close()


def visualize_2d_slice_comparison(output_image, target, pts_image, W, H, output_path, img_i, iteration, mode="train"):
    """
    Visualize side-by-side comparison of output and target with multiple colormaps.
    
    Args:
        output_image: Output image tensor
        target: Target tensor
        pts_image: Sampling points for remapping
        W, H: Image dimensions
        output_path: Path to save visualization
        img_i: Image index
        iteration: Training iteration
        mode: "train" or "eval"
    """
    for cmap in ['gray', 'viridis', 'plasma']:
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        output_image_np = output_image.detach().cpu().numpy()[0, 0].T
        if convex.config['use_convex_mode']:
            output_image_np = remap_output_to_original_image(output_image_np.T, pts_image, W, H)
        plt.imshow(output_image_np, cmap=cmap)
        plt.title(f'NeRF Volume Slice {img_i}' + (f' ({mode.capitalize()})' if mode == "eval" else ''))
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        if convex.config['use_convex_mode']:
            target_img = target.detach().cpu().numpy()[0, 0]
            target_img = remap_output_to_original_image(target_img, pts_image, W, H)
        else:
            target_img = target.detach().cpu().numpy()[0, 0].T
        plt.imshow(target_img, cmap=cmap)
        plt.title(f'Original Volume Slice {img_i}' + (f' ({mode.capitalize()})' if mode == "eval" else ''))
        plt.axis('off')
        
        plt.savefig(
            os.path.join(output_path, "{:08d}_{}_{}_{}.png".format(iteration, cmap, mode, img_i)),
            bbox_inches="tight",
            dpi=200,
        )
        plt.close()


def visualize_3d_volume_slices(rendering_output, target, output_image, pts_image, W, H, D, images_shape, 
                                output_path, img_i, iteration, mode="train", n_slices=3, 
                                nerf_original_vis=True, save_volume=True):
    """
    Visualize random slices from 3D volume and optionally save the full volume.
    
    Args:
        rendering_output: Dictionary of rendering outputs
        target: Target tensor
        output_image: Output image tensor
        pts_image: Sampling points for remapping
        W, H, D: Volume dimensions
        images_shape: Shape of images array for modulo operation
        output_path: Path to save visualization
        img_i: Image index
        iteration: Training iteration
        mode: "train" or "eval"
        n_slices: Number of random slices to visualize
        nerf_original_vis: Whether to remap to original image space
        save_volume: Whether to save the full 3D volume as .npy file
    """
    # Select random slices to visualize
    indices = random.sample(range(D), min(n_slices, D))
    
    for k in indices:
        plt.figure(figsize=(16, 9))
        
        indexes = 11 if args.output_ch == 5 else 12
        for j, m in enumerate(list(rendering_output.keys())[:indexes]):
            plt.subplot(3, 4, j + 1)
            plt.title(m)
            array = rendering_output[m].detach().cpu().numpy()[0, k].T
            
            if convex.config['use_convex_mode']:
                if nerf_original_vis:
                    array = remap_output_to_original_image(array.T, pts_image, W, H, 
                                                          _interpolate=1)#, interpolation_method="nearest")
                # else: keep as is (pixel space)
            plt.imshow(array)

        plt.subplot(3, 4, 12)
        plt.title("Target")
        if convex.config['use_convex_mode']:
            array = target.detach().cpu().numpy()[0, k % images_shape]
            if nerf_original_vis:
                array = remap_output_to_original_image(array, pts_image, W, H, _interpolate=1)
            else:
                array = array.T
            plt.imshow(array)
        else:
            plt.imshow(target.detach().cpu().numpy()[0, k % images_shape].T)

        plt.savefig(
            os.path.join(output_path, "{:08d}_{}_{}_{}.png".format(iteration, k, mode, img_i)),
            bbox_inches="tight",
            dpi=200,
        )
        plt.close()
    
    # Save full 3D volume if requested
    if save_volume:
        slices = []
        for d in range(D):
            array = output_image.detach().cpu().numpy()[0, d]
            slices.append(remap_output_to_original_image(array, pts_image, W, H))
        output_volume = np.stack(slices, axis=0)
        
        # Build target volume in same format for metrics computation
        target_slices = []
        for d in range(D):
            if convex.config['use_convex_mode']:
                array = target.detach().cpu().numpy()[0, d % images_shape]
                if nerf_original_vis:
                    array = remap_output_to_original_image(array, pts_image, W, H, _interpolate=1)
                target_slices.append(array)
            else:
                target_slices.append(target.detach().cpu().numpy()[0, d % images_shape])
        target_volume = np.stack(target_slices, axis=0)
        
        # ========================= COMPUTE EVALUATION METRICS =========================
        print(f"\n{'='*70}")
        print(f"Computing evaluation metrics for {mode} volume {img_i} at iteration {iteration}")
        print(f"{'='*70}")
        
        try:
            # Compute all metrics (3D volume + 2D slices)
            metrics_results = evaluate_volume_with_slices(
                pred_volume=output_volume,
                target_volume=target_volume,
                device=device
            )
            
            # Print metrics to console
            print_evaluation_results(metrics_results)
            
            # Save metrics to text file
            metrics_path = os.path.join(output_path, "metrics_{:08d}_{}_{}.txt".format(iteration, mode, img_i))
            save_metrics_to_file(metrics_results, metrics_path)
            
            # Save metrics as numpy dict for later analysis
            metrics_npy_path = os.path.join(output_path, "metrics_{:08d}_{}_{}.npy".format(iteration, mode, img_i))
            np.save(metrics_npy_path, metrics_results)
            print(f"Metrics saved to: {metrics_path} and {metrics_npy_path}")
            
        except Exception as e:
            print(f"Warning: Failed to compute metrics: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"{'='*70}\n")
        # ========================= END METRICS COMPUTATION =========================
        
        # Visualize middle slices
        plt.figure(figsize=(20, 8))
        
        # Coronal slice (volume[:, H//2, :])
        plt.subplot(2, 2, 1)
        coronal_pred = output_volume[:, output_volume.shape[1] // 2, :]
        plt.imshow(coronal_pred, cmap='gray')
        plt.title(f'Coronal Middle Slice (Predicted) - Vol {img_i}')
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        coronal_target = target_volume[:, target_volume.shape[1] // 2, :]
        plt.imshow(coronal_target, cmap='gray')
        plt.title(f'Coronal Middle Slice (Target) - Vol {img_i}')
        plt.axis('off')
        
        # Axial slice (volume[:, :, W//2])
        plt.subplot(2, 2, 3)
        axial_pred = output_volume[:, :, output_volume.shape[2] // 2]
        plt.imshow(axial_pred, cmap='gray')
        plt.title(f'Axial Middle Slice (Predicted) - Vol {img_i}')
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        axial_target = target_volume[:, :, target_volume.shape[2] // 2]
        plt.imshow(axial_target, cmap='gray')
        plt.title(f'Axial Middle Slice (Target) - Vol {img_i}')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_path, "{:08d}_{}_{}_{}.png".format(iteration, "middle_cut", mode, img_i)),
            bbox_inches="tight",
            dpi=200,
        )
        plt.close()
        
        volume_path = os.path.join(output_path, "volume_{:08d}_{}_{}.npy".format(iteration, mode, img_i))
        np.save(volume_path, output_volume)
        print(f"Saved {mode} volume shape {output_volume.shape} to: {volume_path}")


def parse_convex_args(args):
    """Parses command line arguments for convex mode."""
    convex.config['use_convex_mode'] = False
    if args.use_convex_mode:
        print("Running in convex mode...")
        convex.config['use_convex_mode'] = True
        convex.config['use_mip'] = args.use_mip
        convex.config['full_volume_mode'] = args.full_volume_mode
        convex.config['multires'] = args.multires
        
        # MIP-NeRF parameters
        convex.config['use_elongation'] = args.use_elongation
        convex.config['max_elongation'] = args.max_elongation
        convex.config['mip_voxel_radius_lateral'] = args.mip_voxel_radius_lateral
        convex.config['mip_voxel_radius_depth'] = args.mip_voxel_radius_depth
        
        # Sampling strategy parameters (Phase 2: Concentric circles)
        convex.config['sampling_strategy'] = args.sampling_strategy
        convex.config['n_circles'] = args.n_circles
        convex.config['base_rays_per_circle'] = args.base_rays_per_circle
        convex.config['ray_density_multiplier'] = args.ray_density_multiplier
        convex.config['density_growth_type'] = args.density_growth_type
        
        # Spherical (elevation-angle) volume mode
        convex.config['use_spherical_volume'] = args.use_spherical_volume
        convex.config['n_rays_elevation'] = args.n_rays_elevation
        convex.config['opening_angle_elevation'] = args.opening_angle_elevation
        
        # Radial path mode (tree-ring sampling)
        convex.config['use_radial_path_mode'] = args.use_radial_path_mode
        if args.use_radial_path_mode and args.sampling_strategy != 'concentric_circles':
            print("WARNING: Radial path mode requires concentric_circles sampling strategy.") #TODO
            convex.config['sampling_strategy'] = 'concentric_circles'
            args.sampling_strategy = 'concentric_circles'
        
        if args.convex_n_rays is not None:
            convex.config['convex_n_rays'] = args.convex_n_rays

        if args.convex_n_samples is not None:
            convex.config['convex_n_samples'] = args.convex_n_samples

        if args.convex_center_x is not None and args.convex_center_y is not None:
            convex.config['convex_center'] = (args.convex_center_x, args.convex_center_y)

        if args.convex_fan_angle is not None:
            convex.config['convex_angle'] = args.convex_fan_angle

        if args.convex_big_radius is not None:
            convex.config['convex_radius'] = args.convex_big_radius

        if args.convex_small_radius is not None:
            convex.config['convex_radius2'] = args.convex_small_radius

        if args.convex_scaling is not None:
            convex.config['convex_scaling_x'] = args.convex_scaling
            convex.config['convex_scaling_y'] = args.convex_scaling
            args.convex_scaling_x = args.convex_scaling
            args.convex_scaling_y = args.convex_scaling

        if args.convex_scaling_x is not None:
            convex.config['convex_scaling_x'] = args.convex_scaling_x

        if args.convex_scaling_y is not None:
            convex.config['convex_scaling_y'] = args.convex_scaling_y

        # Is value is None in configuration file use args
        if "convex_use_compiled_cache" not in convex.config or convex.config['convex_use_compiled_cache'] is None:
            convex.config['convex_use_compiled_cache'] = True if args.convex_use_compiled_cache else False

        # Is value is None in configuration file use args
        if "convex_use_compiled_cache_on_the_fly" not in convex.config or convex.config['convex_use_compiled_cache_on_the_fly'] is None:
            convex.config['convex_use_compiled_cache_on_the_fly'] = True if args.convex_use_compiled_cache_on_the_fly else False

        # Is value is None in configuration file use args
        if "convex_save_meta_files" not in convex.config or convex.config['convex_save_meta_files'] is None:
            convex.config['convex_save_meta_files'] = True if args.convex_save_meta_files else False

        if "convex_max_dist" not in convex.config or convex.config['convex_max_dist'] is None:
            convex.config['convex_max_dist'] = 1 / (convex.config['convex_radius'] - convex.config['convex_radius2']) / 0.001
    
    # Debug visualization settings
    if hasattr(args, 'debug_viz_render') and args.debug_viz_render:
        convex.config['debug_viz_render'] = True
    else:
        convex.config['debug_viz_render'] = False

    if convex.config['use_convex_mode']:
        args.N_samples = convex.config['convex_n_samples']

    return args

parser = config_parser()
args = parser.parse_args()
args = parse_convex_args(args)

def get_target(images, img_i, pts_image):
    """ Loads the image as a tensor by its index.
        For convex mode it remaps original image to reflect only points
        along sampling rays according to sampling points (pts_image).
        Sampling points are generated by calling calc_image_sampling_points_convex() method.
        If cache is enabled, it loads from cache."""
    global cached_images
    if not convex.config['use_convex_mode']:
        return images[img_i].transpose(0, 1)  # equivalent to tf.transpose(images[img_i])
    else:
        if convex.config['convex_use_compiled_cache']:
            if cached_images[img_i] is not None:
                out = cached_images[img_i]
            else:
                out = remap_target_for_comparison(images[img_i], pts_image)
                cached_images[img_i] = out
        else:
            out = remap_target_for_comparison(images[img_i], pts_image)

        return out

def train():
    global cached_images

    set_seed(args.random_seed)

    if args.dataset_type == "us":
        # Load data using new flexible data loader
        result = load_us_data(
            datadir=args.datadir, 
            confmap=args.confmap, 
            pose_path=args.pose_path, 
            full_volume_mode=args.full_volume_mode,
            use_multi_volume_data_loading=args.use_multi_volume_data_loading,
            train_volumes=args.train_volumes,
            val_volumes=args.val_volumes,
            test_volumes=args.test_volumes,
            volume_prefix=args.volume_prefix
        )
        
        # Unpack result (handles both old and new format)
        if len(result) == 4:
            images, poses, i_test, dataset = result
            # Use dataset split indices
            i_train = dataset.i_train
            i_val = dataset.i_val
            i_test = dataset.i_test
            
            # Save split info for reproducibility
            split_info_path = os.path.join(args.basedir, args.expname, "split_info.json")
            os.makedirs(os.path.join(args.basedir, args.expname), exist_ok=True)
            dataset.save_split_info(split_info_path)
        else:
            # Legacy format (reconstruction mode)
            images, poses, i_test = result[0], result[1], result[2]
            if not isinstance(i_test, list):
                i_test = [i_test]
            i_val = i_test
            i_train = np.array([i for i in np.arange(int(images.shape[0]))])
        
        # Transform poses to desired coordinate system
        # Apply if you need to change which axis represents depth
        if args.transform_poses:
            poses = transform_poses_coordinate_system(poses, axis_mapping=args.pose_axis_mapping)
        
        # Ensure proper shape for full_volume_mode
        if args.full_volume_mode:
            if images.ndim == 3:
                images = images[np.newaxis, :, :, :]
            print("Using full volume mode, images shape:", images.shape)

        print("Test {}, train {}".format(len(i_test), len(i_train)))
        print(f"Train indices: {i_train},\n Val indices: {i_val},\n Test indices: {i_test}")
    else:
        print("Unknown dataset type:", args.dataset_type, ". Currently not implemented for Ultra-NeRF.")
        return

    print("Images min:", images.min(), "max:", images.max())

    # Cast intrinsics to right types
    # The poses are not normalized. We scale down the space.
    # It is possible to normalize poses and remove scaling.
    scaling = 0.001
    near = 0
    D, H, W = images.shape[-3], images.shape[-2], images.shape[-1]
    if not convex.config['use_convex_mode']:
        probe_depth = args.probe_depth * scaling
        probe_width = args.probe_width * scaling
        far = probe_depth
        sy = probe_depth / float(H)
        sx = probe_width / float(W)
    else:
        center, cx, cy, num_rays, angle, radius, radius2 = convex.get_convex_settings()
        sy = convex.config['convex_scaling_y'] * scaling
        sx = convex.config['convex_scaling_x'] * scaling
        far = (radius - radius2) * convex.config['convex_scaling_y'] * scaling

    sh = sy
    sw = sx
    print("H, W, D:", H, W, D)
    print("Near, Far:", near, far)
    print("Scaling factors sy, sx:", sy, sx)
    print("Scaling factors sh, sw:", sh, sw)

    # Create base dir
    basedir = args.basedir
    expname = args.expname
    if convex.config['use_convex_mode'] and convex.config['convex_save_meta_files']:
        os.makedirs(os.path.join(basedir, "meta"), exist_ok=True)

    # Create tensorboard writer
    if args.tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(basedir, "summaries", expname))

    # Create log dir and copy the config file
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, "config.txt")
        with open(f, "w") as file:
            file.write(open(args.config, "r").read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, optimizer, _, embed_fn = create_nerf(
        args, device=device
    )

    bds_dict = {
        "near": near,
        "far": far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    
    N_iters = args.n_iters
    print("TRAIN views are", i_train)
    print("TEST views are", i_test)
    print("VAL views are", i_val)

    # Losses
    ssim_weight = args.ssim_lambda
    l2_weight = 1.0 - ssim_weight
    ssim_loss = SSIMLoss(
        spatial_dims=3 if args.full_volume_mode else 2,
        data_range=1.0,
        kernel_type="gaussian",
        win_size=args.ssim_filter_size,
        k1=0.01,
        k2=0.1,
    )
    losses = {"l2": img2mse,
              "ssim": ssim_loss,
              "lncc": LocalNormalizedCrossCorrelationLoss(spatial_dims=3)}
    
    # Validate chunk size for full_volume_mode with convex_mode and adjust volumes if necessary
    if args.full_volume_mode and convex.config['use_convex_mode']:
        convex_n_rays = convex.config['convex_n_rays']
        total_rays_per_volume = D * convex_n_rays
        
        if args.chunk < total_rays_per_volume:
            # Chunk size is smaller than full volume, must be multiple of convex_n_rays
            if args.chunk % convex_n_rays != 0:
                raise ValueError(
                    f"When chunk ({args.chunk}) < D*convex_n_rays ({D}*{convex_n_rays}={total_rays_per_volume}), "
                    f"chunk must be a multiple of convex_n_rays ({convex_n_rays}). "
                    f"Current chunk % convex_n_rays = {args.chunk % convex_n_rays}"
                )
            
            depth_slices_per_chunk = args.chunk // convex_n_rays
            print(f"Chunk size validation passed: chunk={args.chunk} is a multiple of convex_n_rays={convex_n_rays}")
            print(f"  Processing {depth_slices_per_chunk} depth slices at a time (out of {D} total)")
            
            # Split volumes into smaller sub-volumes that fit in chunk size
            print(f"\n{'='*60}")
            print(f"SPLITTING VOLUMES INTO SUB-VOLUMES")
            print(f"{'='*60}")
            print(f"Original volumes: {images.shape[0]}, depth per volume: {D}")
            
            new_images = []
            new_poses = []
            original_to_new_indices = {}  # Map original volume index to list of new indices
            
            for vol_idx in range(images.shape[0]):
                volume = images[vol_idx]  # Shape: [D, H, W]
                poses_vol = poses[vol_idx]  # Shape: [D, 3, 4]
                
                # Split this volume into chunks
                n_chunks = (D + depth_slices_per_chunk - 1) // depth_slices_per_chunk
                new_vol_indices = []
                
                for chunk_idx in range(n_chunks):
                    start_depth = chunk_idx * depth_slices_per_chunk
                    end_depth = min(start_depth + depth_slices_per_chunk, D)
                    
                    # Extract sub-volume
                    sub_volume = volume[start_depth:end_depth, :, :]  # [depth_chunk, H, W]
                    sub_poses = poses_vol[start_depth:end_depth, :, :]  # [depth_chunk, 3, 4]
                    
                    new_images.append(sub_volume)
                    new_poses.append(sub_poses)
                    new_vol_indices.append(len(new_images) - 1)
                
                original_to_new_indices[vol_idx] = new_vol_indices
            
            # Stack into new arrays
            images = np.array(new_images)
            poses = np.array(new_poses)
            
            print(f"New volumes: {images.shape[0]}, depth per sub-volume: varies (max {depth_slices_per_chunk})")
            print(f"New images shape: {images.shape}")
            print(f"New poses shape: {poses.shape}")
            
            # Update train/val/test indices
            def remap_indices(old_indices, mapping):
                new_indices = []
                for old_idx in old_indices:
                    if old_idx in mapping:
                        new_indices.extend(mapping[old_idx])
                return np.array(new_indices)
            
            old_i_train = i_train.copy() if isinstance(i_train, np.ndarray) else np.array(i_train)
            old_i_val = i_val.copy() if isinstance(i_val, np.ndarray) else np.array(i_val)
            old_i_test = i_test.copy() if isinstance(i_test, list) else list(i_test)
            
            i_train = remap_indices(old_i_train, original_to_new_indices)
            i_val = remap_indices(old_i_val, original_to_new_indices)
            i_test = remap_indices(old_i_test, original_to_new_indices).tolist()
            
            print(f"\nIndex remapping:")
            print(f"  Train: {len(old_i_train)} volumes → {len(i_train)} sub-volumes")
            print(f"  Val: {len(old_i_val)} volumes → {len(i_val)} sub-volumes")
            print(f"  Test: {len(old_i_test)} volumes → {len(i_test)} sub-volumes")
            print(f"{'='*60}\n")
            
            # Update D to reflect new depth
            D = images.shape[1]
    
    start = start + 1
    
    coord_transform = CoordinateTransformer(W - 1, H - 1)
    pts_image = None
    if convex.config['use_convex_mode']:
        print("Using convex mode for sampling points...")
        
        pts_image = calc_image_sampling_points_convex(**render_kwargs_train)
            
        img = np.ones((H, W), dtype=np.uint8) * 255
        pil_img = Image.fromarray(img, "L")
        pts_pixel_coord = coord_transform.image_to_pixel(pts_image)
        ccord = pts_pixel_coord.reshape((-1, 2))
        img[ccord[:, 1], ccord[:, 0]] = 0

        pil_img = Image.fromarray(img, "L")
        pil_img.save(os.path.join(basedir, expname,"sampling_points.png"))
        assert pts_image.shape == pts_pixel_coord.shape
        pts_image = pts_pixel_coord
        
        if convex.config['full_volume_mode']:            
            # Repeat pts_image along a new dimension to match volume depth
            pts_image_volume = np.repeat(pts_image[None, ...], images.shape[1], axis=0)  # shape: [D, H, W, 2]

        """If cache is enabled in convex mode we need to load them from file.
            If convex.config['convex_use_compiled_cache_on_the_fly'] is enabled it will generate 
            as soon as image is requested and store it in memory cache, later those values 
            will be flashed to the file.
        """
        if convex.config['convex_use_compiled_cache']:
            path = args.basedir
            if not convex.config['convex_use_compiled_cache_on_the_fly']:
                print('Preparing converted data...')
                cached_images = cache_targets_for_comparison(images, pts_pixel_coord, path)
                print('Preparing converted data...DONE')
            else:
                print('Use cache on the fly. Loading initial values...')
                cached_images = np.empty(shape=(images.shape[0]), dtype='O')
                converted_image_file = get_filename_targets_for_comparison(path, pts_pixel_coord)
                if os.path.isfile(converted_image_file):
                    cached_images = np.load(converted_image_file, allow_pickle=True)
                print('Use cache on the fly. Loading initial values... DONE')
    
    # Radial path mode initialization (tree-ring sampling) (TODO)
    radial_origins = None
    radial_directions = None
    radial_metadata = None
    radial_paths = None
    radial_path_metadata = None
    
    if convex.config.get('use_radial_path_mode', False):
        from sampling import define_rays_concentric_circles, build_radial_paths_from_concentric
        
        print("\n" + "="*60)
        print("INITIALIZING RADIAL PATH (TREE-RING) SAMPLING MODE")
        print("="*60)
        
        # Get normalized radii
        norm_inner_radius = convex.config['convex_radius2'] / convex.config['convex_radius']
        norm_outer_radius = 1.0
        
        # Generate concentric circles
        print(f"Generating concentric circles:")
        print(f"  Circles: {convex.config['n_circles']}")
        print(f"  Base rays: {convex.config['base_rays_per_circle']}")
        print(f"  Multiplier: {convex.config['ray_density_multiplier']}")
        print(f"  Growth type: {convex.config['density_growth_type']}")
        
        radial_origins, radial_directions, radial_metadata = define_rays_concentric_circles(
            n_circles=convex.config['n_circles'],
            inner_radius=norm_inner_radius,
            outer_radius=norm_outer_radius,
            base_rays_per_circle=convex.config['base_rays_per_circle'],
            ray_density_multiplier=convex.config['ray_density_multiplier'],
            density_growth_type=convex.config['density_growth_type']
        )
        
        print(f"\nGenerated rays:")
        print(f"  Rays per circle: {radial_metadata['rays_per_circle']}")
        print(f"  Total rays: {radial_metadata['total_rays']}")
        
        # Build radial paths
        print(f"\nBuilding radial paths...")
        radial_paths, radial_path_metadata = build_radial_paths_from_concentric(
            radial_metadata,
            include_center=False
        )
        
        print(f"  Number of paths: {radial_path_metadata['n_paths']}")
        print(f"  Path length: {radial_path_metadata['path_lengths'][0]} nodes each")
        print("="*60 + "\n")
        
        # Store in convex config for access during rendering
        convex.config['_radial_origins'] = radial_origins
        convex.config['_radial_directions'] = radial_directions
        convex.config['_radial_metadata'] = radial_metadata
        convex.config['_radial_paths'] = radial_paths
        convex.config['_radial_path_metadata'] = radial_path_metadata

    print("=== Beginning Training ===")
    training_start_time = time.time()
    total_time_elapsed = 0.0
    
    # Create shuffled training indices to ensure all images/volumes are used each epoch
    train_indices_shuffled = list(i_train)
    np.random.shuffle(train_indices_shuffled)
    train_idx_counter = 0
    
    # Epoch visualization: collect ray origins and endpoints for debugging
    epoch_ray_origins = []
    epoch_ray_endpoints = []
    epoch_poses = []
    epoch_pose_volume_ids = []  # Track which volume each pose belongs to
    epoch_iteration = []
    epoch_volume_ids = []  # Track which volume each sample belongs to
    current_epoch = 0

    for i in trange(start, N_iters + 1):
        print("Training iteration:", i)
        time0 = time.time()
        
        # Reshuffle when we've gone through all training images/volumes (one epoch)
        if train_idx_counter >= len(train_indices_shuffled):
            # Create epoch visualization before reshuffling
            if len(epoch_ray_origins) > 0 and current_epoch == 0:
                print(f"\n{'='*80}")
                print(f"Epoch {current_epoch} complete at iteration {i}")
                print(f"Creating epoch visualization with {len(epoch_ray_origins)} samples...")
                print(f"{'='*80}\n")
                
                # Concatenate all collected data
                all_origins = np.concatenate(epoch_ray_origins, axis=0)
                all_endpoints = np.concatenate(epoch_ray_endpoints, axis=0)
                all_poses_arr = np.array(epoch_poses)
                all_pose_volume_ids = np.array(epoch_pose_volume_ids)
                all_volume_ids = np.concatenate(epoch_volume_ids, axis=0)
                
                # Create visualization directory
                epoch_viz_path = os.path.join(basedir, expname, "epoch_visualizations")
                os.makedirs(epoch_viz_path, exist_ok=True)
                
                # Subsample for visualization to avoid overwhelming the plot (Visualization performed only once)
                max_rays_viz = 5000
                if all_origins.shape[0] > max_rays_viz:
                    viz_indices = np.linspace(0, all_origins.shape[0] - 1, max_rays_viz, dtype=int)
                    viz_origins = all_origins[viz_indices]
                    viz_endpoints = all_endpoints[viz_indices]
                    viz_volume_ids = all_volume_ids[viz_indices]
                else:
                    viz_origins = all_origins
                    viz_endpoints = all_endpoints
                    viz_volume_ids = all_volume_ids
                
                # Generate unique colors for each volume using bright, distinctive colors
                unique_volumes = np.unique(all_volume_ids)
                n_volumes = len(unique_volumes)
                
                # Define bright, distinctive colors: red, green, yellow, blue, cyan, magenta, orange, purple, lime, pink, etc.
                bright_colors = [
                    [1.0, 0.0, 0.0, 1.0],  # Red
                    [0.0, 1.0, 0.0, 1.0],  # Green
                    #[1.0, 1.0, 0.0, 1.0],  # Yellow
                    [0.0, 0.0, 1.0, 1.0],  # Blue
                    [0.0, 1.0, 1.0, 1.0],  # Cyan
                    [1.0, 0.0, 1.0, 1.0],  # Magenta
                    [1.0, 0.5, 0.0, 1.0],  # Orange
                    [0.5, 0.0, 1.0, 1.0],  # Purple
                    [0.5, 1.0, 0.0, 1.0],  # Lime
                    [1.0, 0.0, 0.5, 1.0],  # Pink
                    [0.0, 0.5, 1.0, 1.0],  # Sky blue
                    [1.0, 0.75, 0.0, 1.0], # Gold
                    [0.5, 1.0, 0.5, 1.0],  # Light green
                    [1.0, 0.5, 0.5, 1.0],  # Light red
                    [0.5, 0.5, 1.0, 1.0],  # Light blue
                    [1.0, 1.0, 0.5, 1.0],  # Light yellow
                    [0.75, 0.0, 0.75, 1.0],# Dark magenta
                    [0.0, 0.75, 0.75, 1.0],# Dark cyan
                    [0.75, 0.75, 0.0, 1.0],# Olive
                    [0.75, 0.25, 0.0, 1.0],# Brown-orange
                ]
                
                # If more volumes than predefined colors, cycle through them
                colors = np.array([bright_colors[i % len(bright_colors)] for i in range(n_volumes)])
                volume_to_color = {vol_id: colors[i] for i, vol_id in enumerate(unique_volumes)}
                
                # Map volume IDs to colors for rays
                ray_colors = np.array([volume_to_color[vol_id] for vol_id in viz_volume_ids])
                
                # Map volume IDs to colors for poses (one per pose)
                pose_colors = np.array([volume_to_color[vol_id] for vol_id in all_pose_volume_ids])
                
                # 3D visualization
                fig = plt.figure(figsize=(20, 15))
                
                # Main 3D plot
                ax1 = fig.add_subplot(221, projection='3d')
                ax1.set_title(f'Epoch {current_epoch}: All Rays and Poses (3D) - {n_volumes} Volumes', fontsize=14, fontweight='bold')
                
                # Plot rays (limit for clarity, color by volume)
                for j in range(min(1000, len(viz_origins))):
                    ax1.plot([viz_origins[j, 0], viz_endpoints[j, 0]],
                            [viz_origins[j, 1], viz_endpoints[j, 1]],
                            [viz_origins[j, 2], viz_endpoints[j, 2]],
                            color=ray_colors[j], alpha=0.15, linewidth=0.4)
                
                # Plot ray origins (colored by volume)
                ax1.scatter(viz_origins[:, 0], viz_origins[:, 1], viz_origins[:, 2],
                           c=ray_colors, marker='o', s=15, alpha=0.6, edgecolors='black', linewidths=0.5)
                
                # Plot ray endpoints (colored by volume)
                ax1.scatter(viz_endpoints[:, 0], viz_endpoints[:, 1], viz_endpoints[:, 2],
                           c=ray_colors, marker='.', s=8, alpha=0.4)
                
                # Plot poses (colored by volume, larger markers)
                ax1.scatter(all_poses_arr[:, 0], all_poses_arr[:, 1], all_poses_arr[:, 2],
                           c=pose_colors, marker='^', s=150, alpha=0.9, edgecolors='black',
                           linewidths=2)
                
                # Add legend showing volume IDs
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                             markerfacecolor=volume_to_color[vol_id], 
                                             markersize=10, label=f'Volume {vol_id}',
                                             markeredgecolor='black', markeredgewidth=0.5)
                                  for vol_id in unique_volumes[:min(10, n_volumes)]]  # Limit legend to 10 entries
                if n_volumes > 10:
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                     markerfacecolor='gray', markersize=10, 
                                                     label=f'... and {n_volumes-10} more'))
                ax1.legend(handles=legend_elements, fontsize=9, loc='upper right')
                
                ax1.set_xlabel('X (world)', fontsize=12)
                ax1.set_ylabel('Y (world)', fontsize=12)
                ax1.set_zlabel('Z (world)', fontsize=12)
                ax1.grid(True, alpha=0.3)
                
                # XY projection
                ax2 = fig.add_subplot(222)
                ax2.set_title(f'Epoch {current_epoch}: XY Projection (Top View) - {n_volumes} Volumes', fontsize=14, fontweight='bold')
                ax2.scatter(viz_origins[:, 0], viz_origins[:, 1],
                           c=ray_colors, marker='o', s=10, alpha=0.5, edgecolors='black', linewidths=0.3)
                ax2.scatter(viz_endpoints[:, 0], viz_endpoints[:, 1],
                           c=ray_colors, marker='.', s=5, alpha=0.3)
                ax2.scatter(all_poses_arr[:, 0], all_poses_arr[:, 1],
                           c=pose_colors, marker='^', s=100, alpha=0.9, edgecolors='black',
                           linewidths=2)
                ax2.set_xlabel('X (world)', fontsize=12)
                ax2.set_ylabel('Y (world)', fontsize=12)
                ax2.grid(True, alpha=0.3)
                ax2.set_aspect('equal')
                
                # XZ projection
                ax3 = fig.add_subplot(223)
                ax3.set_title(f'Epoch {current_epoch}: XZ Projection (Front View) - {n_volumes} Volumes', fontsize=14, fontweight='bold')
                ax3.scatter(viz_origins[:, 0], viz_origins[:, 2],
                           c=ray_colors, marker='o', s=10, alpha=0.5, edgecolors='black', linewidths=0.3)
                ax3.scatter(viz_endpoints[:, 0], viz_endpoints[:, 2],
                           c=ray_colors, marker='.', s=5, alpha=0.3)
                ax3.scatter(all_poses_arr[:, 0], all_poses_arr[:, 2],
                           c=pose_colors, marker='^', s=100, alpha=0.9, edgecolors='black',
                           linewidths=2)
                ax3.set_xlabel('X (world)', fontsize=12)
                ax3.set_ylabel('Z (world)', fontsize=12)
                ax3.grid(True, alpha=0.3)
                ax3.set_aspect('equal')
                
                # YZ projection
                ax4 = fig.add_subplot(224)
                ax4.set_title(f'Epoch {current_epoch}: YZ Projection (Side View) - {n_volumes} Volumes', fontsize=14, fontweight='bold')
                ax4.scatter(viz_origins[:, 1], viz_origins[:, 2],
                           c=ray_colors, marker='o', s=10, alpha=0.5, edgecolors='black', linewidths=0.3)
                ax4.scatter(viz_endpoints[:, 1], viz_endpoints[:, 2],
                           c=ray_colors, marker='.', s=5, alpha=0.3)
                ax4.scatter(all_poses_arr[:, 1], all_poses_arr[:, 2],
                           c=pose_colors, marker='^', s=100, alpha=0.9, edgecolors='black',
                           linewidths=2)
                ax4.set_xlabel('Y (world)', fontsize=12)
                ax4.set_ylabel('Z (world)', fontsize=12)
                ax4.grid(True, alpha=0.3)
                ax4.set_aspect('equal')
                
                plt.tight_layout()
                epoch_viz_file = os.path.join(epoch_viz_path, f'epoch_{current_epoch:04d}_iter_{i}.png')
                plt.savefig(epoch_viz_file, dpi=150, bbox_inches='tight')
                plt.close()
                
                # Save statistics
                stats_file = os.path.join(epoch_viz_path, f'epoch_{current_epoch:04d}_stats.txt')
                with open(stats_file, 'w') as f:
                    f.write(f"Epoch {current_epoch} Statistics (Iteration {i})\n")
                    f.write("="*60 + "\n\n")
                    f.write(f"Total samples: {len(epoch_ray_origins)}\n")
                    f.write(f"Total rays: {all_origins.shape[0]}\n")
                    f.write(f"Total poses: {len(all_poses_arr)}\n")
                    f.write(f"Number of volumes: {n_volumes}\n")
                    f.write(f"Volume IDs: {sorted(unique_volumes.tolist())}\n\n")
                    
                    f.write("Ray Origins Range:\n")
                    f.write(f"  X: [{all_origins[:, 0].min():.6f}, {all_origins[:, 0].max():.6f}]\n")
                    f.write(f"  Y: [{all_origins[:, 1].min():.6f}, {all_origins[:, 1].max():.6f}]\n")
                    f.write(f"  Z: [{all_origins[:, 2].min():.6f}, {all_origins[:, 2].max():.6f}]\n\n")
                    
                    f.write("Ray Endpoints Range:\n")
                    f.write(f"  X: [{all_endpoints[:, 0].min():.6f}, {all_endpoints[:, 0].max():.6f}]\n")
                    f.write(f"  Y: [{all_endpoints[:, 1].min():.6f}, {all_endpoints[:, 1].max():.6f}]\n")
                    f.write(f"  Z: [{all_endpoints[:, 2].min():.6f}, {all_endpoints[:, 2].max():.6f}]\n\n")
                    
                    f.write("Poses Range:\n")
                    f.write(f"  X: [{all_poses_arr[:, 0].min():.6f}, {all_poses_arr[:, 0].max():.6f}]\n")
                    f.write(f"  Y: [{all_poses_arr[:, 1].min():.6f}, {all_poses_arr[:, 1].max():.6f}]\n")
                    f.write(f"  Z: [{all_poses_arr[:, 2].min():.6f}, {all_poses_arr[:, 2].max():.6f}]\n\n")
                    
                    # Per-volume statistics
                    f.write("Per-Volume Statistics:\n")
                    for vol_id in unique_volumes:
                        vol_mask = all_volume_ids == vol_id
                        vol_origins = all_origins[vol_mask]
                        f.write(f"\n  Volume {vol_id}:\n")
                        f.write(f"    Rays: {vol_origins.shape[0]}\n")
                        f.write(f"    X range: [{vol_origins[:, 0].min():.6f}, {vol_origins[:, 0].max():.6f}]\n")
                        f.write(f"    Y range: [{vol_origins[:, 1].min():.6f}, {vol_origins[:, 1].max():.6f}]\n")
                        f.write(f"    Z range: [{vol_origins[:, 2].min():.6f}, {vol_origins[:, 2].max():.6f}]\n")
                
                print(f"Saved epoch visualization to: {epoch_viz_file}")
                print(f"Saved epoch statistics to: {stats_file}")
                print(f"  Total rays visualized: {all_origins.shape[0]}")
                print(f"  Total poses: {len(all_poses_arr)}")
                print(f"  Number of volumes: {n_volumes}")
                print()
                
                # Clear epoch data and increment counter
                epoch_ray_origins = []
                epoch_ray_endpoints = []
                epoch_poses = []
                epoch_pose_volume_ids = []
                epoch_iteration = []
                epoch_volume_ids = []
                #current_epoch += 1
            
            train_idx_counter = 0
            np.random.shuffle(train_indices_shuffled)
            current_epoch += 1
            print(f"Completed epoch {current_epoch} at iteration {i-1}, reshuffling training data")
            
        # Use sequential sampling from shuffled list to ensure all images/volumes are used
        img_i = train_indices_shuffled[train_idx_counter]
        train_idx_counter += 1        
        print("Rendering volume/image", img_i, "of", len(i_train))
        
        if not args.full_volume_mode:
            target = torch.Tensor(get_target(images, img_i, pts_image)).to(device).unsqueeze(0).unsqueeze(0)
            pose = torch.from_numpy(poses[img_i, :3, :4]).to(device).unsqueeze(0)
        else:
            # Apply get_target() slice-by-slice with pts_image_volume
            depth = images.shape[1]
            slices = []
            for d in range(depth):
                pts_slice = pts_image_volume[d] 
                slices.append(get_target(images[img_i], d, pts_slice))
            
            target_volume = np.stack(slices, axis=0)  # shape: [D, ...]
            target = torch.Tensor(target_volume).to(device).unsqueeze(0)

            pose = torch.from_numpy(poses[img_i, ..., :3, :4]).to(device).unsqueeze(0)
            
            # Inline visualization of poses and ray directions for debugging
            if args.debug_viz_render and i == start:
                print("Creating inline pose and ray visualization...")
                
                # Get the current pose (convert back to numpy for visualization)
                current_pose = pose[0].cpu().numpy()  # Shape: [D, 3, 4] or [3, 4]
                
                # Handle both 2D and 3D pose formats
                if current_pose.ndim == 3:
                    # 3D volume mode: [D, 3, 4]
                    n_slices = current_pose.shape[0]
                    sample_indices = [0, n_slices // 2, n_slices - 1] if n_slices >= 3 else list(range(n_slices))
                    #sample_indices = list(range(n_slices))
                    poses_to_viz = current_pose[sample_indices]
                else:
                    # 2D mode: [3, 4]
                    poses_to_viz = current_pose[np.newaxis, ...]
                
                fig = plt.figure(figsize=(15, 5))
                
                # 3D visualization
                ax1 = fig.add_subplot(131, projection='3d')
                ax1.set_title('Poses and Ray Origins (3D)')
                ax1.set_xlabel('X')
                ax1.set_ylabel('Y')
                ax1.set_zlabel('Z')
                
                colors = plt.cm.tab10(np.linspace(0, 1, len(poses_to_viz)))
                
                for idx, (pose_mat, color) in enumerate(zip(poses_to_viz, colors)):
                    # Extract camera position (translation)
                    cam_pos = pose_mat[:3, 3]
                    
                    # Extract camera orientation (rotation matrix columns = axes)
                    right = pose_mat[:3, 0]  # X-axis (right)
                    up = pose_mat[:3, 1]     # Y-axis (up)
                    forward = pose_mat[:3, 2] # Z-axis (forward/viewing direction)
                    
                    # Plot camera position
                    ax1.scatter(*cam_pos, c=[color], s=100, marker='o', label=f'Pose {idx}')
                    
                    # Plot camera axes
                    axis_length = 0.01  # 1cm
                    ax1.quiver(*cam_pos, *right, length=axis_length, color='r', alpha=0.6, arrow_length_ratio=0.3)
                    ax1.quiver(*cam_pos, *up, length=axis_length, color='g', alpha=0.6, arrow_length_ratio=0.3)
                    ax1.quiver(*cam_pos, *forward, length=axis_length, color='b', alpha=0.6, arrow_length_ratio=0.3)
                
                ax1.legend()
                ax1.set_box_aspect([1,1,1])
                
                # XY projection
                ax2 = fig.add_subplot(132)
                ax2.set_title('XY Projection')
                ax2.set_xlabel('X')
                ax2.set_ylabel('Y')
                ax2.grid(True, alpha=0.3)
                
                for idx, (pose_mat, color) in enumerate(zip(poses_to_viz, colors)):
                    cam_pos = pose_mat[:3, 3]
                    forward = pose_mat[:3, 2]
                    
                    ax2.scatter(cam_pos[0], cam_pos[1], c=[color], s=100, marker='o', label=f'Pose {idx}')
                    ax2.arrow(cam_pos[0], cam_pos[1], forward[0]*0.01, forward[1]*0.01, 
                             head_width=0.002, head_length=0.002, fc=color, ec=color, alpha=0.6)
                
                ax2.legend()
                ax2.set_aspect('equal')
                
                # XZ projection
                ax3 = fig.add_subplot(133)
                ax3.set_title('XZ Projection')
                ax3.set_xlabel('X')
                ax3.set_ylabel('Z')
                ax3.grid(True, alpha=0.3)
                
                for idx, (pose_mat, color) in enumerate(zip(poses_to_viz, colors)):
                    cam_pos = pose_mat[:3, 3]
                    forward = pose_mat[:3, 2]
                    
                    ax3.scatter(cam_pos[0], cam_pos[2], c=[color], s=100, marker='o', label=f'Pose {idx}')
                    ax3.arrow(cam_pos[0], cam_pos[2], forward[0]*0.01, forward[2]*0.01,
                             head_width=0.002, head_length=0.002, fc=color, ec=color, alpha=0.6)
                
                ax3.legend()
                ax3.set_aspect('equal')
                
                plt.tight_layout()
                viz_path = os.path.join(basedir, expname, f"inline_pose_viz_iter_{i}_img_{img_i}.png")
                plt.savefig(viz_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Saved inline pose visualization to: {viz_path}")
                
                # Disable for subsequent iterations to avoid slowdown
                args.debug_viz_render = False  
        
        #####  Core optimization loop  #####
        if convex.config.get('use_radial_path_mode', False):
            # Radial path (tree-ring) rendering
            from rendering import render_radial_paths
            
            rendering_output = render_radial_paths(
                origins=convex.config['_radial_origins'],
                directions=convex.config['_radial_directions'],
                metadata=convex.config['_radial_metadata'],
                paths=convex.config['_radial_paths'],
                path_metadata=convex.config['_radial_path_metadata'],
                pose=pose.squeeze(0).cpu().numpy(),#.squeeze(0).cpu().numpy(),  # Remove batch dim and convert to numpy
                network_fn=render_kwargs_train['network_fn'],
                network_query_fn=render_kwargs_train['network_query_fn'],
                chunk=args.chunk
            )
        else:
            # Fallback to standard rendering
            rendering_output = render_us(H, W, sw, sh, c2w=pose, chunk=args.chunk, retraw=True, **render_kwargs_train)
        
        # Collect ray information for epoch visualization
        if 'ray_origins' in rendering_output and 'ray_directions' in rendering_output and current_epoch == 0:
            # Get ray origins and directions
            ray_origins = rendering_output['ray_origins'].detach().cpu().numpy()
            ray_directions = rendering_output['ray_directions'].detach().cpu().numpy()
            
            # Subsample rays to avoid memory issues (keep every Nth ray)
            subsample_factor = max(1, ray_origins.shape[0] // 1000)  # Keep ~1000 rays per iteration
            ray_origins_sub = ray_origins[::subsample_factor]
            ray_directions_sub = ray_directions[::subsample_factor]
            
            # Calculate ray endpoints (origins + far * directions)
            far_distance = far #55.0 * 0.001  # Same as render_us far parameter
            ray_endpoints = ray_origins_sub + far_distance * ray_directions_sub
            
            # Store for epoch visualization
            epoch_ray_origins.append(ray_origins_sub)
            epoch_ray_endpoints.append(ray_endpoints)
            
            # Track which volume/image this belongs to
            volume_ids = np.full(ray_origins_sub.shape[0], img_i, dtype=np.int32)
            epoch_volume_ids.append(volume_ids)
            
            # Extract pose translation (camera position)
            if args.full_volume_mode:
                # For 3D volumes, take the center slice pose
                pose_np = pose[0].detach().cpu().numpy()  # [D, 3, 4]
                center_slice = pose_np.shape[0] // 2
                pose_translation = pose_np[center_slice, :3, 3]
            else:
                # For 2D slices
                pose_np = pose[0].detach().cpu().numpy()  # [3, 4]
                pose_translation = pose_np[:3, 3]
            
            epoch_poses.append(pose_translation)
            epoch_pose_volume_ids.append(img_i)  # Track which volume this pose belongs to
            epoch_iteration.append(i)
            
        
        output_image = rendering_output["intensity_map"]
        
        optimizer.zero_grad()
        loss = compute_loss(output_image, target, args, losses, i)

        #print("Loss keys:", loss.keys())
        #print("Loss values:", {k: v[1].item() for k, v in loss.items()})
        
        if args.reg and i > args.r_warm_up_it:
            #print("Computing regularization...")
            reg = compute_regularization(rendering_output, losses,
                                      weights=(args.r_lcc_penalty, args.r_tv_penalty, args.r_max_reflection),
                                      reflection_smoothness_weight=args.reflection_smoothness_weight)
            #print("Regularization keys:", reg.keys())
            #print("Regularization values:", {k: v[1].item() for k, v in reg.items()})
            loss = {**loss, **reg}

        total_loss = 0.0
        for loss_value in loss.values():
            tmp = loss_value[0] * loss_value[1]
            total_loss += tmp
        
        #print("Total loss:", total_loss.item())
        
        if type(total_loss) != torch.Tensor:
            raise ValueError("Loss is not a tensor: Problem with loss calculation")

        total_loss.backward()
        optimizer.step()

        dt = time.time() - time0

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (i / decay_steps))
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lrate
        ################################

        if args.tensorboard:
            writer.add_scalar("Loss/total_loss", total_loss.item(), i)
            for k, v in loss.items():
                writer.add_scalar(f"Loss/{k}", v[1].item(), i)
            writer.add_scalar("Learning rate", new_lrate, i)

        dt = time.time() - time0
        total_time_elapsed += dt
        
        if args.tensorboard:
            writer.add_scalar("Time/iteration_time", dt, i)
            writer.add_scalar("Time/total_elapsed", total_time_elapsed, i)
            
        if (i + 1) % args.i_print == 0:

            rendering_path = os.path.join(basedir, expname, "train_rendering")
            os.makedirs(rendering_path, exist_ok=True)

            print(f"Step: {i+1}, Loss: {total_loss.item()}, Time: {dt:.4f}s, Total Time: {total_time_elapsed/60:.2f}min")
            detailed_loss_string = ", ".join(
                [f"{k}: {v[1].item()}" for k, v in loss.items()]
            )
            print(detailed_loss_string)
            print("Rendering output keys:", rendering_output.keys())
            print("Rendering output shapes:", {k: v.shape for k, v in rendering_output.items()})
            
            if not convex.config['full_volume_mode']:
                # 2D slice visualization
                visualize_2d_slice(rendering_output, target, pts_image, W, H, 
                                  rendering_path, img_i, i + 1, mode="train")
                
                # Save comparison images with different colormaps
                images_path = os.path.join(basedir, expname, "images_train_rendering")
                os.makedirs(images_path, exist_ok=True)
                visualize_2d_slice_comparison(output_image, target, pts_image, W, H,
                                             images_path, img_i, i + 1, mode="train")
            else:
                # 3D volume visualization
                visualize_3d_volume_slices(rendering_output, target, output_image, pts_image, 
                                          W, H, D, images.shape[-3], rendering_path, 
                                          img_i, i + 1, mode="train", n_slices=3,
                                          nerf_original_vis=True, save_volume=args.save_volume)

        ################################ EVALUATE ################################
    
            # Set models to evaluation mode
            render_kwargs_test["network_fn"].eval()
                        
            with torch.no_grad():
                # Select random validation image
                img_i = np.random.choice(i_val)
                
                if not convex.config['full_volume_mode']:
                    target = torch.Tensor(get_target(images, img_i, pts_image)).to(device).unsqueeze(0).unsqueeze(0)
                    pose = torch.from_numpy(poses[img_i, :3, :4]).to(device).unsqueeze(0)
                else:
                    slices = []
                    for d in range(D):
                        pts_slice = pts_image_volume[d]
                        slices.append(get_target(images[img_i], d, pts_slice))
                
                    target_volume = np.stack(slices, axis=0)  # shape: [D, ...]
                    target = torch.Tensor(target_volume).to(device).unsqueeze(0)
                    pose = torch.from_numpy(poses[img_i, ..., :3, :4]).to(device).unsqueeze(0)                    
                
                # Render using test parameters
                if convex.config.get('use_radial_path_mode', False):
                    # Radial path rendering for validation
                    from rendering import render_radial_paths
                    
                    rendering_output = render_radial_paths(
                        origins=convex.config['_radial_origins'],
                        directions=convex.config['_radial_directions'],
                        metadata=convex.config['_radial_metadata'],
                        paths=convex.config['_radial_paths'],
                        path_metadata=convex.config['_radial_path_metadata'],
                        pose=pose.squeeze(0).cpu().numpy(),
                        network_fn=render_kwargs_test['network_fn'],
                        network_query_fn=render_kwargs_test['network_query_fn'],
                        chunk=args.chunk
                    )
                else:
                    rendering_output = render_us(H, W, sw, sh, c2w=pose, chunk=args.chunk, retraw=True, **render_kwargs_test)
                
                print("Rendering output keys:", rendering_output.keys())
                for k in rendering_output:
                    print(f"{k} shape:", rendering_output[k].shape)
                    
            output_image = rendering_output['intensity_map']

            rendering_path = os.path.join(basedir, expname, "eval_rendering")
            os.makedirs(rendering_path, exist_ok=True)

            print(f"Step: {i+1}, Val Image: {img_i}, Time: {dt}")

            if not convex.config['full_volume_mode']:
                # 2D slice visualization
                visualize_2d_slice(rendering_output, target, pts_image, W, H,
                                    rendering_path, img_i, i + 1, mode="eval")
                
                # Save comparison images with different colormaps
                images_path = os.path.join(basedir, expname, "images_eval_rendering")
                os.makedirs(images_path, exist_ok=True)
                visualize_2d_slice_comparison(output_image, target, pts_image, W, H,
                                                images_path, img_i, i + 1, mode="eval")
            else:
                # 3D volume visualization
                visualize_3d_volume_slices(rendering_output, target, output_image, pts_image,
                                            W, H, D, images.shape[-3], rendering_path,
                                            img_i, i + 1, mode="eval", n_slices=3,
                                            nerf_original_vis=True, save_volume=args.save_volume)

            # Set models back to training mode
            render_kwargs_test["network_fn"].train()
            
        ################################ END OF EVALUATE ################################

        if (i + 1) % args.i_weights == 0:
            path = os.path.join(basedir, expname, "{:06d}.tar".format(i + 1))
            torch.save(
                {
                    "global_step": i,
                    "network_fn_state_dict": render_kwargs_train[
                        "network_fn"
                    ].state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                path,
            )
            # print('Saved checkpoints at', path)
    
    # Training complete - print final timing summary
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    avg_iter_time = total_time_elapsed / (N_iters - start + 1) if (N_iters - start + 1) > 0 else 0
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total training time: {total_training_time/3600:.2f} hours ({total_training_time/60:.2f} minutes)")
    print(f"Total iterations: {N_iters - start + 1}")
    print(f"Average time per iteration: {avg_iter_time:.4f}s")
    print(f"Experiment directory: {os.path.join(basedir, expname)}")
    print("="*60 + "\n")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.set_default_device(device)
    train()
