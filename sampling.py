"""
Module containing helper functions and classes for sampling, including:
- Coordinate transformations for poses and rays
- Ray generation for different sampling strategies (e.g., uniform convex, spherical, concentric circles)
    - standard uniform convex fan sampling (define_rays_numpy)
- Interpolation of missing pixel values in images
- Target remapping for convex sampling
- Caching
- TODO: Implementation for spherical and concentric circle sampling strategies are done but need to be further modified and logic integrated into the main training and evaluation pipeline before they can be used. 
    Rendering needs to be done for these strategies, especially for the concentric circle sampling.
"""

import io
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F

import convex_configuration as convex
from torch.distributions import MultivariateNormal

from scipy import interpolate

from PIL import Image
import math
from scipy.interpolate import griddata, NearestNDInterpolator, LinearNDInterpolator, RegularGridInterpolator


def transform_poses_coordinate_system(poses, axis_mapping='z_to_x'):
    """
    Transform poses to a different coordinate system by applying rotation.
    
    This is useful when you need to change which axis represents depth/forward direction.
    For ultrasound, you typically want depth along a specific axis (e.g., Z-axis).
    
    Args:
        poses: numpy array of poses
            - Shape [N, 3, 4] for 2D mode (N poses)
            - Shape [N, D, 3, 4] for 3D volume mode (N volumes, D slices each)
        axis_mapping: str, predefined rotation to apply
            - 'z_to_x': Rotate so Z-axis becomes X-axis (depth along X)
                       Useful when cameras point along Z but you want depth along X
                       Rotation: X'=Z, Y'=Y, Z'=-X
            - 'y_to_x': Rotate so Y-axis becomes X-axis (depth along X)
                       Rotation: X'=Y, Y'=-X, Z'=Z
            - 'x_to_z': Rotate so X-axis becomes Z-axis (depth along Z)
                       Rotation: X'=-Z, Y'=Y, Z'=X
            - 'identity': No rotation (for testing)
            - numpy array [3, 3]: Custom rotation matrix
    
    Returns:
        transformed_poses: numpy array with same shape as input
    """
    
    # Define rotation matrices for common transformations
    rotation_matrices = {
        'z_to_x': np.array([
            [0, 0, 1],   # New X = old Z
            [0, 1, 0],   # New Y = old Y  
            [-1, 0, 0]   # New Z = -old X
        ], dtype=np.float32),
        
        'y_to_x': np.array([
            [0, 1, 0],   # New X = old Y
            [-1, 0, 0],  # New Y = -old X
            [0, 0, 1]    # New Z = old Z
        ], dtype=np.float32),
        
        'x_to_z': np.array([
            [0, 0, -1],  # New X = -old Z
            [0, 1, 0],   # New Y = old Y
            [1, 0, 0]    # New Z = old X
        ], dtype=np.float32),
        
        'identity': np.eye(3, dtype=np.float32)
    }
    
    # Get rotation matrix
    if isinstance(axis_mapping, str):
        if axis_mapping not in rotation_matrices:
            raise ValueError(f"Unknown axis_mapping '{axis_mapping}'. "
                           f"Use one of {list(rotation_matrices.keys())} or provide 3x3 numpy array")
        R_transform = rotation_matrices[axis_mapping]
    elif isinstance(axis_mapping, np.ndarray):
        if axis_mapping.shape != (3, 3):
            raise ValueError(f"Custom rotation matrix must be 3x3, got {axis_mapping.shape}")
        R_transform = axis_mapping.astype(np.float32)
    else:
        raise ValueError(f"axis_mapping must be str or numpy array, got {type(axis_mapping)}")
    
    # Create 4x4 transformation matrix
    T_transform = np.eye(4, dtype=np.float32)
    T_transform[:3, :3] = R_transform
    
    # Apply transformation based on pose shape
    transformed_poses = poses.copy()
    
    if poses.ndim == 3:  # 2D mode: [N, 3, 4]
        N = poses.shape[0]
        for i in range(N):
            # Convert [3, 4] to [4, 4]
            pose_4x4 = np.vstack([poses[i], [0, 0, 0, 1]])
            # Apply transformation: T_new = T_transform @ T_old
            transformed_4x4 = T_transform @ pose_4x4
            # Convert back to [3, 4]
            transformed_poses[i] = transformed_4x4[:3, :4]
            
    elif poses.ndim == 4:  # 3D volume mode: [N, D, 3, 4]
        N, D = poses.shape[0], poses.shape[1]
        for i in range(N):
            for d in range(D):
                # Convert [3, 4] to [4, 4]
                pose_4x4 = np.vstack([poses[i, d], [0, 0, 0, 1]])
                # Apply transformation
                transformed_4x4 = T_transform @ pose_4x4
                # Convert back to [3, 4]
                transformed_poses[i, d] = transformed_4x4[:3, :4]
    else:
        raise ValueError(f"Poses must have shape [N, 3, 4] or [N, D, 3, 4], got {poses.shape}")
    
    print(f"Applied pose transformation: {axis_mapping}")
    print(f"Rotation matrix:\n{R_transform}")
    
    return transformed_poses 

def interpolate_missing_pixels(
        image: np.ndarray,
        mask: np.ndarray,
        method: str = 'nearest',
        fill_value: int = 0,
        _interpolate = 1
):
    """
    Interpolate missing pixels in an image using various methods.
    
    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """

    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]
    
    if _interpolate:
        interp_values = interpolate.griddata(
            (known_x, known_y), known_v, (missing_x, missing_y),
            method=method, fill_value=fill_value
        )

        interp_image = image.copy()
        interp_image[missing_y, missing_x] = interp_values

    else:
        print("Skipping interpolation, returning original image with NaNs")
        interp_image = image.copy()

    return interp_image


# Global variables for ray definitions and coordinate transformations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
__a_mask = None
__delta_radius = None
__origin = None
__direction = None
__origin_val = None
__direction_val = None
__rays_o_image = None
__rays_d_image = None

# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

class CoordinateTransformer:
    def __init__(self, image_width, image_height):
        self.image_width = image_width
        self.image_height = image_height

    # Map image coordinates (normalized) to pixel coordinates for an array
    def image_to_pixel(self, coords_image):
        
        if torch.is_tensor(coords_image):
            coords_image = coords_image.detach().cpu().numpy()
            
        def norm(x):
            return (x - np.min(x)) / (np.max(x) - np.min(x))

        # Assuming coords_image is a numpy array of shape (N, 2)
        # where N is the number of coordinates, and each coordinate is in the form (x_image, y_image)
        x_norm = norm(coords_image[..., 0])
        y_norm = norm(coords_image[..., 1])
        # coords_image[..., 0] = np.clip(norm(coords_image[..., 0]), 0., 1.)
        # coords_image[..., 1] = np.clip(norm(coords_image[..., 1]), 0., 1.)
        x_pixel = x_norm * self.image_width
        y_pixel = y_norm * self.image_height
        return np.stack((x_pixel, y_pixel), axis=-1).astype(int)

    # Map pixel coordinates to image coordinates (normalized) for an array
    def pixel_to_image(self, coords_pixel):
        # Assuming coords_pixel is a numpy array of shape (N, 2)
        # where N is the number of coordinates, and each coordinate is in the form (x_pixel, y_pixel)
        x_image = coords_pixel[..., 0] / self.image_width
        y_image = coords_pixel[..., 1] / self.image_height
        return np.stack((x_image, y_image), axis=-1)

def define_rays_numpy(num_rays, inner_radius, opening_angle, center=None):
    """
    Function to plot rays defined by number of rays, inner radius, outer radius,
    and the opening angle. The function returns the ray origins and directions after
    rotating the rays by -90 degrees.

    Args:
        num_rays (int): Number of rays.
        inner_radius (float): Inner radius of the fan shape.
        outer_radius (float): Outer radius of the fan shape.
        opening_angle (float): Opening angle of the fan in degrees.
        center (tuple): The center of the fan. Defaults to the center of the plot.

    Returns:
        origins (numpy.ndarray): Ray origins for each ray.
        directions (numpy.ndarray): Directions of the rays after rotation.
    """
    if center is None:
        center = (0, 0)  # Default center at origin (0, 0)

    # Convert opening angle to radians
    opening_angle_rad = np.deg2rad(opening_angle)

    # Create an array of angles for the rays
    angle_step = opening_angle_rad / (num_rays - 1)
    angles = np.linspace(-opening_angle_rad / 2, opening_angle_rad / 2, num_rays)
    origins = []
    directions = []
    a = np.deg2rad(90)
    # Rotation matrix for -90 degrees
    rotation_matrix = np.array([[np.cos(a), -np.sin(a)],
                                [np.sin(a), np.cos(a)]])  # For rotating by -90 degrees
    # Plot rays from inner radius to outer radius
    for angle in angles:
        # Directions (unit vectors) for the rays
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)

        # Apply the -90 degree rotation (counterclockwise)
        rotated_direction = rotation_matrix @ direction

        directions.append(rotated_direction)

        # Origins at the inner radius and outer radius
        inner_origin = center + inner_radius * rotated_direction
        # outer_origin = center + outer_radius * rotated_direction
        origins.append(inner_origin)

    origins, directions = np.array(origins, dtype=np.float32), np.array(directions, dtype=np.float32)

    y_min = np.min(origins[..., 1])
    origins[..., 1] -= y_min

    return origins, directions


def define_rays_spherical_numpy(n_rays_azimuth, n_rays_elevation, inner_radius,
                                 opening_angle_azimuth, opening_angle_elevation,
                                 center=None):
    """
    Generate 3D fan-beam rays in spherical coordinates (r, θ, φ) for volumetric
    ultrasound imaging with a convex probe.

    Instead of stacking D independent 2D fan slices with separate poses, this
    function creates a single 3D cone of diverging rays by sweeping the 2D
    azimuth fan through an elevation angle φ. This is physically correct for
    rotational ICE probes and electronically-steered 2D arrays.

    Coordinate convention (after 90° rotation applied):
        - Depth (radial) points along +y
        - Lateral (azimuth θ) spreads along x
        - Elevation (φ) spreads along z

    Geometry:
        For azimuth angle θ_k and elevation angle φ_l, the 3D direction is:
            d = R_90 @ [cos(θ) cos(φ), sin(θ) cos(φ), sin(φ)]^T

        After the 90° rotation (matching define_rays_numpy convention):
            d' = [-sin(θ) cos(φ), cos(θ) cos(φ), sin(φ)]^T

        Origins sit on the inner arc at radius r_inner:
            o = center + r_inner * d'

    Resolution properties:
        At depth r from the apex, the spacing between adjacent samples is:
        - Lateral:    Δx ≈ r · Δθ · cos(φ)    (shrinks toward poles)
        - Elevation:  Δz ≈ r · Δφ
        - Radial:     Δr = (r_outer - r_inner) / N_samples (uniform)

        Both lateral AND elevation resolution degrade with depth, unlike the
        multi-planar approach where elevation resolution depends only on the
        Euclidean inter-slice spacing.

    Args:
        n_rays_azimuth (int): Number of rays in the azimuth (lateral) direction.
            Corresponds to convex_n_rays in the 2D fan case.
        n_rays_elevation (int): Number of rays in the elevation direction.
            Corresponds to D (number of depth slices) in multi-planar mode.
        inner_radius (float): Normalized inner radius of the fan
            (typically radius2/radius, e.g., 217/860 ≈ 0.252).
        opening_angle_azimuth (float): Azimuth opening angle in degrees
            (same as convex_angle, e.g., 70°).
        opening_angle_elevation (float): Elevation opening angle in degrees.
            For ICE, typically 10-30° depending on the probe.
        center (tuple or None): 3D center point (apex) of the fan.
            Defaults to (0, 0, 0).

    Returns:
        origins (np.ndarray): Ray origins, shape [N_azimuth * N_elevation, 3].
            Points on the inner spherical surface.
        directions (np.ndarray): Unit direction vectors, shape [N_azimuth * N_elevation, 3].
            Diverging outward from the apex.
        metadata (dict): Information about the ray structure:
            - 'n_rays_azimuth': Number of azimuth rays
            - 'n_rays_elevation': Number of elevation rays
            - 'total_rays': Total number of rays (azimuth × elevation)
            - 'azimuth_angles': 1D array of azimuth angles [N_azimuth]
            - 'elevation_angles': 1D array of elevation angles [N_elevation]
            - 'elevation_indices': Array mapping each ray to its elevation index
            - 'azimuth_indices': Array mapping each ray to its azimuth index
            - 'opening_angle_azimuth': Azimuth opening angle (degrees)
            - 'opening_angle_elevation': Elevation opening angle (degrees)
            - 'strategy': 'spherical'

    Example:
        >>> origins, directions, meta = define_rays_spherical_numpy(
        ...     n_rays_azimuth=100, n_rays_elevation=68,
        ...     inner_radius=0.252, opening_angle_azimuth=70,
        ...     opening_angle_elevation=20
        ... )
        >>> origins.shape   # (6800, 3)
        >>> meta['total_rays']  # 6800
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    else:
        center = np.array(center, dtype=np.float32)

    # Convert angles to radians
    theta_open = np.deg2rad(opening_angle_azimuth)
    phi_open = np.deg2rad(opening_angle_elevation)

    # Generate angle arrays
    # Azimuth: symmetric about 0, same as define_rays_numpy
    azimuth_angles = np.linspace(-theta_open / 2, theta_open / 2,
                                 n_rays_azimuth, dtype=np.float32)
    # Elevation: symmetric about 0 (centered fan)
    elevation_angles = np.linspace(-phi_open / 2, phi_open / 2,
                                   n_rays_elevation, dtype=np.float32)

    # 90° rotation matrix (same convention as define_rays_numpy)
    # This rotates the fan so depth points along +y instead of +x
    a = np.deg2rad(90)
    cos_a, sin_a = np.cos(a), np.sin(a)
    # 3D rotation matrix: rotate around z-axis by 90°
    #   [cos -sin  0]     [0 -1 0]
    #   [sin  cos  0]  =  [1  0 0]
    #   [ 0    0   1]     [0  0 1]
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ], dtype=np.float32)

    # Pre-allocate arrays
    total_rays = n_rays_azimuth * n_rays_elevation
    all_origins = np.empty((total_rays, 3), dtype=np.float32)
    all_directions = np.empty((total_rays, 3), dtype=np.float32)
    elevation_indices = np.empty(total_rays, dtype=np.int32)
    azimuth_indices = np.empty(total_rays, dtype=np.int32)

    # Generate rays: outer loop elevation, inner loop azimuth
    # This ordering means rays for slice l are contiguous:
    #   indices [l*N_azimuth : (l+1)*N_azimuth]
    idx = 0
    for l, phi in enumerate(elevation_angles):
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        for k, theta in enumerate(azimuth_angles):
            # 3D direction before rotation:
            #   d = [cos(θ)cos(φ), sin(θ)cos(φ), sin(φ)]
            direction = np.array([
                np.cos(theta) * cos_phi,
                np.sin(theta) * cos_phi,
                sin_phi
            ], dtype=np.float32)

            # Apply 90° z-rotation to match existing coordinate convention
            rotated_dir = rotation_matrix @ direction
            # Normalize (should already be unit, but ensure numerical precision)
            rotated_dir /= np.linalg.norm(rotated_dir) + 1e-12

            # Origin on the inner spherical surface
            origin = center + inner_radius * rotated_dir

            all_origins[idx] = origin
            all_directions[idx] = rotated_dir
            elevation_indices[idx] = l
            azimuth_indices[idx] = k
            idx += 1

    # Shift y so minimum is at 0 (same convention as define_rays_numpy)
    y_min = np.min(all_origins[:, 1])
    all_origins[:, 1] -= y_min

    metadata = {
        'n_rays_azimuth': n_rays_azimuth,
        'n_rays_elevation': n_rays_elevation,
        'total_rays': total_rays,
        'azimuth_angles': azimuth_angles,
        'elevation_angles': elevation_angles,
        'elevation_indices': elevation_indices,
        'azimuth_indices': azimuth_indices,
        'opening_angle_azimuth': opening_angle_azimuth,
        'opening_angle_elevation': opening_angle_elevation,
        'inner_radius': inner_radius,
        'y_shift': y_min,
        'original_center': center.copy(),
        'strategy': 'spherical',
    }

    return all_origins, all_directions, metadata


def define_rays_concentric_circles(n_circles, inner_radius, outer_radius, 
                                   base_rays_per_circle, ray_density_multiplier, 
                                   density_growth_type='exponential', center=None):
    """
    Generate rays in concentric circles with increasing density (like tree rings).
    
    This function creates a radial sampling pattern where:
    - Rays originate from points on concentric circles
    - Each successive circle has more rays (density increases)
    - Full 360° coverage on each circle
    - Matches the existing coordinate system (90° rotation applied)
    
    Args:
        n_circles (int): Number of concentric circles (e.g., 5)
        inner_radius (float): Radius of innermost circle in pixels (e.g., 50.0)
        outer_radius (float): Radius of outermost circle in pixels (e.g., 900.0)
        base_rays_per_circle (int): Number of rays in innermost circle (e.g., 8)
        ray_density_multiplier (float): Controls ray growth rate
        density_growth_type (str): Type of density growth:
            - 'exponential': Circle k has floor(base_rays * multiplier^k) rays
            - 'linear': Circle k has floor(base_rays + k * multiplier) rays
            - 'quadratic': Circle k has floor(base_rays + k^2 * multiplier) rays
        center (tuple): Center point of circles, default (0, 0)
        
    Returns:
        origins (np.ndarray): Ray origin points, shape [N_total, 2]
        directions (np.ndarray): Ray direction unit vectors, shape [N_total, 2]
        metadata (dict): Information about the ray structure:
            - 'n_circles': Number of circles
            - 'rays_per_circle': List of ray counts per circle
            - 'total_rays': Total number of rays
            - 'circle_indices': Array indicating which circle each ray belongs to
            - 'circle_radii': Array of circle radii
            - 'density_growth_type': Growth type used
    
    Examples:
        >>> # Exponential growth: 4, 8, 16
        >>> origins, directions, metadata = define_rays_concentric_circles(
        ...     n_circles=3, inner_radius=100.0, outer_radius=500.0,
        ...     base_rays=4, ray_multiplier=2.0, density_growth_type='exponential'
        ... )
        
        >>> # Linear growth: 8, 12, 16
        >>> origins, directions, metadata = define_rays_concentric_circles(
        ...     n_circles=3, inner_radius=100.0, outer_radius=500.0,
        ...     base_rays=8, ray_multiplier=4.0, density_growth_type='linear'
        ... )
        
        >>> # Quadratic growth: 8, 12, 20
        >>> origins, directions, metadata = define_rays_concentric_circles(
        ...     n_circles=3, inner_radius=100.0, outer_radius=500.0,
        ...     base_rays=8, ray_multiplier=3.0, density_growth_type='quadratic'
        ... )
    """
    
    if center is None:
        center = np.array([0.0, 0.0], dtype=np.float32)
    else:
        center = np.array(center, dtype=np.float32)
    
    # Step 1: Calculate radii for each circle (linearly spaced)
    radii = np.linspace(inner_radius, outer_radius, n_circles, dtype=np.float32)
    
    # Step 2: Calculate number of rays for each circle based on growth type
    rays_per_circle = []
    for k in range(n_circles):
        if density_growth_type == 'exponential':
            # Exponential: base * multiplier^k
            n_k = int(base_rays_per_circle * (ray_density_multiplier ** k))
        elif density_growth_type == 'linear':
            # Linear: base + k * multiplier
            n_k = int(base_rays_per_circle + k * ray_density_multiplier)
        elif density_growth_type == 'quadratic':
            # Quadratic: base + k^2 * multiplier
            n_k = int(base_rays_per_circle + (k ** 2) * ray_density_multiplier)
        else:
            raise ValueError(f"Unknown density_growth_type: {density_growth_type}. "
                           f"Must be 'exponential', 'linear', or 'quadratic'")
        rays_per_circle.append(n_k)
    
    total_rays = sum(rays_per_circle)
    
    # Step 3: Setup rotation matrix (90 degrees counterclockwise to match existing system)
    rotation_angle = np.deg2rad(90)
    rotation_matrix = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle)],
        [np.sin(rotation_angle), np.cos(rotation_angle)]
    ], dtype=np.float32)
    
    # Step 4: Generate rays for each circle
    all_origins = []
    all_directions = []
    all_angles = []  # Track original angles before rotation
    circle_indices = []  # Track which circle each ray belongs to
    rays_by_circle = []  # List of dicts with per-circle info for radial path building
    
    global_idx = 0  # Track global index across all circles
    
    for k, radius in enumerate(radii):
        n_rays = rays_per_circle[k]
        
        # Full circle: angles from 0 to 2π (360 degrees)
        # endpoint=False ensures we don't duplicate 0 and 2π
        angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False, dtype=np.float32)
        
        # Store per-circle information for radial path building
        circle_info = {
            'circle_idx': k,
            'radius': radius,
            'n_rays': n_rays,
            'angles': angles.copy(),  # Original angles before rotation
            'global_indices': []  # Will store global indices for this circle's rays
        }
        
        for j, angle in enumerate(angles):
            # Direction vector before rotation (unit vector at angle θ)
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            
            # Apply 90-degree rotation to match coordinate system
            rotated_direction = rotation_matrix @ direction
            
            # Origin point: center + radius × direction
            origin = center + radius * rotated_direction
            
            all_origins.append(origin)
            all_directions.append(rotated_direction)
            all_angles.append(angle)  # Store original angle
            circle_indices.append(k)
            circle_info['global_indices'].append(global_idx)
            global_idx += 1
        
        rays_by_circle.append(circle_info)
    
    # Step 5: Convert to arrays
    origins = np.array(all_origins, dtype=np.float32)
    directions = np.array(all_directions, dtype=np.float32)
    angles = np.array(all_angles, dtype=np.float32)
    circle_indices = np.array(circle_indices, dtype=np.int32)
    
    # Step 6: Apply y-minimum adjustment to match existing system
    # This shifts the coordinate system so the minimum y is at 0
    y_min = np.min(origins[:, 1])
    origins[:, 1] -= y_min
    
    # Step 7: Metadata for downstream use
    metadata = {
        'n_circles': n_circles,
        'rays_per_circle': rays_per_circle,
        'total_rays': total_rays,
        'circle_indices': circle_indices,
        'circle_radii': radii,
        'angles': angles,  # All angles in order
        'rays_by_circle': rays_by_circle,  # Detailed per-circle info for path building
        'inner_radius': inner_radius,
        'outer_radius': outer_radius,
        'base_rays': base_rays_per_circle,
        'ray_multiplier': ray_density_multiplier,
        'density_growth_type': density_growth_type,
        'strategy': 'concentric_circles',
        'y_shift': y_min,  # Store the y-shift for visualization
        'original_center': center.copy()  # Store original center before shift
    }
    
    return origins, directions, metadata



def compute_total_rays_concentric(n_circles, base_rays, ray_multiplier, density_growth_type='exponential'):
    """
    Calculate the total number of rays for concentric circle sampling.
    
    This is a utility function used to compute expected ray counts
    without generating the full ray structure.
    
    Args:
        n_circles (int): Number of concentric circles
        base_rays (int): Number of rays in innermost circle
        ray_multiplier (float): Controls ray growth rate
        density_growth_type (str): Type of density growth:
            - 'exponential': Circle k has floor(base_rays * multiplier^k) rays
            - 'linear': Circle k has floor(base_rays + k * multiplier) rays
            - 'quadratic': Circle k has floor(base_rays + k^2 * multiplier) rays
        
    Returns:
        int: Total number of rays across all circles
        
    Examples:
        >>> # Exponential: 4 + 8 + 16 = 28
        >>> compute_total_rays_concentric(3, 4, 2.0, 'exponential')
        28
        
        >>> # Linear: 8 + 12 + 16 = 36
        >>> compute_total_rays_concentric(3, 8, 4.0, 'linear')
        36
        
        >>> # Quadratic: 8 + 12 + 20 = 40
        >>> compute_total_rays_concentric(3, 8, 3.0, 'quadratic')
        40
    """
    total = 0
    for k in range(n_circles):
        if density_growth_type == 'exponential':
            n_k = int(base_rays * (ray_multiplier ** k))
        elif density_growth_type == 'linear':
            n_k = int(base_rays + k * ray_multiplier)
        elif density_growth_type == 'quadratic':
            n_k = int(base_rays + (k ** 2) * ray_multiplier)
        else:
            raise ValueError(f"Unknown density_growth_type: {density_growth_type}")
        total += n_k
    return total


def find_nearest_ray_by_angle(target_angle, candidate_angles):
    """
    Find the index of the ray with the nearest angle to the target angle.
    Handles wrap-around at 2π (e.g., 359° is close to 1°).
    
    Args:
        target_angle (float): Target angle in radians [0, 2π)
        candidate_angles (np.ndarray): Array of candidate angles in radians
        
    Returns:
        int: Index of the nearest ray in candidate_angles
    """
    # Calculate angular distance accounting for wrap-around
    delta = np.abs(candidate_angles - target_angle)
    # Handle wrap-around: min(delta, 2π - delta)
    delta = np.minimum(delta, 2 * np.pi - delta)
    # Find index of minimum distance
    nearest_idx = np.argmin(delta)
    return nearest_idx


def build_radial_paths_from_concentric(metadata, include_center=True):
    """
    Build radial paths connecting rays from outermost circle to center.
    Each path connects rays by finding nearest neighbors at each inner circle.
    
    This creates the "tree-ring" effect where we integrate along radial paths
    from the outer edge inward to the center.
    
    Args:
        metadata (dict): Metadata from define_rays_concentric_circles()
                        Must contain 'rays_by_circle' with per-circle info
        include_center (bool): Whether to add a center point (radius=0) to each path
        
    Returns:
        paths (list): List of paths, where each path is a list of global indices
                     Format: [[outer_idx, ..., inner_idx, center_idx], ...]
                     Length: number of rays at outermost circle
        path_metadata (dict): Additional information about paths:
            - 'n_paths': Number of paths (same as rays at outer circle)
            - 'path_lengths': List of path lengths (all equal if include_center=True)
            - 'center_idx': Global index of center point (if include_center=True)
            
    Example:
        >>> # With 3 circles: 4, 8, 16 rays
        >>> origins, directions, metadata = define_rays_concentric_circles(
        ...     n_circles=3, inner_radius=100, outer_radius=300,
        ...     base_rays=4, ray_multiplier=2.0, density_growth_type='exponential'
        ... )
        >>> paths, path_meta = build_radial_paths_from_concentric(metadata)
        >>> len(paths)  # 16 paths (rays at outermost circle)
        16
        >>> len(paths[0])  # Each path has 4 nodes (3 circles + center)
        4
    """
    rays_by_circle = metadata['rays_by_circle']
    n_circles = len(rays_by_circle)
    
    if n_circles == 0:
        return [], {'n_paths': 0, 'path_lengths': []}
    
    # Start with outermost circle (highest detail)
    outer_circle = rays_by_circle[-1]
    
    paths = []
    path_lengths = []
    
    # For each ray in the outermost circle, build a path to center
    for ray_idx in range(outer_circle['n_rays']):
        path = []
        
        # Start with this ray's global index
        current_global_idx = outer_circle['global_indices'][ray_idx]
        current_angle = outer_circle['angles'][ray_idx]
        
        path.append(current_global_idx)
        
        # Work backwards through inner circles, finding nearest rays
        for circle_idx in range(n_circles - 2, -1, -1):
            circle_info = rays_by_circle[circle_idx]
            
            # Find nearest ray by angle
            nearest_local_idx = find_nearest_ray_by_angle(
                current_angle,
                circle_info['angles']
            )
            
            # Get global index of nearest ray
            nearest_global_idx = circle_info['global_indices'][nearest_local_idx]
            path.append(nearest_global_idx)
            
            # Update current angle to the matched ray's angle
            current_angle = circle_info['angles'][nearest_local_idx]
        
        # Optionally add center point
        if include_center:
            # Center point gets a special index (total_rays)
            center_idx = metadata['total_rays']
            path.append(center_idx)
        
        paths.append(path)
        path_lengths.append(len(path))
    
    # Package path metadata
    path_metadata = {
        'n_paths': len(paths),
        'path_lengths': path_lengths,
        'center_idx': metadata['total_rays'] if include_center else None,
        'include_center': include_center
    }
    
    return paths, path_metadata


def define_rays(num_rays, inner_radius, opening_angle, center=None, device=device):
    """
    Function to define rays using PyTorch tensors, defined by number of rays, inner radius,
    opening angle, and the center. The function returns the ray origins and directions after
    rotating the rays by -90 degrees.

    Args:
        num_rays (int): Number of rays.
        inner_radius (float): Inner radius of the fan shape.
        opening_angle (float): Opening angle of the fan in degrees.
        center (tuple): The center of the fan. Defaults to the center of the plot.
        device (str): The device ('cpu' or 'cuda'). Default is 'cpu'.

    Returns:
        origins (torch.Tensor): Ray origins for each ray.
        directions (torch.Tensor): Directions of the rays after rotation.
    """
    # if center is None:
    #     center = (0, 0)  # Default center at origin (0, 0)
    #
    # # Convert the center to a PyTorch tensor
    # center = torch.tensor(center, dtype=torch.float32, device=device)

    # Convert opening angle to radians
    opening_angle_rad = torch.tensor(opening_angle, dtype=torch.float32, device=device) * (torch.pi / 180)

    # Create an array of angles for the rays
    angle_step = opening_angle_rad / (num_rays - 1)
    angles = torch.linspace(-opening_angle_rad / 2, opening_angle_rad / 2, num_rays, device=device)

    # Rotation matrix for -90 degrees
    a = torch.deg2rad(torch.tensor(90.0, dtype=torch.float32, device=device))  # Convert 90 degrees to radians
    rotation_matrix = torch.tensor([[torch.cos(a), -torch.sin(a)],
                                    [torch.sin(a), torch.cos(a)]], device=device)

    origins = []
    directions = []

    # Plot rays from inner radius to outer radius
    for angle in angles:
        # Directions (unit vectors) for the rays
        direction = torch.tensor([torch.cos(angle), torch.sin(angle)], dtype=torch.float32, device=device)

        # Apply the -90 degree rotation (counterclockwise)
        rotated_direction = torch.matmul(rotation_matrix, direction)

        directions.append(rotated_direction)

        # Origins at the inner radius (no outer radius used here)
        inner_origin = inner_radius * rotated_direction
        origins.append(inner_origin)

    # Stack the origins and directions into tensors
    origins = torch.stack(origins, dim=0)
    directions = torch.stack(directions, dim=0)

    # Adjust origins based on y_min
    y_min = torch.min(origins[..., 1])
    origins[..., 1] -= y_min

    return origins, directions



def create_and_save_image(W, H, filename="image.png", color=(255, 255, 255)):
    """
    Creates an image of size W x H and saves it as a PNG file.

    Args:
        W (int): Width of the image.
        H (int): Height of the image.
        filename (str): The name of the file to save (default is 'image.png').
        color (tuple): The color to fill the image with (default is white (255, 255, 255)).
                       Color is given as an (R, G, B) tuple with values between 0 and 255.
    """
    # Create a new image with the specified width, height, and color
    img = Image.new("RGB", (W, H), color)

    # Save the image as a PNG file
    img.save(filename, "PNG")
    print(f"Image saved as {filename} ({W}x{H}).")


def get_filename_targets_for_comparison(path, sample_points):
    """Returns the name of the file to store converted images cache.
    
    Cache filename includes:
    - Number of rays (sample_points.shape[0])
    - Number of samples per ray (sample_points.shape[1])
    - Sampling strategy (uniform_fan or concentric_circles)
    - Density growth type (exponential, linear, or quadratic)
    
    This ensures different sampling strategies don't share caches.
    """    
    n_rays = sample_points.shape[0]
    n_samples = sample_points.shape[1]
    strategy = convex.config.get('sampling_strategy', 'uniform_fan')
    growth_type = convex.config.get('density_growth_type', 'exponential')
    
    # Create strategy-specific filename
    if strategy == 'concentric_circles':
        return f'{path}/converted_images_{strategy}_{growth_type}_{n_rays}_{n_samples}.npy'
    else:
        return f'{path}/converted_images_{strategy}_{n_rays}_{n_samples}.npy'


def cache_targets_for_comparison(images, sample_points, path):
    """
    We convert original convex image to new one containing only points from sampling rays.
    Some images could be used several times for training, to reduce time of converting
    we store converted images to the file and use it as a cache.
    """
    converted_image_file = get_filename_targets_for_comparison(path, sample_points)
    if os.path.isfile(converted_image_file):
        cached_images = np.load(converted_image_file, allow_pickle=True)
    else:
        cached_images = np.empty(shape=(images.shape[0], sample_points.shape[0], sample_points.shape[1]))
        for img_i, image in enumerate(images):
            cached_images[img_i] = remap_target_for_comparison(images[img_i], sample_points)

        with open(converted_image_file, 'wb') as f:
            np.save(converted_image_file, cached_images, allow_pickle=True, fix_imports=True)

    return cached_images


def remap_target_for_comparison(target, sample_points):
    """
    Build the image from original convex image for comparison
    based on original target and sampling points."""
    new_target = target[sample_points[..., 1], sample_points[..., 0]]
    return new_target


def is_in_convex_fan(x, y):
    """
    The function checks if the point (x, y) is in a convex fan area."""
    center, cx, cy, num_rays, angle, radius, radius2 = convex.get_convex_settings()
    dist = math.sqrt(math.pow((x - cx), 2) + math.pow((y - cy), 2))
    if dist > radius or dist < radius2:
        return False

    angel = math.degrees(math.atan2(y - cy, x - cx))
    startAngle = 90 - angle / 2
    endAngle = 90 + angle / 2
    return angel >= startAngle and angel <= endAngle

def remap_output_to_original_image(output, sample_points, W, H, interpolation_method='nearest', _interpolate=1, _mask=1):
    """
    Build full-size image based on MLP output and sampling points.
    Uses opencv `inpaint` function to interpolate missing pixels.

    Args:
        output (np.ndarray): Array of sample points (values from MLP).
        sample_points (np.ndarray): Coordinates of the sample points.
        W (int): Image width.
        H (int): Image height.

    Returns:
        np.ndarray: Image with missing pixels filled using interpolation.
    """
    img = np.full((H, W), np.nan, dtype=np.float32)
    flattened_arr = sample_points.reshape(-1, 2)  # Reshape to (num_samples, 2)

    unique_vals, indices = np.unique(flattened_arr, axis=0, return_index=True)

    img[unique_vals[:, 1], unique_vals[:, 0]] = output.reshape(-1)[indices]
    missing_mask = np.ma.masked_invalid(img)
    inpaint_img = interpolate_missing_pixels(img, missing_mask.mask, method=interpolation_method, _interpolate=_interpolate)

    # If H=W (square image), mask circular region and set outside pixels to 0
    if H == W and _mask:
        # Center is the middle of the square
        cx, cy = W // 2, H // 2
        radius = W // 2  # Circle fits exactly in the square
        
        # Create coordinate grids
        y_coords, x_coords = np.ogrid[:H, :W]
        
        # Calculate distance from center for each pixel
        dist_from_center = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        
        # Create circular mask: keep only pixels within radius
        circular_mask = dist_from_center <= radius
        
        # Apply mask
        inpaint_img = np.where(circular_mask, inpaint_img, 0.0).astype(np.float32)

    return inpaint_img.astype(np.float32)

def remap_output_to_original_volume_3d(output_volume, sample_points_volume, W, H, D, 
                                       interpolation_method='linear', _interpolate=1):
    """
    Build full-size 3D volume using true 3D interpolation. (slow)
    
    Args:
        output_volume (np.ndarray): [D, n_rays, n_samples] from NeRF rendering
        sample_points_volume (np.ndarray): [D, n_rays, n_samples, 2] - (x,y) coords per slice
        W, H, D (int): Target volume dimensions (Width, Height, Depth)
        interpolation_method (str): 'linear' or 'nearest'
        _interpolate (int): 1 to enable interpolation
    
    Returns:
        np.ndarray: Interpolated volume [D, H, W]
    """
    # Initialize output volume with NaNs
    volume = np.full((D, H, W), np.nan, dtype=np.float32)
    
    # Collect all 3D sampling points and their values
    points_3d = []  # Will be (N, 3) - (x, y, z) coordinates
    values = []     # Will be (N,) - intensity values
    
    for d in range(D):
        # Get 2D points for this slice
        pts_2d = sample_points_volume[d].reshape(-1, 2)  # [n_points, 2]
        vals = output_volume[d].reshape(-1)  # [n_points]
        
        # Create 3D points by adding depth coordinate
        pts_3d = np.column_stack([pts_2d, np.full(len(pts_2d), d)])  # [n_points, 3]
        
        points_3d.append(pts_3d)
        values.append(vals)
    
    # Concatenate all points
    points_3d = np.vstack(points_3d)  # [total_points, 3]
    values = np.concatenate(values)    # [total_points]
    
    # Remove duplicates
    unique_points, indices = np.unique(points_3d, axis=0, return_index=True)
    unique_values = values[indices]
    
    if _interpolate:
        # Create 3D query grid
        zz, yy, xx = np.meshgrid(
            np.arange(D), np.arange(H), np.arange(W), indexing='ij'
        )
        query_points = np.column_stack([
            xx.ravel(), yy.ravel(), zz.ravel()
        ])
        
        # Perform 3D interpolation        
        if interpolation_method == 'nearest':
            print("Using nearest neighbor interpolation for 3D volume.")
            interpolator = NearestNDInterpolator(unique_points, unique_values)
            interpolated_values = interpolator(query_points)
        elif interpolation_method == 'linear':  # linear
            interpolated_values = griddata(
                unique_points, unique_values, query_points,
                method='linear', fill_value=0.0
            )
            # Fill remaining NaNs with nearest neighbor
            nan_mask = np.isnan(interpolated_values)
            if nan_mask.any():
                nn_interp = NearestNDInterpolator(unique_points, unique_values)
                interpolated_values[nan_mask] = nn_interp(query_points[nan_mask])
        else:
            print("Using cubic interpolation for 3D volume.")
            interpolator = RegularGridInterpolator(unique_points, unique_values, method='cubic')
            interpolated_values = interpolator(query_points)
        
        # Reshape to volume
        volume = interpolated_values.reshape(D, H, W)
    else:
        # Just fill known points without interpolation
        for pt, val in zip(unique_points, unique_values):
            x, y, z = pt.astype(int)
            if 0 <= x < W and 0 <= y < H and 0 <= z < D:
                volume[z, y, x] = val
    
    return volume.astype(np.float32)


def calc_image_sampling_points_convex(N_samples, lindisp=False, near=0., far=55. * 0.001, **kwargs):
    """
    Calculates all sampling points on convex image.
    Don't use scaling to keep original image coordinates in pixels.
    Note: `convex_n_rays` parameter provided for convex mode overrides `N_samples`.
    
    Supports two sampling strategies:
    - 'uniform_fan': Original uniform fan sampling (default)
    - 'concentric_circles': Concentric circles with exponential ray density

    """
    center, cx, cy, num_rays, angle, radius, radius2 = convex.get_convex_settings()
    norm_radius = radius2 / radius
    
    # Check sampling strategy
    sampling_strategy = convex.config.get('sampling_strategy', 'uniform_fan')
    
    if sampling_strategy == 'concentric_circles':
        # Concentric circles mode
        n_circles = convex.config.get('n_circles', 5)
        base_rays = convex.config.get('base_rays_per_circle', 8)
        ray_multiplier = convex.config.get('ray_density_multiplier', 1.5)
        density_growth = convex.config.get('density_growth_type', 'exponential')
        
        origin, direction, metadata = define_rays_concentric_circles(
            n_circles=n_circles,
            inner_radius=norm_radius,
            outer_radius=1.0,
            base_rays_per_circle=base_rays,
            ray_density_multiplier=ray_multiplier,
            density_growth_type=density_growth
        )
    else:
        # Default: uniform fan mode
        origin, direction = define_rays_numpy(num_rays, norm_radius, angle)
    
    delta_radius = (radius - radius2) / radius
    rays_o_image = origin
    rays_d_image = direction
    t_vals = np.linspace(0., delta_radius, N_samples)
    
    # Broadcast z_vals to match rays_d_image shape
    z_vals = np.ones_like(rays_d_image[..., :1]) * t_vals
    
    # Compute points along rays
    origin_image = rays_o_image[..., None, :]
    step_image = rays_d_image[..., None, :] * z_vals[..., :, None]
    pts_image = step_image + origin_image

    return pts_image


def get_rays_us_convex(sw, sh, c2w):
    """Is used in convex mode only
    Defines ray's origin and direction in `real space` by applying translation
    and rotation from poses. Scale to real image in mm from pixels.
    
    Supports two sampling strategies:
    - 'uniform_fan': Original uniform fan sampling (default)
    - 'concentric_circles': Concentric circles with exponential ray density (TODO)
    """
    center, cx, cy, num_rays, angle, radius, radius2 = convex.get_convex_settings()
    norm_radius = radius2 / radius
    global __origin, __direction
    global __origin_val, __direction_val
    
    # Check sampling strategy
    sampling_strategy = convex.config.get('sampling_strategy', 'uniform_fan')
    
    if sampling_strategy == 'concentric_circles':
        # Concentric circles mode: use new sampling strategy
        n_circles = convex.config.get('n_circles', 5)
        base_rays = convex.config.get('base_rays_per_circle', 8)
        ray_multiplier = convex.config.get('ray_density_multiplier', 1.5)
        density_growth = convex.config.get('density_growth_type', 'exponential')

        # Generate rays with concentric circles
        if __origin is None and __direction is None:
            __origin, __direction, metadata = define_rays_concentric_circles(
                n_circles=n_circles,
                inner_radius=norm_radius,
                outer_radius=1.0,
                base_rays_per_circle=base_rays,
                ray_density_multiplier=ray_multiplier,
                density_growth_type=density_growth
            )
            print(metadata, "Ray metadata")
            # Store metadata for debugging/validation
            convex.config['_ray_metadata'] = metadata
        
        origin, direction = __origin, __direction
        
        # For validation mode (different number of circles)
        expected_total_rays = compute_total_rays_concentric(n_circles, base_rays, ray_multiplier, density_growth)
        #print(__origin.shape[0], expected_total_rays, "Ray count check")
        if __origin.shape[0] != expected_total_rays:
            if __origin_val is None and __direction_val is None:
                __origin_val, __direction_val, metadata_val = define_rays_concentric_circles(
                    n_circles=n_circles,
                    inner_radius=norm_radius,
                    outer_radius=1.0,
                    base_rays_per_circle=base_rays,
                    ray_density_multiplier=ray_multiplier,
                    density_growth_type=density_growth
                )
                convex.config['_ray_metadata_val'] = metadata_val
            
            origin, direction = __origin_val, __direction_val
        num_rays = origin.shape[0]
    else:
        # Default: uniform_fan mode (original behavior)
        if __origin is None and __direction is None:
            __origin, __direction = define_rays_numpy(num_rays, norm_radius, angle)     
        
        origin, direction = __origin, __direction
        
        if __origin.shape[0] != num_rays:
            if __origin_val is None and __direction_val is None:
                __origin_val, __direction_val = define_rays_numpy(num_rays, norm_radius, angle)
                
            origin, direction = __origin_val, __direction_val

    rays_o_image = torch.from_numpy(origin)
    rays_d_image = torch.from_numpy(direction)
    rays_o_x, rays_o_y = rays_o_image[..., 0:1], rays_o_image[..., 1:]
    rays_d_x, rays_d_y = rays_d_image[..., 0:1], rays_d_image[..., 1:]
    
    # Set up the camera transformation matrix
    t = c2w[:3, -1]  # Translation vector
    R = c2w[:3, :3]  # Rotation matrix
    
    oz = torch.zeros_like(rays_o_x)
    rays_o_tmp = torch.stack([rays_o_x, rays_o_y, oz], dim=-1)
    rays_o_base = rays_o_tmp.to(device)
    dz = torch.zeros_like(rays_d_x)
    
    rays_d_tmp = torch.stack([rays_d_x, rays_d_y, dz], dim=-1)
    rays_d_base = rays_d_tmp.to(device)
    origin_base_prim = rays_o_base.unsqueeze(-2)  # Add an extra dimension for matrix multiplication
    
    # Apply rotation to origins, so origins are now in space
    origin_rotated = R * origin_base_prim
    ray_o_r = torch.sum(origin_rotated, dim=-1)
    
    # Apply translation to ray origins
    rays_o = ray_o_r + t
    dirs_base_tmp = rays_d_base.reshape(num_rays, 3).unsqueeze(-2)
    dirs_r = R * dirs_base_tmp
    rays_d = torch.sum(dirs_r, dim=-1)

    return rays_o, rays_d


# ─────────────────────────────────────────────────────────────────────────────
# Spherical (elevation-angle) ray generation for 3D volumetric imaging
# ─────────────────────────────────────────────────────────────────────────────

# Module-level cache for spherical rays
__origin_spherical = None
__direction_spherical = None
__spherical_metadata = None


def get_rays_us_convex_spherical(sw, sh, c2w):
    """Generate all 3D rays for a spherical fan-beam volume in a single call.

    Instead of calling get_rays_us_convex() D times (once per slice pose),
    this function generates all N_azimuth × N_elevation rays at once using
    a single volume pose. The elevation dimension is geometrically built-in
    through the spherical angle sweep rather than through separate poses.

    Key differences from get_rays_us_convex:
        1. Rays are natively 3D (z ≠ 0) — no need to append zeros
        2. A single pose transforms the entire cone, not per-slice poses
        3. The volume is a spherical frustum, not a stack of flat fans

    Args:
        sw (float): Scaling factor x (pixels to meters). Passed for interface
            compatibility but NOT used (same as get_rays_us_convex).
        sh (float): Scaling factor y (pixels to meters). Same note.
        c2w (torch.Tensor): Camera-to-world pose matrix [3, 4] or [4, 4].
            A SINGLE pose for the entire volume (not per-slice).

    Returns:
        rays_o (torch.Tensor): Ray origins in world space,
            shape [N_azimuth * N_elevation, 3].
        rays_d (torch.Tensor): Ray directions in world space,
            shape [N_azimuth * N_elevation, 3].
    """
    global __origin_spherical, __direction_spherical, __spherical_metadata

    center, cx, cy, num_rays_azimuth, angle, radius, radius2 = convex.get_convex_settings()
    norm_radius = radius2 / radius

    n_rays_elevation = convex.config.get('n_rays_elevation', 68)
    opening_angle_elevation = convex.config.get('opening_angle_elevation', 20.0)

    # Generate (and cache) the spherical ray geometry
    if __origin_spherical is None or __direction_spherical is None:
        __origin_spherical, __direction_spherical, __spherical_metadata = \
            define_rays_spherical_numpy(
                n_rays_azimuth=num_rays_azimuth,
                n_rays_elevation=n_rays_elevation,
                inner_radius=norm_radius,
                opening_angle_azimuth=angle,
                opening_angle_elevation=opening_angle_elevation,
            )
        convex.config['_spherical_metadata'] = __spherical_metadata
        print(f"[Spherical] Generated {__spherical_metadata['total_rays']} rays: "
              f"{num_rays_azimuth} azimuth × {n_rays_elevation} elevation")

    origin = __origin_spherical    # [N_total, 3]
    direction = __direction_spherical  # [N_total, 3]
    num_rays = origin.shape[0]

    # Convert to torch
    rays_o_base = torch.from_numpy(origin).to(device).float()   # [N, 3]
    rays_d_base = torch.from_numpy(direction).to(device).float()  # [N, 3]

    # Apply single volume pose: world = R @ local + t
    R = c2w[:3, :3]   # [3, 3]
    t = c2w[:3, -1]   # [3]

    # Batch matrix-vector multiply
    rays_o = (R @ rays_o_base.unsqueeze(-1)).squeeze(-1) + t   # [N, 3]
    rays_d = (R @ rays_d_base.unsqueeze(-1)).squeeze(-1)       # [N, 3]

    return rays_o, rays_d


def calc_image_sampling_points_spherical(N_samples, lindisp=False,
                                         near=0., far=55. * 0.001, **kwargs):
    """Compute all 3D sampling points for spherical fan-beam imaging.

    Analogous to calc_image_sampling_points_convex() but for the spherical
    (elevation-angle) mode. Points lie on a 3D grid:
        [N_elevation, N_azimuth, N_samples_per_ray, 3]

    These can be used for target remapping and cache generation.

    Args:
        N_samples (int): Number of samples per ray (overridden by convex config).
        **kwargs: Absorbs extra keyword arguments for compatibility.

    Returns:
        pts_image (np.ndarray): Sampling points in local (image) coordinates,
            shape [N_elevation, N_azimuth, N_samples, 3].
    """
    center, cx, cy, num_rays_azimuth, angle, radius, radius2 = convex.get_convex_settings()
    norm_radius = radius2 / radius

    n_rays_elevation = convex.config.get('n_rays_elevation', 68)
    opening_angle_elevation = convex.config.get('opening_angle_elevation', 20.0)

    origins, directions, metadata = define_rays_spherical_numpy(
        n_rays_azimuth=num_rays_azimuth,
        n_rays_elevation=n_rays_elevation,
        inner_radius=norm_radius,
        opening_angle_azimuth=angle,
        opening_angle_elevation=opening_angle_elevation,
    )

    delta_radius = (radius - radius2) / radius
    t_vals = np.linspace(0., delta_radius, N_samples, dtype=np.float32)

    # pts[i, j, :] = origin[i] + t[j] * direction[i]
    # origins:    [N_total, 3]
    # directions: [N_total, 3]
    # t_vals:     [N_samples]
    origin_exp = origins[:, None, :]               # [N_total, 1, 3]
    step = directions[:, None, :] * t_vals[None, :, None]  # [N_total, N_samples, 3]
    pts = origin_exp + step                         # [N_total, N_samples, 3]

    # Reshape to [N_elevation, N_azimuth, N_samples, 3]
    pts = pts.reshape(n_rays_elevation, num_rays_azimuth, N_samples, 3)

    return pts


def show_colorbar(image, cmap="rainbow"):
    figure = plt.figure(figsize=(5, 5))
    plt.imshow(image.numpy(), cmap=cmap)
    plt.colorbar()
    buf = io.BytesIO()
    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format="png")
    plt.close(figure)
    return buf


def define_image_grid_3D_np(x_size, y_size):
    y = np.array(range(x_size))
    x = np.array(range(y_size))
    xv, yv = np.meshgrid(x, y, indexing="ij")
    image_grid_xy = np.vstack((xv.ravel(), yv.ravel()))
    z = np.zeros(image_grid_xy.shape[1])
    image_grid = np.vstack((image_grid_xy, z))
    return image_grid
