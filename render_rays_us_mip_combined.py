"""
Combined multivariate Gaussian estimation for NeRF rendering for ultrasound with optional additonal manual elongation.

This module combines render_rays_us_mip (conical frustums with multivariate Gaussian estimation for 3 US resolution directions) and 
render_rays_us_mip_elongated (manual sideways-elongated Gaussians) into a single
unified function with an elongation boolean flag.

Used for experiments and thesis: render_rays_us_mip

The rendering is an extension of the mip-NeRF paper, using proper conical frustum mathematics, but for anisotropic Gaussian distributions with 3 variance.
The depth direction is additional that represents the elevational axis.
"""

from cmath import sqrt
import sys
import torch
import torch.nn.functional as F
import convex_configuration as convex
from rendering import render_method_3, render_method_ultra_nerf
import matplotlib.pyplot as plt
import numpy as np
import os

def render_rays_us_mip_combined(
    ray_batch,
    network_fn,
    network_query_fn,
    N_samples,
    use_elongation=False,
    center_point=None,
    max_elongation=5.0,
    voxel_radius_lateral=0.173,
    voxel_radius_depth=0.173,
    constant_gaussian_size=True,
    **kwargs,
):
    """
    Volumetric rendering with mip-NeRF conical frustums or elongated Gaussians.
    
    Args:
        ray_batch: Tensor of shape [N_rays, 8] containing ray origins, directions, near, far
        network_fn: NeRF network
        network_query_fn: Function to query network with encoded positions
        N_samples: Number of samples per ray
        use_elongation: If True, uses elongated sideways Gaussians; if False, uses standard conical frustums
        center_point: Center point for elongation calculation [3] (only used if use_elongation=True)
        max_elongation: Maximum elongation factor for sideways stretching (only used if use_elongation=True)
        voxel_radius_lateral: Base voxel footprint radius for lateral (x-y) spread (default: 0.173)
        voxel_radius_depth: Base voxel footprint radius for depth (z) spread -- elevational resolution (default: 0.173)
        constant_gaussian_size: If True, Gaussians have constant size (not growing with distance which is correct for depth).
                                If False, uses mip-NeRF formula where Gaussians grow with distance.
        **kwargs: Additional keyword arguments
    
    Returns:
        ret: Dictionary with rendered outputs (intensity_map, attenuation_coeff, etc.)
    """
    
    #print(f"=== Mip-NeRF Rendering (elongation={'ON' if use_elongation else 'OFF'}) ===")
    #print(f"Voxel radius lateral: {voxel_radius_lateral}, depth: {voxel_radius_depth}")
    
    def raw2outputs(raw, z_vals, distance):
        """Transforms model's predictions to semantically meaningful values."""
        if convex.config['full_volume_mode']:
            if raw.shape[-1] == 5:
                ret = render_method_ultra_nerf(raw)
            elif raw.shape[-1] == 3:
                ret = render_method_3(raw)
        else:
            ret = render_method_3(raw)
        return ret

    def compute_conical_frustum_gaussians(rays_o, rays_d, z_vals, voxel_radius_lateral=0.173, voxel_radius_depth=0.173,
                                         constant_size=False):
        """
        Compute Gaussian parameters for ellipsoidal conical frustums (extended mip-NeRF for 3D ultrasound).
        Uses numerically stable formulas from mip-NeRF paper with separate radii for lateral and depth spread.
        
        Args:
            rays_o: Ray origins [N_rays, 3]
            rays_d: Ray directions [N_rays, 3] 
            z_vals: Sample depths [N_rays, N_samples]
            voxel_radius_lateral: Voxel footprint radius for lateral (x-y plane) spread
            voxel_radius_depth: Voxel footprint radius for depth (z-axis) spread
            constant_size: If True, Gaussians have constant size regardless of distance
            
        Returns:
            means: Gaussian means [N_rays, N_samples, 3]
            covs: Gaussian covariances [N_rays, N_samples, 3, 3]
        """
        N_rays, N_samples = z_vals.shape
        device = rays_o.device
                
        # Get frustum boundaries: t0 and t1
        # t0 is the start of each frustum, t1 is the end
        t0 = torch.cat([
            torch.zeros((*z_vals.shape[:-1], 1), device=device),
            z_vals[..., :-1]
        ], dim=-1)  # [N_rays, N_samples]
        t1 = z_vals  # [N_rays, N_samples]
        
        # Compute midpoint and half-width for numerical stability
        mu = (t0 + t1) / 2  # [N_rays, N_samples]
        hw = (t1 - t0) / 2  # [N_rays, N_samples]
        
        # Prevent division by zero and numerical instabilities
        mu = torch.maximum(mu, torch.tensor(1e-10, device=device))
        hw = torch.maximum(hw, torch.tensor(1e-10, device=device))
        
        # Precompute common terms with numerical stability
        mu_sq = mu ** 2
        hw_sq = hw ** 2
        hw_quad = hw_sq ** 2
        denom = 3 * mu_sq + hw_sq
        denom = torch.maximum(denom, torch.tensor(1e-10, device=device))
        
        # Stable formulas from mip-NeRF paper (Equation 7)
        # Mean distance along ray
        t_mean = mu + (2 * mu * hw_sq) / denom  # [N_rays, N_samples]
        
        # Variance along ray direction (longitudinal)
        t_var = hw_sq / 3 - (4 / 15) * (
            (hw_quad * (12 * mu_sq - hw_sq)) / (denom ** 2)
        )  # [N_rays, N_samples]
        t_var = torch.maximum(t_var, torch.tensor(0.0, device=device))
        
        # Variance perpendicular to ray direction (radial - lateral x-y)
        if constant_size:
            # Constant size: variance only depends on frustum width, not distance
            r_var_lateral = voxel_radius_lateral**2 * (
                (5 / 12) * hw_sq - 
                (4 / 15) * hw_quad / denom
            )  # [N_rays, N_samples]
        else:
            # mip-NeRF: variance grows with distance (mu_sq / 4 term)
            r_var_lateral = voxel_radius_lateral**2 * (
                mu_sq / 4 + (5 / 12) * hw_sq - 
                (4 / 15) * hw_quad / denom
            )  # [N_rays, N_samples]
        r_var_lateral = torch.maximum(r_var_lateral, torch.tensor(0.0, device=device))
        
        # Variance perpendicular to ray direction (radial - depth z)
        if constant_size:
            # Constant size: variance only depends on frustum width, not distance
            r_var_depth = voxel_radius_depth**2 * (
                (5 / 12) * hw_sq - 
                (4 / 15) * hw_quad / denom
            )  # [N_rays, N_samples]
        else:
            # mip-NeRF: variance grows with distance (mu_sq / 4 term)
            r_var_depth = voxel_radius_depth**2 * (
                mu_sq / 4 + (5 / 12) * hw_sq - 
                (4 / 15) * hw_quad / denom
            )  # [N_rays, N_samples]
        r_var_depth = torch.maximum(r_var_depth, torch.tensor(0.0, device=device))
        
        # Compute sample positions (means of Gaussians) using t_mean
        means = rays_o[..., None, :] + rays_d[..., None, :] * t_mean[..., :, None]  # [N_rays, N_samples, 3]
        
        # Create covariance matrices
        covs = torch.zeros(N_rays, N_samples, 3, 3, device=device)
        
        var_along_ray = t_var  # [N_rays, N_samples]
        
        # Normalize ray directions
        rays_d_norm = F.normalize(rays_d, dim=-1)  # [N_rays, 3]
        
        # Create orthonormal basis for each ray direction
        # Identify which perpendicular direction corresponds to lateral vs depth
        # Assuming world z-axis is depth direction
        world_z = torch.tensor([0., 0., 1.], device=device)
        
        # Check if ray is nearly aligned with z-axis
        is_nearly_vertical = torch.abs(rays_d_norm[:, 2]) > 0.9  # [N_rays]
        
        # For non-vertical rays, create basis aligned with world coordinates
        ref_vec = torch.zeros_like(rays_d_norm)  # [N_rays, 3]
        ref_vec[~is_nearly_vertical] = world_z
        ref_vec[is_nearly_vertical, 0] = 1.0  # Use x-axis for vertical rays
        
        # Compute first perpendicular vector
        v1 = torch.cross(rays_d_norm, ref_vec, dim=-1)  # [N_rays, 3]
        v1 = F.normalize(v1, dim=-1, eps=1e-8)
        
        # Compute second perpendicular vector
        v2 = torch.cross(rays_d_norm, v1, dim=-1)  # [N_rays, 3]
        v2 = F.normalize(v2, dim=-1, eps=1e-8)
        
        # For most rays, v1 will be lateral (in x-y plane) and v2 will have z-component (depth)
        # Stack to create basis matrices: [ray_direction, lateral, depth]
        basis = torch.stack([rays_d_norm, v1, v2], dim=-1)  # [N_rays, 3, 3]
        
        # Create diagonal variance matrices with ellipsoidal cross-section
        diag_vars = torch.zeros(N_rays, N_samples, 3, 3, device=device)
        diag_vars[:, :, 0, 0] = var_along_ray      # Along ray direction
        diag_vars[:, :, 1, 1] = r_var_lateral      # Lateral perpendicular (x-y)
        diag_vars[:, :, 2, 2] = r_var_depth        # Depth perpendicular (z-component) (elevational resolution)
        
        # Transform to world coordinates: Cov = R @ diag(variances) @ R^T
        basis_expanded = basis.unsqueeze(1)  # [N_rays, 1, 3, 3]
        basis_T_expanded = basis_expanded.transpose(-2, -1)  # [N_rays, 1, 3, 3]
        
        temp = torch.matmul(basis_expanded, diag_vars)  # [N_rays, N_samples, 3, 3]
        covs = torch.matmul(temp, basis_T_expanded)     # [N_rays, N_samples, 3, 3]
        
        # Replace NaN or Inf with small positive values
        covs = torch.where(torch.isnan(covs) | torch.isinf(covs), 
                          torch.tensor(1e-10, device=device), 
                          covs)
        
        return means, covs
    
    def compute_elongated_sideways_gaussians(rays_o, rays_d, z_vals, voxel_radius_lateral=0.173, voxel_radius_depth=0.173,
                                            center_point=None, max_elongation=5.0):
        """
        Compute Gaussian parameters with sideways elongation (experimental).
        Uses the same stable mu/hw formulas as compute_conical_frustum_gaussians
        but adds elongation that increases with distance from center point.
        
        Args:
            rays_o: Ray origins [N_rays, 3]
            rays_d: Ray directions [N_rays, 3] 
            z_vals: Sample depths [N_rays, N_samples]
            voxel_radius_lateral: Base voxel footprint radius for lateral spread
            voxel_radius_depth: Base voxel footprint radius for depth spread
            center_point: Center point for elongation calculation [3]
            max_elongation: Maximum elongation factor
            
        Returns:
            means: Gaussian means [N_rays, N_samples, 3]
            covs: Gaussian covariances [N_rays, N_samples, 3, 3]
        """
        N_rays, N_samples = z_vals.shape
        device = rays_o.device
        
        # Use origin as center if not specified
        if center_point is None:
            center_point = torch.zeros(3, device=device)
        else:
            center_point = center_point.to(device)
        
        # Get frustum boundaries: t0 and t1 (same as standard method)
        t0 = torch.cat([
            torch.zeros((*z_vals.shape[:-1], 1), device=device),
            z_vals[..., :-1]
        ], dim=-1)  # [N_rays, N_samples]
        t1 = z_vals  # [N_rays, N_samples]
        
        # Compute midpoint and half-width for numerical stability
        mu = (t0 + t1) / 2  # [N_rays, N_samples]
        hw = (t1 - t0) / 2  # [N_rays, N_samples]
        
        # Prevent division by zero and numerical instabilities
        mu = torch.maximum(mu, torch.tensor(1e-10, device=device))
        hw = torch.maximum(hw, torch.tensor(1e-10, device=device))
        
        # Precompute common terms with numerical stability
        mu_sq = mu ** 2
        hw_sq = hw ** 2
        hw_quad = hw_sq ** 2
        denom = 3 * mu_sq + hw_sq
        denom = torch.maximum(denom, torch.tensor(1e-10, device=device))
        
        # Stable formulas from mip-NeRF paper (Equation 7)
        t_mean = mu + (2 * mu * hw_sq) / denom  # [N_rays, N_samples]
        
        # Variance along ray direction (longitudinal)
        t_var = hw_sq / 3 - (4 / 15) * (
            (hw_quad * (12 * mu_sq - hw_sq)) / (denom ** 2)
        )  # [N_rays, N_samples]
        t_var = torch.maximum(t_var, torch.tensor(0.0, device=device))
        
        # Base radial variance (before elongation) - lateral (x-y plane)
        r_var_base_lateral = voxel_radius_lateral**2 * (
            mu_sq / 4 + (5 / 12) * hw_sq - 
            (4 / 15) * hw_quad / denom
        )  # [N_rays, N_samples]
        r_var_base_lateral = torch.maximum(r_var_base_lateral, torch.tensor(0.0, device=device))
        
        # Base radial variance (before elongation) - depth (z-axis)
        r_var_base_depth = voxel_radius_depth**2 * (
            mu_sq / 4 + (5 / 12) * hw_sq - 
            (4 / 15) * hw_quad / denom
        )  # [N_rays, N_samples]
        r_var_base_depth = torch.maximum(r_var_base_depth, torch.tensor(0.0, device=device))
        
        # Compute sample positions using t_mean
        means = rays_o[..., None, :] + rays_d[..., None, :] * t_mean[..., :, None]  # [N_rays, N_samples, 3]
        
        # Compute distance from each sample point to center (horizontal plane)
        horizontal_distance = torch.sqrt(
            (means[..., 0] - center_point[0])**2 + 
            (means[..., 1] - center_point[1])**2 + 
            1e-8
        )  # [N_rays, N_samples]
        
        # Normalize distance to create elongation factor
        max_distance = torch.max(horizontal_distance)
        if max_distance > 1e-6:
            normalized_distance = torch.clamp(horizontal_distance / max_distance, 0, 1)
        else:
            normalized_distance = torch.zeros_like(horizontal_distance)
        
        # Elongation factor: grows from 1.0 at center to max_elongation at edges
        elongation_factor = 1.0 + normalized_distance * (max_elongation - 1.0)  # [N_rays, N_samples]
        
        # Normalize ray directions
        rays_d_norm = F.normalize(rays_d, dim=-1)  # [N_rays, 3]
        
        # Create orthonormal basis for coordinate transformation
        # Determine reference vector based on ray direction
        use_x_ref = torch.abs(rays_d_norm[:, 0]) < 0.9  # [N_rays]
        
        ref_vec = torch.zeros_like(rays_d_norm)  # [N_rays, 3]
        ref_vec[use_x_ref, 0] = 1.0   # Use [1,0,0] for most rays
        ref_vec[~use_x_ref, 1] = 1.0  # Use [0,1,0] for rays aligned with x-axis
        
        # Compute first perpendicular vector
        v1 = torch.cross(rays_d_norm, ref_vec, dim=-1)  # [N_rays, 3]
        v1 = F.normalize(v1, dim=-1, eps=1e-8)
        
        # Compute second perpendicular vector
        v2 = torch.cross(rays_d_norm, v1, dim=-1)  # [N_rays, 3]
        v2 = F.normalize(v2, dim=-1, eps=1e-8)
        
        # Determine sideways direction (perpendicular to ray, in horizontal plane)
        # Project ray onto horizontal plane
        rays_d_horizontal = rays_d_norm.clone()
        rays_d_horizontal[:, 2] = 0
        rays_d_horizontal_norm = torch.norm(rays_d_horizontal, dim=-1, keepdim=True)
        
        # Handle vertical rays
        is_nearly_vertical = rays_d_horizontal_norm.squeeze(-1) < 1e-6
        rays_d_horizontal = torch.where(
            rays_d_horizontal_norm > 1e-6,
            rays_d_horizontal / rays_d_horizontal_norm,
            torch.tensor([1., 0., 0.], device=device).unsqueeze(0).expand_as(rays_d_horizontal)
        )
        
        # Sideways direction (perpendicular to horizontal ray direction)
        up_vector = torch.tensor([0., 0., 1.], device=device)
        sideways_dir = torch.cross(up_vector.unsqueeze(0).expand_as(rays_d_horizontal), 
                                  rays_d_horizontal, dim=-1)
        sideways_dir = F.normalize(sideways_dir, dim=-1, eps=1e-8)
        
        # Create basis: [ray_direction, sideways (elongated), perpendicular]
        basis = torch.stack([rays_d_norm, sideways_dir, up_vector.unsqueeze(0).expand_as(rays_d_norm)], dim=-1)
        
        # For nearly vertical rays, use computed perpendicular vectors instead
        basis[is_nearly_vertical, :, 1] = v1[is_nearly_vertical]
        basis[is_nearly_vertical, :, 2] = v2[is_nearly_vertical]
        
        # Ensure orthonormality
        basis = F.normalize(basis, dim=1, eps=1e-8)
        
        # Create diagonal variance matrices with sideways elongation
        diag_vars = torch.zeros(N_rays, N_samples, 3, 3, device=device)
        diag_vars[:, :, 0, 0] = t_var                                      # Along ray (unchanged)
        diag_vars[:, :, 1, 1] = r_var_base_lateral * elongation_factor    # Sideways lateral (elongated)
        diag_vars[:, :, 2, 2] = r_var_base_depth                           # Depth perpendicular (unchanged)
        
        # Transform to world coordinates: Cov = R @ diag(variances) @ R^T
        basis_expanded = basis.unsqueeze(1)  # [N_rays, 1, 3, 3]
        basis_T_expanded = basis_expanded.transpose(-2, -1)  # [N_rays, 1, 3, 3]
        
        # Batch matrix multiplication
        temp = torch.matmul(basis_expanded, diag_vars)  # [N_rays, N_samples, 3, 3]
        covs = torch.matmul(temp, basis_T_expanded)     # [N_rays, N_samples, 3, 3]
        
        # Final numerical stability checks
        covs = torch.where(torch.isnan(covs) | torch.isinf(covs), 
                          torch.tensor(1e-10, device=device), 
                          covs)
        
        return means, covs

    def integrated_positional_encoding_fast(means, covs, L_pos=10):
        """
        Vectorized integrated positional encoding for Gaussians.
        
        Args:
            means: [N_rays, N_samples, 3]
            covs: [N_rays, N_samples, 3, 3]
            L_pos: Number of frequency bands
        Returns:
            encoded: [N_rays, N_samples, 6*L_pos]
        """
        N_rays, N_samples, _ = means.shape
        device = means.device
        
        # Frequency bands [L_pos]
        # Frequency bands: 2^0, 2^1, ..., 2^(L_pos-1)
        freqs = 2. ** torch.linspace(0., L_pos-1, L_pos, device=device)
        
        # Broadcasting setup
        means_expanded = means.unsqueeze(-1)  # [N, S, 3, 1]
        freqs_expanded = freqs.view(1, 1, 1, -1)  # [1, 1, 1, L]
        
        # Extract diagonal covariances [N_rays, N_samples, 3]
        sigma_diag = torch.diagonal(covs, dim1=-2, dim2=-1)
        sigma_expanded = sigma_diag.unsqueeze(-1)  # [N, S, 3, 1]
        
        # Vectorized computation
        # Compute expected values of sin and cos for Gaussian distribution
        # E[sin(freq * X)] where X ~ N(mu, sigma^2)
        # = sin(freq * mu) * exp(-0.5 * freq^2 * sigma^2)
        freq_means = freqs_expanded * means_expanded  # [N, S, 3, L]
        exp_factor = torch.exp(-0.5 * (freqs_expanded ** 2) * sigma_expanded)
        
        sin_encoding = torch.sin(freq_means) * exp_factor  # [N, S, 3, L]
        cos_encoding = torch.cos(freq_means) * exp_factor  # [N, S, 3, L]
        
        # Efficient interleaving
        encoded = torch.stack([sin_encoding, cos_encoding], dim=-1)
        encoded = encoded.flatten(-3)  # Merge last 3 dims to [N, S, 6*L_pos]
        
        return encoded

    ###############################
    # Main rendering logic
    ###############################
    
    N_rays = ray_batch.shape[0]
    #print("N_rays", N_rays)
    #print(N_samples, "N_samples")
    
    # Extract ray origin, direction
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    #print("rays_o shape", rays_o.shape)
    #print("rays_d shape", rays_d.shape)
    
    # Extract unit-normalized viewing direction
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    
    # Extract lower, upper bound for ray distance
    bounds = ray_batch[..., 6:8].reshape(-1, 1, 2)
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]
    
    center, cx, cy, num_rays, angle, radius, radius2 = None, None, None, None, None, None, None
    if convex.config["use_convex_mode"]:
        center, cx, cy, num_rays_img, angle, radius, radius2 = convex.get_convex_settings()
        #print("num_rays", num_rays)
    
    # Decide where to sample/frustum along each ray
    delta_radius = (radius - radius2) / radius if convex.config["use_convex_mode"] else 1.0
    t_vals = torch.linspace(0., delta_radius, N_samples)
    #print("t_vals shape", t_vals.shape)
    
    if not convex.config['use_convex_mode']:
        # Standard mode: use all rays
        z_vals = torch.ones_like(rays_o[..., :1]) * t_vals
        z_vals = z_vals.expand(N_rays, N_samples)
    else:
        if convex.config['full_volume_mode']:
            num_rays = rays_o.shape[0]
        # Convex mode: create z_vals for subset of rays
        z_vals = t_vals.unsqueeze(0).expand(num_rays, N_samples)  # [num_rays, N_samples]
    
    #print("z_vals shape", z_vals.shape)
    #print(num_rays if convex.config['use_convex_mode'] else num_rays, N_samples, "num_rays, N_samples")

    # Compute Gaussians (standard or elongated)
    #print("voxel_radius_lateral", voxel_radius_lateral)
    #print("voxel_radius_depth", voxel_radius_depth)
    
    if use_elongation:
        #print("Using ELONGATED sideways Gaussians")
        means, covs = compute_elongated_sideways_gaussians(
            rays_o, rays_d, z_vals, 
            voxel_radius_lateral=voxel_radius_lateral,
            voxel_radius_depth=voxel_radius_depth,
            center_point=center_point, 
            max_elongation=max_elongation
        )
    else:
        #mode_str = "constant-size" if constant_gaussian_size else "distance-growing (mip-NeRF)"
        #print(f"Using STANDARD ellipsoidal conical frustum Gaussians ({mode_str})")
        means, covs = compute_conical_frustum_gaussians(
            rays_o, rays_d, z_vals, 
            voxel_radius_lateral=voxel_radius_lateral,
            voxel_radius_depth=voxel_radius_depth,
            constant_size=constant_gaussian_size
        )
    
    # Integrated positional encoding
    integrated_pe = integrated_positional_encoding_fast(
        means, covs, 
        L_pos=convex.config.get('multires', 10)
    )    
    distance = None
    
    # Evaluate model at each Gaussian
    if convex.config['use_convex_mode']:
        raw = network_query_fn(integrated_pe, rays_d, network_fn)  # [N_rays, N_samples, output_dim]
    else:
        raw = network_query_fn(integrated_pe, None, network_fn)
    
    #print("raw shape", raw.shape)
    
    # Transform model predictions to outputs
    ret = raw2outputs(raw, z_vals, distance)
    #print("ret keys", ret.keys())
    
    # Reshape outputs if needed
    if convex.config["use_convex_mode"]:
        # TODO: for potential future spherial or concentric circle sampling strategies, we may need to adjust this reshaping logic
        if convex.config.get('use_spherical_volume', False):
            meta = convex.config.get('_spherical_metadata', {})
            num_rays_img = meta.get('n_rays_azimuth', num_rays_img)
        elif convex.config.get('sampling_strategy', 'uniform_fan') == 'concentric_circles':
            num_rays_img = convex.config.get('_ray_metadata')['total_rays']
        
        for k, v in ret.items():
            ret[k] = ret[k].reshape(ret[k].shape[0], -1, num_rays_img, N_samples)
    
    #for k, v in ret.items():
    #    if isinstance(v, torch.Tensor):
    #        print(f"ret['{k}'] shape: {v.shape}")
    #    else:
    #        print(f"ret['{k}'] type: {type(v)}")
    
    return ret


# Convenience wrapper functions for backward compatibility
def render_rays_us_mip(ray_batch, 
                       network_fn, 
                       network_query_fn, 
                       N_samples, 
                       voxel_radius_lateral=0.173,
                       voxel_radius_depth=0.173, 
                       **kwargs):
    """Standard mip-NeRF rendering (conical frustums)."""
    return render_rays_us_mip_combined(
        ray_batch, network_fn, network_query_fn, N_samples,
        use_elongation=False,
        voxel_radius_lateral=voxel_radius_lateral,
        voxel_radius_depth=voxel_radius_depth,
        **kwargs
    )


def render_rays_us_mip_elongated(ray_batch, 
                                network_fn, 
                                network_query_fn, 
                                N_samples,
                                center_point=None, 
                                max_elongation=5.0,
                                voxel_radius_lateral=0.173, 
                                voxel_radius_depth=0.173, 
                                **kwargs):
    """Mip-NeRF rendering with elongated sideways Gaussians."""
    return render_rays_us_mip_combined(
        ray_batch, network_fn, network_query_fn, N_samples,
        use_elongation=True,
        center_point=center_point,
        max_elongation=max_elongation,
        voxel_radius_lateral=voxel_radius_lateral,
        voxel_radius_depth=voxel_radius_depth,
        **kwargs
    )