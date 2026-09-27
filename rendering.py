import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

from sampling import *


def cumsum_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    r"""Mimick functionality of tf.math.cumsum(..., exclusive=True), as it isn't available in PyTorch.

    Args:
      tensor (torch.Tensor): Tensor whose cumsum (cumulative product, see `torch.cumsum`) along dim=-1
        is to be computed.

    Returns:
      cumsum (torch.Tensor): cumsum of Tensor along dim=-1, mimiciking the functionality of
        tf.math.cumsum(..., exclusive=True) (see `tf.math.cumsum` for details).
    """
    dim = -1
    # Compute regular cumsum first (this is equivalent to `tf.math.cumsum(..., exclusive=False)`).
    cumsum = torch.cumsum(tensor, dim)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumsum = torch.roll(cumsum, 1, dim)
    # Replace the first element by "0" as this is what tf.cumsum(..., exclusive=True) does.
    cumsum[..., 0] = 0.0

    return cumsum


def gaussian_kernel(size: int, mean: float, std: float):
    delta_t = 1
    x_cos = np.array(list(range(-size, size + 1)), dtype=np.float32)
    x_cos *= delta_t

    d1 = torch.distributions.Normal(mean, std * 3)
    d2 = torch.distributions.Normal(mean, std)
    vals_x = d1.log_prob(
        torch.arange(-size, size + 1, dtype=torch.float32) * delta_t
    ).exp()
    vals_y = d2.log_prob(
        torch.arange(-size, size + 1, dtype=torch.float32) * delta_t
    ).exp()

    gauss_kernel = torch.einsum("i,j->ij", vals_x, vals_y)

    return gauss_kernel / torch.sum(gauss_kernel).reshape(1, 1)


def gaussian_kernel_3d(size: int, mean: float, std: float):
    delta_t = 1
    x_cos = np.array(list(range(-size, size + 1)), dtype=np.float32)
    x_cos *= delta_t

    d1 = torch.distributions.Normal(mean, std * 3)
    d2 = torch.distributions.Normal(mean, std)

    vals_x = d1.log_prob(
        torch.arange(-size, size + 1, dtype=torch.float32) * delta_t
    ).exp()
    vals_y = d2.log_prob(
        torch.arange(-size, size + 1, dtype=torch.float32) * delta_t
    ).exp()
    vals_z = d2.log_prob(
        torch.arange(-size, size + 1, dtype=torch.float32) * delta_t
    ).exp()

    # Create 3D Gaussian kernel by taking the outer product
    gauss_kernel_3d = torch.einsum("i,j,k->ijk", vals_x, vals_y, vals_z)

    # Normalize the kernel so that the sum of all elements equals 1
    gauss_kernel_3d = gauss_kernel_3d / torch.sum(gauss_kernel_3d)

    return gauss_kernel_3d


g_kernel = gaussian_kernel(3, 0.0, 1.0).float().to(device)

size = 3
mean = 0.0
std = 1.0
g_kernel3D = gaussian_kernel_3d(size, mean, std).float().to(device)


def render_method_3(raw):
    def raw2attention(raw, dists):
        return torch.exp(-raw * dists)

    raw = raw[None, None, ...]

    batch_size, C, W, H, maps = raw.shape

    t_vals = torch.linspace(0.0, 1.0, H, device=raw.device)
    z_vals = t_vals.expand(batch_size, W, -1)  # * 2

    # Compute 'distance' between each integration location (sample/frustum) along a ray.
    dists = torch.abs(z_vals[..., :-1, None] - z_vals[..., 1:, None])
    dists = torch.squeeze(dists)
    dists = torch.cat([dists, dists[:, -1, None]], dim=-1)
    #dists = torch.cat([
    #        z_vals[..., 1:] - z_vals[..., :-1],
    #        torch.full((*z_vals.shape[:-1], 1), 1e10, device="cuda")
    #    ], dim=-1)
    #dists = torch.squeeze(dists) 

    # ATTENUATION
    attenuation_coeff = torch.abs(raw[..., 0])
    attenuation = raw2attention(attenuation_coeff, dists)
    log_attenuation = torch.log(attenuation + 1e-12)
    log_attenuation_total = cumsum_exclusive(log_attenuation)
    attenuation_total = torch.exp(log_attenuation_total)

    # REFLECTION
    # reflection_coeff = torch.zeros_like(raw[..., 0])
    reflection_coeff = torch.sigmoid(raw[..., 1])
    reflection_transmission = 1.0 - reflection_coeff
    # reflection_transmission = raw2reflection(reflection_coeff)
    log_reflection_transmission = torch.log(reflection_transmission + 1e-12)
    log_reflection_total = cumsum_exclusive(log_reflection_transmission)
    reflection_total = torch.exp(log_reflection_total)

    # BACKSCATTERING
    # density_coeff = torch.sigmoid(raw[..., 2])
    density_coeff = torch.ones_like(reflection_coeff) * 0.75
    scatter_density_distribution = RelaxedBernoulli(
        temperature=0.1, probs=density_coeff
    )
    scatterers_density = scatter_density_distribution.sample()
    
    amplitude = torch.sigmoid(raw[..., 2])
    
    if convex.config['use_mip']: 
        # Use MVG mode, no PSF convolution
        scatterers_map = amplitude
    else:
        scatterers_map = scatterers_density * amplitude
    

    if 'g_kernel3D' in globals() and convex.config['full_volume_mode'] and not convex.config['use_mip']:
        # Reshape for 3D convolution
        if convex.config["use_convex_mode"]:
            center, cx, cy, num_rays_img, angle, radius, radius2 = convex.get_convex_settings()
        
        scatterers_3d = scatterers_map.view(1, W//num_rays_img, num_rays_img, H)

        psf_scatter = F.conv3d(
            scatterers_3d, 
            g_kernel3D[None, None, ...], 
            stride=1, 
            padding="same"
        )
        psf_scatter = psf_scatter.view(W, H)
        #print("psf_scatter shape:", psf_scatter.shape)    
    elif convex.config['use_mip']:
        # No PSF convolution for Multivariate Gaussian estimation
        psf_scatter = scatterers_map 
    else:
        psf_scatter = scatterers_map
        psf_scatter = F.conv2d(
            scatterers_map, g_kernel[None, None, ...], stride=1, padding="same"
        )
    psf_scatter = amplitude #no psf for all
    # Compute remaining intensity at a point n
    confidence_maps = torch.sigmoid(attenuation_total * reflection_total)
    confidence_maps = attenuation_total * reflection_total

    # Compute backscattering and reflection parts of the final echo
    b = confidence_maps * psf_scatter
    r = confidence_maps * reflection_coeff

    # Compute the final echo
    amplification_constant = torch.tensor(np.pi)
    alpha_amplification = lambda x: torch.log(
        torch.tensor(1.0) + amplification_constant * x
    ) * torch.log(torch.tensor(1.0) + amplification_constant)
    r_amplified = alpha_amplification(r)
    intensity_map = b + r_amplified

    return {
        "intensity_map": intensity_map,
        "attenuation_coeff": attenuation_coeff,
        "reflection_coeff": reflection_coeff,
        "attenuation_total": attenuation_total,
        "reflection_total": reflection_total,
        "scatterers_density": scatterers_density,
        "scatterers_density_coeff": density_coeff,
        "scatter_amplitude": amplitude,
        "b": b,
        "r": r,
        "confidence_maps": confidence_maps,
        "r_amplified": r_amplified,
    }


def render_method_ultra_nerf(raw):
    print(raw.shape, "raw shape in render_method_ultra_nerf")
    
    def raw2attention(raw, dists):
        return torch.exp(-raw * dists)

    raw = raw[None, None, ...]

    batch_size, C, W, H, maps = raw.shape

    t_vals = torch.linspace(0.0, 1.0, H, device=raw.device)
    z_vals = t_vals.expand(batch_size, W, -1)  # * 2

    # Compute 'distance' between each integration location (sample/frustum) along a ray.
    dists = torch.abs(z_vals[..., :-1, None] - z_vals[..., 1:, None])
    dists = torch.squeeze(dists)
    dists = torch.cat([dists, dists[:, -1, None]], dim=-1)

    # ATTENUATION
    attenuation_coeff = torch.abs(raw[..., 0])
    attenuation = raw2attention(attenuation_coeff, dists)
    #attenuation = attenuation.permute(0, 1, 3, 2)
    log_attenuation = torch.log(attenuation + 1e-12)
    log_attenuation_total = cumsum_exclusive(log_attenuation)
    attenuation_total = torch.exp(log_attenuation_total)
    #attenuation_total = attenuation_total.permute(0, 1, 3, 2)

    # REFLECTION
    # reflection_coeff = torch.zeros_like(raw[..., 0])
    prob_border = torch.sigmoid(raw[..., 2])
    b_prob_dist = RelaxedBernoulli(temperature=0.1, probs=prob_border)
    b_prob = b_prob_dist.sample()
    reflection_coeff = torch.sigmoid(raw[..., 1])
    reflection_transmission = 1.0 - reflection_coeff * b_prob
    #reflection_transmission = reflection_transmission.permute(0, 1, 3, 2)
    log_reflection_transmission = torch.log(reflection_transmission + 1e-12)
    log_reflection_total = cumsum_exclusive(log_reflection_transmission)
    reflection_total = torch.exp(log_reflection_total)
    #reflection_total = reflection_total.permute(0, 1, 3, 2)

    # BACKSCATTERING
    density_coeff = torch.sigmoid(raw[..., 3])
    # density_coeff = torch.ones_like(reflection_coeff) * 0.75
    scatter_density_distribution = RelaxedBernoulli(
        temperature=0.1, probs=density_coeff
    )
    scatterers_density = scatter_density_distribution.sample()
    
    amplitude = torch.sigmoid(raw[..., 4])
    
    if convex.config['use_mip']:
        # Use MVG mode, no PSF convolution
        scatterers_map = amplitude
    else:
        scatterers_map = scatterers_density * amplitude
    

    if 'g_kernel3D' in globals() and convex.config['full_volume_mode'] and not convex.config['use_mip']:
        # Reshape for 3D convolution
        if convex.config["use_convex_mode"]:
            center, cx, cy, num_rays_img, angle, radius, radius2 = convex.get_convex_settings()
        
        scatterers_3d = scatterers_map.view(1, W//num_rays_img, num_rays_img, H)

        psf_scatter = F.conv3d(
            scatterers_3d, 
            g_kernel3D[None, None, ...], 
            stride=1, 
            padding="same"
        )
        psf_scatter = psf_scatter.view(W, H)
        #print("psf_scatter shape:", psf_scatter.shape)    
    elif convex.config['use_mip']:
        psf_scatter = scatterers_map
    else:
        psf_scatter = scatterers_map
        psf_scatter = F.conv2d(
            scatterers_map, g_kernel[None, None, ...], stride=1, padding="same"
        )
    # psf_scatter = torch.squeeze(psf_scatter)
    psf_scatter = amplitude #no psf for all
    # Compute remaining intensity at a point n
    confidence_maps = attenuation_total * reflection_total

    # Compute backscattering and reflection parts of the final echo
    b = confidence_maps * psf_scatter
    r = confidence_maps * reflection_coeff

    intensity_map = b + r

    return {
        "intensity_map": intensity_map,
        "attenuation_coeff": attenuation_coeff,
        "reflection_coeff": reflection_coeff,
        "attenuation_total": attenuation_total,
        "reflection_total": reflection_total,
        "scatterers_density": scatterers_density,
        "scatterers_density_coeff": density_coeff,
        "scatter_amplitude": amplitude,
        "b": b,
        "r": r,
        "confidence_maps": confidence_maps,
    }


def render_rays_us(
    ray_batch,
    network_fn,
    network_query_fn,
    N_samples,
    lindisp=False,
    **kwargs,
):
    """Volumetric rendering.

    Args:
    ray_batch: Tensor of shape [batch_size, ...]. We define rays and do not sample.

    Returns:
    Rendered outputs.
    """

    def raw2outputs(raw, z_vals, distance=None):
        """Transforms model's predictions to semantically meaningful values."""        
        if raw.shape[-1] == 5:
            ret = render_method_ultra_nerf(raw)
        elif raw.shape[-1] == 3:
            ret = render_method_3(raw)
        else:
            raise NotImplementedError
        return ret

    ###############################
    # Batch size
    N_rays = ray_batch.shape[0]

    # Extract ray origin, direction
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each

    # Extract unit-normalized viewing direction
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None

    # Extract lower, upper bound for ray distance
    bounds = ray_batch[..., 6:8].reshape(-1, 1, 2)
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]
    
    center, cx, cy, num_rays, angle, radius, radius2 = None, None, None, None, None, None, None
    if convex.config["use_convex_mode"]:
        center, cx, cy, num_rays_img, angle, radius, radius2 = convex.get_convex_settings()

    # Decide where to sample along each ray
    delta_radius = (radius - radius2) / radius
    t_vals = torch.linspace(0., delta_radius, N_samples)

    if not convex.config['use_convex_mode']:
        # Standard mode: use all rays
        z_vals = torch.ones_like(rays_o[..., :1]) * t_vals
        z_vals = z_vals.expand(N_rays, N_samples)
    else:
        if convex.config['full_volume_mode']:
            num_rays = rays_o.shape[0]
        num_rays = rays_o.shape[0]    
        # Convex mode: create z_vals for subset of rays
        z_vals = t_vals.unsqueeze(0).expand(num_rays, N_samples)  # [num_rays, N_samples]
    
    #print(num_rays if convex.config['use_convex_mode'] else num_rays, N_samples, "num_rays, N_samples")

    # Points in space to evaluate model at
    origin = rays_o[..., None, :]
    step = rays_d[..., None, :] * z_vals[..., :, None]
    pts = step + origin
    #print("pts shape", pts.shape)  # [N_rays, N_samples, 3]
    #print("origin shape", origin.shape)  # [N_rays, 1, 3]
    #print("step shape", step.shape)  # [N_rays, N_samples, 3]

    # Evaluate model at each location
    if convex.config['use_convex_mode']:
        raw = network_query_fn(pts, rays_d, network_fn)  # [N_rays, N_samples, 5]
    else:
        raw = network_query_fn(pts, None, network_fn)

    ret = raw2outputs(raw, z_vals)
    
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
    # if retraw:
    #     ret['raw'] = raw

    # In PyTorch, there is no direct equivalent of tf.debugging.check_numerics
    # You might want to implement a custom check or use torch.isnan or torch.isinf
    # for checking invalid numerics.


def render_rays_us_with_reconstruction_pts(
        ray_batch,
        network_fn,
        network_rec,
        network_query_fn,
        network_query_fn_rec,
        N_samples,
        pts,
        lindisp=False,
        **kwargs,
):
    """Volumetric rendering.

    Args:
    ray_batch: Tensor of shape [batch_size, ...]. We define rays and do not sample.

    Returns:
    Rendered outputs.
    """
    raise NotImplementedError
    #
    # def raw2outputs(raw, z_vals=None):
    #     """Transforms model's predictions to semantically meaningful values."""
    #     ret = render_method_3(
    #         raw
    #     )  # Assuming render_method_3 is defined elsewhere
    #     # ret = rendering(raw, z_vals)
    #     return ret
    # # Evaluate model at each point
    # raw = network_query_fn(pts, network_fn)  # [N_rays, N_samples , 3]
    # ret = raw2outputs(raw)
    #
    # # input_reconstruction = torch.cat([pts.detach().clone(), raw.detach().clone()], dim=-1)
    # #
    # # ret_reconstruction = network_query_fn_rec(input_reconstruction, network_rec)
    # #
    # # ret["reconstruction"] = ret_reconstruction.permute(2, 1, 0)[None, ...]
    # # ret['pts'] = pts
    #
    # # if retraw:
    # #     ret['raw'] = raw
    #
    # return ret
    
def render_rays_us_with_reconstruction(
        ray_batch,
        network_fn,
        network_rec,
        network_query_fn,
        network_query_fn_rec,
        N_samples,
        lindisp=False,
        **kwargs,
):
    """Volumetric rendering.

    Args:
    ray_batch: Tensor of shape [batch_size, ...]. We define rays and do not sample.

    Returns:
    Rendered outputs.
    """
    raise NotImplementedError
#
#     def raw2outputs(raw, z_vals):
#         """Transforms model's predictions to semantically meaningful values."""
#         ret = render_method_3(
#             raw
#         )  # Assuming render_method_3 is defined elsewhere
#         # ret = rendering(raw, z_vals)
#         return ret
#
#     ###############################
#     # Batch size
#     N_rays = ray_batch.shape[0]
#
#     # Extract ray origin, direction
#     rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
#
#     # Extract lower, upper bound for ray distance
#     bounds = ray_batch[..., 6:8].reshape(-1, 1, 2)
#     near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]
#
#     # Decide where to sample along each ray
#     t_vals = torch.linspace(0.0, 1.0, N_samples).to(ray_batch.device)
#     if not lindisp:
#         z_vals = near * (1.0 - t_vals) + far * t_vals
#     else:
#         z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
#     z_vals = z_vals.expand(N_rays, N_samples)
#
#     # Points in space to evaluate model at
#     origin = rays_o.unsqueeze(-2)
#     step = rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1)
#
#     pts = step + origin
#     # Evaluate model at each point
#     raw = network_query_fn(pts, network_fn)  # [N_rays, N_samples , 5]
#     ret = raw2outputs(raw, z_vals)
#
#     # input_reconstruction = torch.cat([pts.detach().clone(), raw.detach().clone()], dim=-1)
#     #
#     # ret_reconstruction = network_query_fn_rec(input_reconstruction, network_rec)
#     #
#     # ret["reconstruction"] = ret_reconstruction.permute(2, 1, 0)[None, ...]
#     # ret['pts'] = pts
#
#     # if retraw:
#     #     ret['raw'] = raw
#
#     return ret


def render_radial_paths(
    origins: np.ndarray,
    directions: np.ndarray,
    metadata: dict,
    paths: list,
    path_metadata: dict,
    pose: np.ndarray,
    network_fn,
    network_query_fn,
    N_samples=None,
    chunk=1024*32,
    **kwargs
):
    """
    Render ultrasound image using radial path integration with depth sampling.
    
    This is the main rendering function for the radial tree-ring sampling strategy.
    It samples along depth for each ray (like render_rays_us) and also integrates
    physics along radial paths (outer to inner circles).
    
    Args:
        origins (np.ndarray): Ray origin points from concentric circles, shape [N_total, 2]
        directions (np.ndarray): Ray directions, shape [N_total, 2]
        metadata (dict): Metadata from define_rays_concentric_circles()
        paths (list): List of paths (each path is list of global indices)
        path_metadata (dict): Metadata about paths from build_radial_paths_from_concentric()
        pose (np.ndarray): Probe pose transformation matrix [3, 4] or [4, 4]
        network_fn: Neural network function
        network_query_fn: Network query wrapper function
        N_samples (int): Number of depth samples per ray (from convex_n_samples config)
        chunk (int): Batch size for network queries
        
    Returns:
        dict: Rendering outputs with keys matching render_rays_us format:
            - 'intensity_map': shape [1, 1, N_rays, N_samples]
            - 'attenuation_total': shape [1, 1, N_rays, N_samples]
            - 'reflection_total': shape [1, 1, N_rays, N_samples]
            - 'b': Backscattering contribution, shape [1, 1, N_rays, N_samples]
            - 'r': Reflection contribution, shape [1, 1, N_rays, N_samples]
            - 'confidence_maps': shape [1, 1, N_rays, N_samples]
    """
    raise NotImplementedError
#