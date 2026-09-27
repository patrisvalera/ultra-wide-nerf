"""
NeRF model and helper functions for training and rendering.

Functions such as:
- get_embedder: for original NeRF positional encoding
- get_rays_us_linear: for generating rays in the linear case (non-convex mode)
- batchify: for batching inputs to avoid OOM
- run_network: for running the NeRF MLP on inputs
- batchify_rays: for rendering rays in batches (both 2D and 3D cases, with support for convex mode and full volume mode)
- create_nerf: for creating the NeRF model and loading checkpoints
- create_barf: for creating the BARF model and loading checkpoints
- render_us: main method for rendering rays using the Ultra-NeRF model, 
            with support for both linear and convex modes, 2D and 3D cases, 
            and debug visualization of rays (here based on the parementers, the actual renddering 
            is called in batchify_rays with the appropriate renderer)
- compute_loss: for computing the loss between rendered outputs and targets, with support for both L2 and SSIM losses, and handling of 3D volumes for SSIM computation
- compute_reg: for computing regularization terms such as LCC penalty, with support for both 3D and 4D tensors
- visualize_rays: for visualizing rays in 3D space (used in debug visualization in render_us)
"""

import os
import random

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from model import NeRF, BARF, PoseRefine, Reconstruction
from rendering import render_rays_us, render_rays_us_with_reconstruction, render_rays_us_with_reconstruction_pts
from sampling import *
from render_rays_us_mip_combined import render_rays_us_mip_combined

def set_seed(seed):
    """Seed numpy, torch and random for reproducible runs; a negative seed disables seeding."""
    if seed < 0:
        print("Random seed disabled (negative --random_seed)")
        return
    print(f"Setting random seed: {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10.0 * torch.log(x) / torch.log(torch.Tensor([10.0]))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]

        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d
            
        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]
        B = self.kwargs["B"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(
                0.0, max_freq, steps=N_freqs, device=self.kwargs["device"]
            )
        else:
            freq_bands = torch.linspace(
                2.0**0.0, 2.0**max_freq, steps=N_freqs, device=self.kwargs["device"]
            )

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                if B is not None:
                    embed_fns.append(
                        lambda x, p_fn=p_fn, freq=freq, B=B: p_fn(
                            x @ torch.transpose(B, 0, 1) * freq
                        )
                    )
                    out_dim += d
                    out_dim += B.shape[1]
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, device, i=0, b=0, input_dim=3):
    if i == -1:
        return nn.Identity(), 3

    if b != 0:
        B = torch.randn(size=(b, 3))
    else:
        B = None

    embed_kwargs = {
        "include_input": True,
        "input_dims": input_dim,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
        "B": B,
        "device": device,
    }
    
    embedder_obj = Embedder(**embed_kwargs)
        
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


def get_rays_us_linear(H, W, sw, sh, c2w):
    t = c2w[:3, -1]
    R = c2w[:3, :3]
    x = torch.arange(-W / 2, W / 2, dtype=torch.float32, device=c2w.device) * sw
    y = torch.zeros_like(x)
    z = torch.zeros_like(x)

    origin_base = torch.stack([x, y, z], dim=1).to(c2w.device)

    origin_rotated = R @ origin_base.transpose(
        0, 1
    )  # THIS WAS HADAMARD PRODUCT IN THE ORIGINAL CODE !!!
    ray_o_r = origin_rotated.transpose(0, 1)
    rays_o = ray_o_r + t

    dirs_base = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=c2w.device)
    dirs_r = R @ dirs_base
    rays_d = dirs_r.expand_as(rays_o)

    return rays_o, rays_d


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat(
            [fn(inputs[i : i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0
        )

    return ret


def run_network(inputs, fn, embed_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'."""
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

    if convex.config['use_mip']:
        # Embedder implemented in Multivariate Gaussian computation (render_rays_us_mip_combined)
        embedded = inputs_flat 
    else:
        embedded = embed_fn(inputs_flat)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(
        outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]
    )
    return outputs


def run_barf_network(inputs, fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'."""
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

    outputs_flat = batchify(fn, netchunk)(inputs_flat)
    outputs = torch.reshape(
        outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]
    )
    return outputs


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}
    
    if kwargs["network_rec"] is not None:
        renderer = render_rays_us_with_reconstruction
        
    elif convex.config['use_mip']:
        # Multivariate Gaussian with cone casting and rendering, implemented in render_rays_us_mip_combined.py
        renderer = render_rays_us_mip_combined 
        
        # Add elongation parameters from config to kwargs
        if 'use_elongation' not in kwargs:
            kwargs['use_elongation'] = convex.config.get('use_elongation', False)
        if 'max_elongation' not in kwargs:
            kwargs['max_elongation'] = convex.config.get('max_elongation', 5.0)
        if 'voxel_radius_lateral' not in kwargs:
            kwargs['voxel_radius_lateral'] = convex.config.get('mip_voxel_radius_lateral', 0.173)
        if 'voxel_radius_depth' not in kwargs:
            kwargs['voxel_radius_depth'] = convex.config.get('mip_voxel_radius_depth', 0.173)
        # Center point for elongation (if use_convex_mode, use the convex center)
        if 'center_point' not in kwargs and convex.config.get('use_convex_mode', False):
            cx, cy = convex.config.get('convex_center', (0, 0))
            kwargs['center_point'] = torch.tensor([cx, cy, 0.0], dtype=torch.float32)
    else:
        renderer = render_rays_us
        
    try:
        if kwargs["pts"] is not None:
            renderer = render_rays_us_with_reconstruction_pts
            rays_flat = kwargs['pts']
    except KeyError:
        pass
    
    for i in range(0, rays_flat.shape[0], chunk):
        #print("Batching rays from", i, "to", i + chunk, "of", rays_flat.shape[0])
        ret = renderer(rays_flat[i : i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    # Handle concatenation differently for full_volume_mode + convex_mode
    if convex.config.get('full_volume_mode', False) and convex.config.get('use_convex_mode', False):
        # In volume mode, each chunk corresponds to depth slices
        # Concatenate along the depth dimension (dim=1)
        # Expected shape after rendering: [batch, depth, height, width]
        all_ret = {k: torch.cat(all_ret[k], dim=1) for k in all_ret}
    else:
        # Standard concatenation along batch dimension (dim=0)
        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    
    return all_ret


def create_nets_for_reconstruction(args, device, mode="train"):
    """Instantiate NeRF's MLP model."""
    embed_fn, input_ch = get_embedder(args.multires, device, args.i_embed)
    if args.rec_only_theta or args.rec_only_occ:
        embed_fn_rec, input_ch_rec = get_embedder(args.multires, device, args.i_embed, input_dim=3)
    else:
        embed_fn_rec, input_ch_rec = get_embedder(args.multires, device, args.i_embed, input_dim=6)
    output_ch = args.output_ch
    skips = [4]
    model = NeRF(
        D=args.netdepth,
        W=args.netwidth,
        input_ch=input_ch,
        output_ch=output_ch,
        skips=skips,
    ).to(device)

    grad_vars = list(model.parameters())

    # network_query_fn = lambda inputs, network_fn: run_network(
    #     inputs, network_fn, embed_fn=embed_fn, netchunk=args.netchunk
    # )
    #
    # network_query_fn_rec = lambda inputs, network_fn: run_network(
    #     inputs, network_fn, embed_fn=embed_fn_rec, netchunk=args.netchunk
    # )

    def network_query_fn(inputs, view_dirs, network_fn):
        """
            q(t) function, takes query point
        """
        return run_network(
            inputs, network_fn,
            embed_fn=embed_fn,
            netchunk=args.netchunk)

    if args.reconstruction:
        model_rec = Reconstruction(
            D=args.netdepth,
            W=args.netwidth,
            input_ch= input_ch_rec,
            output_ch=1,
            skips=skips,
        ).to(device)

        for i, l in enumerate(model.pts_linears):
            if i < 6:
                for param in l.parameters():
                    param.requires_grad = False

        grad_vars_reg = list(model_rec.parameters())
        optimizer_reg = torch.optim.Adam(params=grad_vars_reg, lr=args.lrate, betas=(0.9, 0.999))
    else:
        model_rec = None
        optimizer_reg = None
    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))


    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != "None":
        ckpts = [args.ft_path]
    else:
        ckpts = [
            os.path.join(basedir, expname, f)
            for f in sorted(os.listdir(os.path.join(basedir, expname)))
            if "tar" in f
        ]

    print("Found ckpts", ckpts)
    if len(ckpts) > 0:
        ckpt_path = ckpts[-1]
        print("Reloading from", ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt["global_step"]
        optimizer_reg.load_state_dict(ckpt["optimizer_rec_state_dict"])

        # remove paramerts "views_linears.0.weight", "views_linears.0.bias" from state_dict
        # if exists, this is for compatibility with the old code

        new_state_dict_rec = {}
        for k, v in ckpt["network_rec_state_dict"].items():
            if "views_linears.0" not in k:
                new_state_dict_rec[k] = v
        model_rec.load_state_dict(new_state_dict_rec)
    ckpts_nerf = [
        os.path.join(basedir, args.expname_nerf, f)
        for f in sorted(os.listdir(os.path.join(basedir, args.expname_nerf)))
        if "tar" in f
    ]
    nerf_weights = torch.load(ckpts_nerf[-1])
    # Load NeRF model
    new_state_dict = {}
    for k, v in nerf_weights["network_fn_state_dict"].items():
        if "views_linears.0" not in k:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)


    ##########################

    render_kwargs_train = {
        "network_query_fn": network_query_fn,
        "network_query_fn_rec": network_query_fn_rec,
        "N_samples": args.N_samples,
        "network_fn": model,
        "network_rec": model_rec
    }

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test["perturb"] = False
    render_kwargs_test["raw_noise_std"] = 0.0

    return render_kwargs_train, render_kwargs_test, start, optimizer, optimizer_reg


def create_nerf(args, device, mode="train"):
    """Instantiate NeRF's MLP model."""
    embed_fn, input_ch = get_embedder(args.multires, device, args.i_embed)
    
    if not convex.config['use_mip']:
        print("Using standard positional encoding, input channels:", input_ch)
    else:
        print(f"=== Mip-NeRF Rendering (elongation={'ON' if convex.config.get('use_elongation', False) else 'OFF'}) ===")
        print(f"Voxel radius lateral: {convex.config.get('mip_voxel_radius_lateral', 0.173)}, depth: {convex.config.get('mip_voxel_radius_depth', 0.173)}")
    
    if args.rec_only_theta or args.rec_only_occ:
        embed_fn_rec, input_ch_rec = get_embedder(args.multires, device, args.i_embed, input_dim=3)
    else:
        embed_fn_rec, input_ch_rec = get_embedder(args.multires, device, args.i_embed, input_dim=6)
    
    if convex.config['use_mip'] and convex.config['full_volume_mode']:
        input_ch = args.multires * args.i_embed
    
    print("Input channels:", input_ch, "Output channels:", args.output_ch)
    
    output_ch = args.output_ch
    skips = [4]
    model = NeRF(
        D=args.netdepth,
        W=args.netwidth,
        input_ch=input_ch,
        output_ch=output_ch,
        skips=skips,
    ).to(device)

    grad_vars = list(model.parameters())

    # network_query_fn = lambda inputs, network_fn: run_network(
    #     inputs, network_fn, embed_fn=embed_fn, netchunk=args.netchunk
    # )
    #
    # network_query_fn_rec = lambda inputs, network_fn: run_network(
    #     inputs, network_fn, embed_fn=embed_fn_rec, netchunk=args.netchunk
    # )

    def network_query_fn(inputs, view_dirs, network_fn):
        """
            q(t) function, takes query point
        """
        return run_network(
            inputs, network_fn,
            embed_fn=embed_fn,
            netchunk=args.netchunk)

    if args.reconstruction:
        model_rec = Reconstruction(
            D=args.netdepth,
            W=args.netwidth,
            input_ch= input_ch_rec,
            output_ch=1,
            skips=skips,
        ).to(device)

        grad_vars_reg = list(model_rec.parameters())
        optimizer_reg = torch.optim.Adam(params=grad_vars_reg, lr=args.lrate, betas=(0.9, 0.999))
    else:
        model_rec = None
        optimizer_reg = None
        
    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != "None":
        ckpts = [args.ft_path]
    else:
        ckpts = [
            os.path.join(basedir, expname, f)
            for f in sorted(os.listdir(os.path.join(basedir, expname)))
            if "tar" in f
        ]

    print("Found ckpts", ckpts)
    if len(ckpts) > 0:
        ckpt_path = ckpts[-1]
        print("Reloading from", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)

        start = ckpt["global_step"]

        if mode == "train":
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if args.reconstruction:
                optimizer_reg.load_state_dict(ckpt["optimizer_rec_state_dict"])

        # Load model directly without filtering views_linears parameters
        model.load_state_dict(ckpt["network_fn_state_dict"])
        if args.reconstruction:
            model_rec.load_state_dict(ckpt["network_rec_state_dict"])


    ##########################

    render_kwargs_train = {
        "network_query_fn": network_query_fn,
        "N_samples": args.N_samples,
        "network_fn": model,
        "network_rec": model_rec
    }
    print("Render kwargs train:", render_kwargs_train)
    print("Render kwargs train network_fn:", render_kwargs_train['network_fn'])
    print("Render kwargs train network_rec:", render_kwargs_train['network_rec'])
    print("Render kwargs train network_query_fn:", render_kwargs_train['network_query_fn'])
    print("Embed fn:", embed_fn)
    print("Input ch:", input_ch)
    print("Output ch:", output_ch)
    print("NERF model:", model)
    print("Number of NeRF parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test["perturb"] = False
    render_kwargs_test["raw_noise_std"] = 0.0

    return render_kwargs_train, render_kwargs_test, start, optimizer, optimizer_reg, embed_fn


def create_barf(poses: torch.Tensor, args, device, mode="train"):
    """Instantiate BARF's MLP model."""

    output_ch = args.output_ch
    skips = [4]
    input_ch = 3

    model = BARF(
        D=args.netdepth,
        W=args.netwidth,
        input_ch=input_ch,
        output_ch=output_ch,
        skips=skips,
        L=args.L,
    ).to(device)

    pose_refine = PoseRefine(poses=poses, mode=mode).to(device)

    network_query_fn = lambda inputs, network_fn: run_barf_network(
        inputs, network_fn, netchunk=args.netchunk
    )

    # Create optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=args.lrate, betas=(0.9, 0.999)
    )

    pose_optim = torch.optim.Adam(pose_refine.parameters(), args.pose_lr)
    gamma = (args.pose_lr_end / args.pose_lr) ** (1.0 / args.n_iters)
    pose_sched = torch.optim.lr_scheduler.ExponentialLR(pose_optim, gamma)

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != "None":
        ckpts = [args.ft_path]
    else:
        ckpts = [
            os.path.join(basedir, expname, f)
            for f in sorted(os.listdir(os.path.join(basedir, expname)))
            if "tar" in f
        ]

    print("Found ckpts", ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print("Reloading from", ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt["global_step"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        pose_optim.load_state_dict(ckpt["pose_optim_state_dict"])
        pose_sched.load_state_dict(ckpt["pose_sched_state_dict"])

        # Load model
        model.load_state_dict(ckpt["network_fn_state_dict"])
        pose_refine.load_state_dict(ckpt["pose_refine_state_dict"])

    ##########################

    render_kwargs_train = {
        "network_query_fn": network_query_fn,
        "N_samples": args.N_samples,
        "network_fn": model,
        "pose_refine": pose_refine,
        "ckpt": ckpt if len(ckpts) > 0 else None,
    }

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}

    return (
        render_kwargs_train,
        render_kwargs_test,
        start,
        optimizer,
        pose_optim,
        pose_sched,
    )


def render_us(
    H,
    W,
    sw,
    sh,
    chunk=1024 * 32,
    rays=None,
    c2w=None,
    near=0.0,
    far=55.0 * 0.001,
    **kwargs
):
    assert rays is not None or c2w is not None
    rays_o = None
    rays_d = None
    
    if c2w is not None:
        # Linear ray generation case (for non-convex mode) -- full sampling
        if not convex.config['use_convex_mode']:
            rays_o, rays_d = get_rays_us_linear(H, W, sw, sh, c2w)

            for c in c2w:
                if rays_o is None:
                    rays_o, rays_d = get_rays_us_linear(H, W, sw, sh, c)
                else:
                    o, d = get_rays_us_linear(H, W, sw, sh, c)
                    rays_o = torch.concatenate((rays_o, o))
                    rays_d = torch.concatenate((rays_d, d))
        else:
            # 3D volume case (Multi-planar stack mode)
            if c2w.dim() > 3:
                for c_batch in c2w:
                    for c in c_batch:
                        if rays_o is None:
                            rays_o, rays_d = get_rays_us_convex(sw, sh, c)
                        else:
                            o, d = get_rays_us_convex(sw, sh, c)
                            rays_o = torch.cat((rays_o, o))
                            rays_d = torch.cat((rays_d, d))
            else: 
                for c in c2w:
                    rays_o, rays_d = get_rays_us_convex(sw, sh, c)

    else:
        # Use provided ray batch
        rays_o, rays_d = rays
        
    sh = rays_d.shape  # [..., 3]

    # Create ray batch
    rays_o = rays_o.reshape(-1, 3).float()
    rays_d = rays_d.reshape(-1, 3).float()
    near = near * torch.ones_like(rays_d[..., :1])
    far = far * torch.ones_like(rays_d[..., :1])
    #print("Rays origin shape:", rays_o.shape, "Rays direction shape:", rays_d.shape)
    #print("Near shape:", near.shape, "Far shape:", far.shape)
    
    rays = torch.cat([rays_o, rays_d, near, far], dim=-1)
    #print("Rays shape:", rays.shape)
    
    # Debug visualization: Visualize rays in 3D
    if convex.config.get('debug_viz_render', False):
        
        print("Creating debug visualization of rays...")
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Sample rays to visualize (avoid plotting too many)
        n_rays_to_plot = min(100000, rays_o.shape[0])
        indices = np.linspace(0, rays_o.shape[0] - 1, n_rays_to_plot, dtype=int)
        
        rays_o_np = rays_o[indices].cpu().numpy()
        rays_d_np = rays_d[indices].cpu().numpy()
        near_np = near[indices].cpu().numpy()
        far_np = far[indices].cpu().numpy()
        
        # Plot ray origins
        ax.scatter(rays_o_np[:, 0], rays_o_np[:, 1], rays_o_np[:, 2],
                  c='red', marker='o', s=50, label='Ray Origins', alpha=0.6)
        
        # Plot rays from origin to near plane
        for i in range(n_rays_to_plot):
            origin = rays_o_np[i]
            direction = rays_d_np[i]
            near_val = near_np[i, 0]
            far_val = far_np[i, 0]
            
            # Near point
            near_point = origin + direction * near_val
            # Far point
            far_point = origin + direction * far_val
            
            # Plot ray from origin to near (cyan)
            ax.plot([origin[0], near_point[0]],
                   [origin[1], near_point[1]],
                   [origin[2], near_point[2]],
                   color='cyan', alpha=0.3, linewidth=0.5)
            
            # Plot ray from near to far (yellow)
            ax.plot([near_point[0], far_point[0]],
                   [near_point[1], far_point[1]],
                   [near_point[2], far_point[2]],
                   color='yellow', alpha=0.5, linewidth=1.0)
        
        # Plot near and far points
        near_points = rays_o_np + rays_d_np * near_np
        far_points = rays_o_np + rays_d_np * far_np
        
        ax.scatter(near_points[:, 0], near_points[:, 1], near_points[:, 2],
                  c='green', marker='^', s=30, label=f'Near plane (d={near_np[0,0]:.4f})', alpha=0.6)
        ax.scatter(far_points[:, 0], far_points[:, 1], far_points[:, 2],
                  c='blue', marker='v', s=30, label=f'Far plane (d={far_np[0,0]:.4f})', alpha=0.6)
        
        ax.set_xlabel('X (world)')
        ax.set_ylabel('Y (world)')
        ax.set_zlabel('Z (world)')
        ax.set_title(f'Ray Visualization ({n_rays_to_plot}/{rays_o.shape[0]} rays)\nCyan: origin→near, Yellow: near→far')
        ax.legend()
        
        # Set equal aspect ratio
        all_points = np.concatenate([rays_o_np, near_points, far_points], axis=0)
        max_range = np.array([all_points[:, 0].max() - all_points[:, 0].min(),
                             all_points[:, 1].max() - all_points[:, 1].min(),
                             all_points[:, 2].max() - all_points[:, 2].min()]).max() / 2.0
        
        mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        plt.savefig('docs/debug_rays_3d.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Now create 2D projections for all axis permutations
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Prepare data for 2D plots
        projections = [
            ('X-Y (Top view)', 0, 1, 'X', 'Y'),
            ('X-Z (Front view)', 0, 2, 'X', 'Z'),
            ('Y-Z (Side view)', 1, 2, 'Y', 'Z'),
        ]
        
        for idx, (title, axis1, axis2, label1, label2) in enumerate(projections):
            # Plot with rays (top row)
            ax_rays = axes[0, idx]
            
            # Plot ray segments
            for i in range(min(500, n_rays_to_plot)):  # Limit for clarity in 2D
                origin = rays_o_np[i]
                direction = rays_d_np[i]
                near_val = near_np[i, 0]
                far_val = far_np[i, 0]
                
                near_point = origin + direction * near_val
                far_point = origin + direction * far_val
                
                # Origin to near (cyan)
                ax_rays.plot([origin[axis1], near_point[axis1]],
                            [origin[axis2], near_point[axis2]],
                            color='cyan', alpha=0.2, linewidth=0.5)
                
                # Near to far (yellow)
                ax_rays.plot([near_point[axis1], far_point[axis1]],
                            [near_point[axis2], far_point[axis2]],
                            color='yellow', alpha=0.3, linewidth=0.8)
            
            # Plot points
            ax_rays.scatter(rays_o_np[:, axis1], rays_o_np[:, axis2],
                           c='red', marker='o', s=20, label='Origins', alpha=0.5)
            ax_rays.scatter(near_points[:, axis1], near_points[:, axis2],
                           c='green', marker='^', s=15, label='Near', alpha=0.5)
            ax_rays.scatter(far_points[:, axis1], far_points[:, axis2],
                           c='blue', marker='v', s=15, label='Far', alpha=0.5)
            
            ax_rays.set_xlabel(f'{label1} (world)')
            ax_rays.set_ylabel(f'{label2} (world)')
            ax_rays.set_title(f'{title} - With Rays')
            ax_rays.legend()
            ax_rays.grid(True, alpha=0.3)
            ax_rays.set_aspect('equal', adjustable='box')
            
            # Plot without rays (bottom row) - just points for clarity
            ax_points = axes[1, idx]
            
            ax_points.scatter(rays_o_np[:, axis1], rays_o_np[:, axis2],
                             c='red', marker='o', s=30, label='Origins', alpha=0.6)
            ax_points.scatter(near_points[:, axis1], near_points[:, axis2],
                             c='green', marker='^', s=20, label='Near', alpha=0.6)
            ax_points.scatter(far_points[:, axis1], far_points[:, axis2],
                             c='blue', marker='v', s=20, label='Far', alpha=0.6)
            
            ax_points.set_xlabel(f'{label1} (world)')
            ax_points.set_ylabel(f'{label2} (world)')
            ax_points.set_title(f'{title} - Points Only')
            ax_points.legend()
            ax_points.grid(True, alpha=0.3)
            ax_points.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.savefig('docs/debug_rays_2d_projections.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Debug ray visualizations saved:")
        print(f"  - docs/debug_rays_3d.png (3D view)")
        print(f"  - docs/debug_rays_2d_projections.png (XY, XZ, YZ projections)")
        print(f"  Total rays: {rays_o.shape[0]}")
        print(f"  Visualized: {n_rays_to_plot}")
        print(f"  Near distance: {near_np[0,0]:.6f}")
        print(f"  Far distance: {far_np[0,0]:.6f}")
        print(f"  Ray origins range: X[{rays_o_np[:, 0].min():.4f}, {rays_o_np[:, 0].max():.4f}], "
              f"Y[{rays_o_np[:, 1].min():.4f}, {rays_o_np[:, 1].max():.4f}], "
              f"Z[{rays_o_np[:, 2].min():.4f}, {rays_o_np[:, 2].max():.4f}]")
        
        # Disable after first call
        convex.config['debug_viz_render'] = False

    # Render and reshape
    all_ret = batchify_rays(rays, chunk=chunk, **kwargs)
    # for k in all_ret:
    #     k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
    #     all_ret[k] = all_ret[k].reshape(k_sh)
    
    # Include ray origins and directions for debugging/visualization
    all_ret['ray_origins'] = rays_o
    all_ret['ray_directions'] = rays_d
    
    return all_ret


def compute_loss(output, target, args, losses, i):
    loss = {}
    # 2D: (1, 1, H, W) or (B, 1, H, W)
    # 3D: (1, D, H, W) or (B, D, H, W) where D > 1
    is_3d_volume = output.dim() == 4 and output.shape[1] > 1
    
    # Check if dimensions are large enough for SSIM (needs at least win_size in each spatial dim)
    # SSIM window size is typically 7x7, so we need at least 7 in each dimension
    min_size_for_ssim = args.ssim_filter_size  # Usually 7
    spatial_dims = output.shape[-2:]  # Last two dimensions (height, width)
    needs_padding = any(dim < min_size_for_ssim for dim in spatial_dims)
    
    output_for_ssim = output
    target_for_ssim = target
    
    if needs_padding:
        # Pad the spatial dimensions by repeating values to make them large enough for SSIM
        # This happens when N_samples=1, creating shape [1, 1, N_rays, 1]
        #print(f"Padding output from {output.shape} for SSIM computation (needs >={min_size_for_ssim} in each spatial dim)")
        
        # Calculate how much padding needed for each dimension
        pad_h = max(0, min_size_for_ssim - spatial_dims[0])
        pad_w = max(0, min_size_for_ssim - spatial_dims[1])
        
        # Use reflection padding to repeat edge values
        # pad format: (left, right, top, bottom)
        padding = (pad_w // 2, (pad_w + 1) // 2, pad_h // 2, (pad_h + 1) // 2)
        
        output_for_ssim = F.pad(output, padding, mode='replicate')
        target_for_ssim = F.pad(target, padding, mode='replicate')
        
        print(f"Padded to {output_for_ssim.shape}")

    if args.loss == "l2" or i < args.r_warm_up_it:
        l2_intensity_loss = losses['l2'](output, target)
        loss["l2"] = (1.0, l2_intensity_loss)
        
    elif args.loss == "ssim":
        if is_3d_volume:
            #print(f"Computing SSIM for 3D volume: {output_for_ssim.shape}")
            ssim_intensity_loss = losses['ssim'](output_for_ssim.unsqueeze(1), target_for_ssim.unsqueeze(1))
        else:
            # 2D case
            ssim_intensity_loss = losses['ssim'](output_for_ssim, target_for_ssim)
        
        loss["ssim"] = (args.ssim_lambda, ssim_intensity_loss)
        l2_intensity_loss = img2mse(output, target)
        loss["l2"] = (1.-args.ssim_lambda, l2_intensity_loss)

    # Add gradient magnitude loss for texture preservation
    if args.gradient_loss_weight > 0 and i > args.r_warm_up_it:
        grad_loss = compute_gradient_magnitude_loss(output, target, is_3d=is_3d_volume)
        loss["gradient_mag"] = (args.gradient_loss_weight, grad_loss)
        
    return loss

def reflection_smoothness_loss(reflection_coeff, dists=None):
    """
    Encourage smooth reflection coefficients along rays (depth direction).
    
    Args:
        reflection_coeff: [B, D, H, W] - Reflection coefficients
        dists: Distance between samples along rays (optional)
            If None, assumes uniform spacing
    """
    # Compute differences along the depth direction (dimension 1)
    # reflection_coeff shape: [B, D, H, W]
    diff = torch.abs(reflection_coeff[:, 1:, :, :] - reflection_coeff[:, :-1, :, :])  # [B, D-1, H, W]
    
    # Weight by distance if provided (larger gaps need more smoothness)
    if dists is not None:
        # dists should be broadcastable to [B, D-1, H, W]
        if dists.dim() == 1:  # [D] or [D-1]
            # Expand to match diff shape
            dists_expanded = dists[:-1].view(1, -1, 1, 1)  # [1, D-1, 1, 1]
        elif dists.dim() == 2:  # [H, W]
            # Same distance for all depths
            dists_expanded = dists.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif dists.dim() == 3:  # [D-1, H, W]
            dists_expanded = dists.unsqueeze(0)  # [1, D-1, H, W]
        else:
            # Assume already correct shape
            dists_expanded = dists
        
        # Penalize abrupt changes weighted by distance
        smoothness = torch.mean(diff / (dists_expanded + 1e-8))
    else:
        # Unweighted smoothness (assumes uniform spacing)
        smoothness = torch.mean(diff)
    
    return smoothness

def compute_gradient_magnitude_loss(pred, target, is_3d=False):
    """
    Matches gradient magnitudes to preserve edge sharpness and texture.
    Critical for ultrasound speckle patterns.
    """
    if is_3d:
        # Compute 3D Sobel-like gradients
        # pred shape: [B, D, H, W]
        grad_pred_x = pred[:, :, 1:, :] - pred[:, :, :-1, :]  # [B, D, H-1, W]
        grad_target_x = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        grad_pred_y = pred[:, :, :, 1:] - pred[:, :, :, :-1]  # [B, D, H, W-1]
        grad_target_y = target[:, :, :, 1:] - target[:, :, :, :-1]
        
        grad_pred_z = pred[:, 1:, :, :] - pred[:, :-1, :, :]  # [B, D-1, H, W]
        grad_target_z = target[:, 1:, :, :] - target[:, :-1, :, :]
        
        # Align all gradients to common shape [B, D-1, H-1, W-1]
        # Crop each gradient to the smallest common dimensions
        grad_pred_x = grad_pred_x[:, :-1, :, :-1]    # [B, D-1, H-1, W-1]
        grad_target_x = grad_target_x[:, :-1, :, :-1]
        
        grad_pred_y = grad_pred_y[:, :-1, :-1, :]    # [B, D-1, H-1, W-1]
        grad_target_y = grad_target_y[:, :-1, :-1, :]
        
        grad_pred_z = grad_pred_z[:, :, :-1, :-1]    # [B, D-1, H-1, W-1]
        grad_target_z = grad_target_z[:, :, :-1, :-1]
        
        # Magnitude of gradients
        mag_pred = torch.sqrt(grad_pred_x**2 + grad_pred_y**2 + grad_pred_z**2 + 1e-8)
        mag_target = torch.sqrt(grad_target_x**2 + grad_target_y**2 + grad_target_z**2 + 1e-8)
    else:
        # 2D gradients
        # pred shape: [B, 1, H, W]
        grad_pred_x = pred[:, :, 1:, :] - pred[:, :, :-1, :]  # [B, 1, H-1, W]
        grad_target_x = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        grad_pred_y = pred[:, :, :, 1:] - pred[:, :, :, :-1]  # [B, 1, H, W-1]
        grad_target_y = target[:, :, :, 1:] - target[:, :, :, :-1]
        
        # Align to common shape [B, 1, H-1, W-1]
        grad_pred_x = grad_pred_x[:, :, :, :-1]    # [B, 1, H-1, W-1]
        grad_target_x = grad_target_x[:, :, :, :-1]
        
        grad_pred_y = grad_pred_y[:, :, :-1, :]    # [B, 1, H-1, W-1]
        grad_target_y = grad_target_y[:, :, :-1, :]
        
        # Magnitude of gradients
        mag_pred = torch.sqrt(grad_pred_x**2 + grad_pred_y**2 + 1e-8)
        mag_target = torch.sqrt(grad_target_x**2 + grad_target_y**2 + 1e-8)
    
    return F.l1_loss(mag_pred, mag_target)

def compute_regularization(rendering_output, reg_funcs, weights=(0.01, 0.00001, 0.34), reflection_smoothness_weight=0.2):
    lncc = reg_funcs['lncc']
    
    scatter_amplitude = rendering_output['scatter_amplitude']
    attenuation_coeff = rendering_output['attenuation_coeff']
    reflection_coeff = rendering_output['reflection_coeff']

    # For spatial_dims=3, LNCC expects 5D tensors [batch, channel, depth, height, width]
    if scatter_amplitude.dim() == 3:
        scatter_amplitude = scatter_amplitude.unsqueeze(0).unsqueeze(0)  # e.g [1, 1, 68, 450, 150]
        attenuation_coeff = attenuation_coeff.unsqueeze(0).unsqueeze(0)  # e.g [1, 1, 68, 450, 150]
        reflection_coeff = reflection_coeff.unsqueeze(0).unsqueeze(0)  # e.g [1, 1, 68, 450, 150]
    elif scatter_amplitude.dim() == 4:
        scatter_amplitude = scatter_amplitude.unsqueeze(0)  # Add batch dimension
        attenuation_coeff = attenuation_coeff.unsqueeze(0)  # Add batch dimension
        reflection_coeff = reflection_coeff.unsqueeze(0)  # Add batch dimension
    
    lncc_w, tv_w, refl_max = weights
    lcc_penalty_scatter_attenuation = lncc(scatter_amplitude, attenuation_coeff)
    lcc_penalty_scatter_reflection = lncc(scatter_amplitude, reflection_coeff)
    reg = {}

    reg["lcc_penalty"] = (lncc_w, lcc_penalty_scatter_attenuation)
    reg["lcc_penalty_reflection"] = (lncc_w, lcc_penalty_scatter_reflection)
    
    # TV penalty computation - handle both 3D and 4D tensors
    scatter_amp_orig = rendering_output['scatter_amplitude']
    reflection_coeff = rendering_output['reflection_coeff']
    
    # Determine if 3D volume (has depth dimension) or 2D
    # Check actual dimensions: 3D is [B, D, H, W], 2D is [B, 1, H, W]
    is_3d_volume = scatter_amp_orig.dim() == 4 and scatter_amp_orig.shape[1] > 1
    
    if is_3d_volume and reflection_smoothness_weight > 0:
        reg["reflection_smoothness"] = (reflection_smoothness_weight, reflection_smoothness_loss(rendering_output['reflection_coeff'], None))
    
    if is_3d_volume:
        # 3D case: [B, D, H, W]
        # Compute gradients along all three dimensions
        # Note: scatter_amp_orig shape is [B, D, H, W]
        dz_ampl = scatter_amp_orig[:, 1:, :, :] - scatter_amp_orig[:, :-1, :, :]  # [B, D-1, H, W]
        dy_ampl = scatter_amp_orig[:, :, :, 1:] - scatter_amp_orig[:, :, :, :-1]  # [B, D, H, W-1]
        dx_ampl = scatter_amp_orig[:, :, 1:, :] - scatter_amp_orig[:, :, :-1, :]  # [B, D, H-1, W]
        
        # Reflection coefficient weighting (same idea as 2D)
        # Weight gradients less in regions with high reflection (edges/boundaries)
        if reflection_coeff.shape == scatter_amp_orig.shape:
            # Apply weighting consistently with 2D case
            # Crop reflection_coeff to match gradient shapes
            refl_z = reflection_coeff[:, :-1, :, :]  # [B, D-1, H, W]
            refl_y = reflection_coeff[:, :, :, :-1]  # [B, D, H, W-1]
            refl_x = reflection_coeff[:, :, :-1, :]  # [B, D, H-1, W]
            
            tv_z = torch.sum((refl_max - refl_z) * torch.abs(dz_ampl))
            tv_y = torch.sum((refl_max - refl_y) * torch.abs(dy_ampl))
            tv_x = torch.sum((refl_max - refl_x) * torch.abs(dx_ampl))
        else:
            # Fallback: unweighted TV (shouldn't happen if rendering is consistent)
            print(f"Warning: reflection_coeff shape {reflection_coeff.shape} doesn't match scatter_amplitude {scatter_amp_orig.shape}")
            tv_z = torch.sum(torch.abs(dz_ampl))
            tv_y = torch.sum(torch.abs(dy_ampl))
            tv_x = torch.sum(torch.abs(dx_ampl))
        
        amplitude_tv_penalty = tv_x + tv_y + tv_z
    else:
        # 2D case: [B, 1, H, W] or [B, C, H, W]
        dy_ampl = scatter_amp_orig[:, :, :, 1:] - scatter_amp_orig[:, :, :, :-1]
        dy_ampl = torch.cat([dy_ampl, dy_ampl[:, :, :, -1:]], dim=-1)  # Pad y direction

        dx_ampl = scatter_amp_orig[:, :, 1:, :] - scatter_amp_orig[:, :, :-1, :]
        dx_ampl = torch.cat([dx_ampl, dx_ampl[:, :, -1:, :]], dim=-2)
        
        # Calculate TV penalties weighted by reflection coefficient
        # Higher reflection = boundary/edge → allow more variation (less penalty)
        total_variation_penalty_y_ampl = torch.sum(
            (refl_max - reflection_coeff) * torch.abs(dy_ampl.squeeze()))
        total_variation_penalty_x_ampl = torch.sum(
            (refl_max - reflection_coeff) * torch.abs(dx_ampl.squeeze()))

        amplitude_tv_penalty = total_variation_penalty_x_ampl + total_variation_penalty_y_ampl
    
    # Uncomment to enable TV regularization
    #reg["tv_penalty"] = (tv_w, amplitude_tv_penalty)
    
    return reg

def compute_pts_from_pose(H, W, sw, sh, pose, near, far):
    o, d = get_rays_us_linear(H, W, sw, sh, pose)
    o = o.reshape(-1, 3).float()
    d = d.reshape(-1, 3).float()
    # Decide where to sample along each ray
    N_samples = H
    t_vals = torch.linspace(0.0, 1.0, N_samples).to(pose.device)
    z_vals = near * (1.0 - t_vals) + far * t_vals

    z_vals = z_vals.expand(W, N_samples)

    # Points in space to evaluate model at
    origin = o.unsqueeze(-2)
    step = d.unsqueeze(-2) * z_vals.unsqueeze(-1)

    pts = step + origin

    return pts
