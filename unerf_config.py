import configargparse

def config_parser():

    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--expname", type=str, help="experiment name")
    parser.add_argument("--expname_nerf", type=str, help="reconstruction mode only: experiment folder in basedir of the trained NeRF to load")
    parser.add_argument(
        "--basedir", type=str, default="./logs/", help="where to store ckpts and logs"
    )
    parser.add_argument(
        "--datadir",
        type=str,
        default="./data/ice",
        help="input data directory",
    )

    # training options
    parser.add_argument("--n_iters", type=int, default=100000)
    parser.add_argument("--ssim_filter_size", type=int, default=7)
    parser.add_argument("--ssim_lambda", type=float, default=0.75)
    parser.add_argument("--loss", type=str, default="l2")
    parser.add_argument("--probe_depth", type=int, default=140)
    parser.add_argument("--probe_width", type=int, default=80)
    parser.add_argument("--output_ch", type=int, default=5)

    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--confmap", type=bool, default=False)
    parser.add_argument("--pose_path", type=str, default=None)
    
    # Multi-volume data loading options
    parser.add_argument("--use_multi_volume_data_loading", action="store_true", 
                        help="Enable multi-volume loading from volumeX/ subdirectories")
    parser.add_argument("--train_volumes", type=str, default=None,
                        help="Train volume split (e.g., '0-5', 'even', '0,1,2')")
    parser.add_argument("--val_volumes", type=str, default=None,
                        help="Validation volume split")
    parser.add_argument("--test_volumes", type=str, default=None,
                        help="Test volume split")
    parser.add_argument("--volume_prefix", type=str, default="volume",
                        help="Prefix for volume directories (default: 'volume')")

    parser.add_argument(
        "--random_seed", type=int, default=42,
        help="seed for numpy, torch and random; a negative value disables seeding",
    )

    parser.add_argument("--netdepth", type=int, default=8, help="layers in network")
    parser.add_argument("--netwidth", type=int, default=128, help="channels per layer")
    parser.add_argument(
        "--netdepth_fine", type=int, default=8, help="not used"
    )
    parser.add_argument(
        "--netwidth_fine",
        type=int,
        default=128,
        help="not used",
    )
    parser.add_argument(
        "--N_rand",
        type=int,
        default=32 * 32 * 4,
        help="not used",
    )
    parser.add_argument("--lrate", type=float, default=1e-4, help="learning rate")
    parser.add_argument(
        "--lrate_decay",
        type=int,
        default=250,
        help="learning rate decays by 10x every lrate_decay * 1000 iterations",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=4096 * 16,
        help="number of rays processed in parallel, decrease if running out of memory; "
             "with --full_volume_mode it must be a multiple of convex_n_rays",
    )
    parser.add_argument(
        "--netchunk",
        type=int,
        default=4096 * 16,
        help="number of pts sent through network in parallel, decrease if running out of memory",
    )

    parser.add_argument(
        "--ft_path",
        type=str,
        default=None,
        help="checkpoint (.tar) to load; training resumes from it (default: latest checkpoint in basedir/expname), evaluation requires it",
    )

    # rendering options
    parser.add_argument(
        "--N_samples", type=int, default=64, help="samples per ray in linear mode; convex mode uses --convex_n_samples instead"
    )
    parser.add_argument(
        "--i_embed",
        type=int,
        default=0,
        help="positional encoding: -1 disables it; with --use_mip and --full_volume_mode it is the number "
             "of encoding values per frequency (6 = sin and cos of x, y, z)",
    )
    parser.add_argument(
        "--i_embed_gauss",
        type=int,
        default=0,
        help="not used",
    )

    parser.add_argument(
        "--multires",
        type=int,
        default=10,
        help="number of frequency bands L of the positional encoding (max frequency 2^(L-1)); "
             "also used by the integrated positional encoding with --use_mip",
    )
    parser.add_argument(
        "--multires_views",
        type=int,
        default=4,
        help="not used",
    )
    
    # MIP-NeRF elongation parameters
    parser.add_argument(
        "--use_elongation",
        action="store_true",
        help="Enable sideways elongation for MIP-NeRF Gaussians (experimental)",
    )
    parser.add_argument(
        "--max_elongation",
        type=float,
        default=2.0,
        help="Maximum elongation factor for sideways stretching of Gaussians",
    )
    parser.add_argument(
        "--mip_voxel_radius_lateral",
        type=float,
        default=0.173,
        help="Base voxel radius for MIP-NeRF conical frustum calculations (lateral/x-y spread)",
    )
    parser.add_argument(
        "--mip_voxel_radius_depth",
        type=float,
        default=0.173,
        help="Base voxel radius for MIP-NeRF depth/z-axis direction spread (elevational resolution)",
    )
    
    parser.add_argument(
        "--raw_noise_std",
        type=float,
        default=0.0,
        help="not used",
    )

    parser.add_argument(
        "--render_only",
        action="store_true",
        help="not used (use evaluate_ultranerf.py to render from a checkpoint)",
    )
    parser.add_argument(
        "--render_test",
        action="store_true",
        help="not used",
    )
    parser.add_argument(
        "--render_factor",
        type=int,
        default=0,
        help="not used",
    )

    # training options

    # dataset options
    parser.add_argument("--dataset_type", type=str, default="us", help="options: us")
    parser.add_argument(
        "--testskip",
        type=int,
        default=8,
        help="not used",
    )

    # logging/saving options
    parser.add_argument(
        "--i_print",
        type=int,
        default=1000,
        help="every N iterations: print the losses and save training and validation renderings "
             "(with --save_volume also the volumes and their metrics)",
    )
    parser.add_argument(
        "--i_img", type=int, default=1000, help="not used"
    )
    parser.add_argument(
        "--i_weights", type=int, default=10000, help="frequency of weight ckpt saving"
    )
    parser.add_argument(
        "--save_volume",
        action='store_true',
        help="save rendered 3D volumes as .npy and compute metrics against the target (full_volume_mode only)"
    )
    
    # Debug visualization
    parser.add_argument(
        "--debug_viz_poses", 
        action='store_true',
        help="not used (see --debug_viz_render)"
    )
    
    # Pose transformation
    parser.add_argument(
        "--transform_poses",
        action='store_true',
        help="Apply coordinate system transformation to poses after loading"
    )
    parser.add_argument(
        "--pose_axis_mapping",
        type=str,
        default='z_to_x',
        choices=['z_to_x', 'y_to_x', 'x_to_z', 'identity'],
        help="Coordinate system transformation: 'z_to_x' (depth along X from Z), "
             "'y_to_x' (depth along X from Y), 'x_to_z' (depth along Z from X), "
             "'identity' (no transformation)"
    )

    parser.add_argument('--reg', action='store_true',
                        help='enable the LNCC and reflection smoothness regularization after r_warm_up_it')
    parser.add_argument("--r_tv_penalty", type=float, default=0.00001,
                        help='weight of the TV penalty on the scatter amplitude (currently computed but not added to the loss)')
    parser.add_argument("--r_lcc_penalty", type=float, default=0.001,
                        help='weight of the LNCC penalty between scatter amplitude and attenuation / reflection coefficients')
    parser.add_argument("--r_clustering", type=float, default=0.,
                        help='not used')
    parser.add_argument("--r_clustering_distance", type=float, default=0.,
                        help='not used')
    parser.add_argument("--r_max_reflection", type=float, default=0.34,
                        help='reflection level for the TV penalty, weighted by (r_max_reflection - reflection) '
                             'so strong reflectors are smoothed less; only used by the TV penalty')
    parser.add_argument("--r_warm_up_it", type=int, default=10000,
                        help='iterations trained with plain L2 loss; afterwards SSIM (with --loss ssim), '
                             'the gradient loss and the --reg terms are switched on')
    parser.add_argument("--gradient_loss_weight", type=float, default=0.1,
                        help='Weight of the gradient-magnitude loss, added after r_warm_up_it (0 disables it)')
    parser.add_argument("--reflection_smoothness_weight", type=float, default=0.2,
                        help='Weight of the reflection smoothness term along depth, 3D volumes with --reg only (0 disables it)')
    # segmentation options (not used)
    parser.add_argument('--segm_head', action='store_true',
                        help='not used')
    parser.add_argument('--segmentation', action='store_true',
                        help='not used')
    parser.add_argument("--segm_frac", type=int, default=5,
                        help='not used')
    # reconstruction network options (not functional: its renderer raises NotImplementedError)
    parser.add_argument('--reconstruction', action='store_true',
                        help='train an additional reconstruction network (not functional)')
    parser.add_argument('--confidence', action='store_true',
                        help='not used')
    parser.add_argument('--rec_only_theta', action='store_true',
                        help='reconstruction mode only: reconstruction network input has 3 instead of 6 channels')
    parser.add_argument("--rec_step", type=int, default=20,
                        help='not used')
    parser.add_argument("--rec_iter", type=int, default=20000,
                        help='not used')
    parser.add_argument('--rec_only_occ', action='store_true',
                        help='reconstruction mode only: reconstruction network input has 3 instead of 6 channels')


    # convex probe options
    parser.add_argument('--use_convex_mode', action='store_true',
                        help='use the convex (fan) probe geometry; otherwise a linear probe of probe_width x probe_depth mm')
    
    # Sampling strategy (experimental: concentric_circles is not supported end to end)
    parser.add_argument('--sampling_strategy', type=str, default='uniform_fan',
                        choices=['uniform_fan', 'concentric_circles'],
                        help='ray sampling for convex mode: uniform_fan (default) or concentric_circles (experimental)')
    parser.add_argument('--n_circles', type=int, default=5,
                        help='Number of concentric circles (for concentric_circles strategy)')
    parser.add_argument('--base_rays_per_circle', type=int, default=8,
                        help='Base number of rays for innermost circle (for concentric_circles strategy)')
    parser.add_argument('--ray_density_multiplier', type=float, default=1.5,
                        help='Multiplier for rays in each successive circle (for concentric_circles strategy)')
    parser.add_argument('--density_growth_type', type=str, default='exponential',
                        choices=['exponential', 'linear', 'quadratic'],
                        help='Type of ray density growth: exponential (base*mult^k), linear (base+k*mult), or quadratic (base+k^2*mult)')
    
    # Radial path integration (tree-ring sampling): not implemented
    parser.add_argument('--use_radial_path_mode', action='store_true',
                        help='radial path integration (tree-ring sampling); not implemented, rendering raises NotImplementedError')
    
    parser.add_argument('--use_mip', action='store_true',
                        help='cast cones and approximate each sample by a multivariate Gaussian with integrated '
                             'positional encoding (mip-NeRF style) instead of point sampling')
    parser.add_argument("--convex_n_rays", type=int,
                        help='number of rays (scanlines) per fan image')
    parser.add_argument("--convex_n_samples", type=int,
                        help='number of samples per ray; overrides --N_samples in convex mode')
    parser.add_argument("--convex_center_x", type=int,
                        help='x coordinate of center of convex fan')
    parser.add_argument("--convex_center_y", type=int,
                        help='y coordinate of center of convex fan')
    parser.add_argument("--convex_fan_angle", type=float,
                        help='full opening angle of the convex fan in degrees')
    parser.add_argument("--convex_big_radius", type=float,
                        help='outer radius of the convex fan in pixels')
    parser.add_argument("--convex_small_radius", type=float,
                        help='inner radius of the convex fan (probe surface) in pixels')
    parser.add_argument("--convex_scaling", type=float,
                        help='pixel spacing in mm for x and y; when set it overrides --convex_scaling_x/_y. '
                             'In convex mode it only sets the far bound used in visualizations')
    parser.add_argument("--convex_scaling_x", type=float,
                        help='pixel spacing in mm along x; ignored when --convex_scaling is set')
    parser.add_argument("--convex_scaling_y", type=float,
                        help='pixel spacing in mm along y; ignored when --convex_scaling is set')
    parser.add_argument('--convex_use_compiled_cache', default=False, action='store_true',
                        help='cache the target images remapped onto the fan sampling points (in memory and in a file in basedir)')
    parser.add_argument('--convex_use_compiled_cache_on_the_fly', default=True, action='store_true',
                        help="with --convex_use_compiled_cache: fill the cache as images are first used instead of precomputing it")
    
    parser.add_argument('--debug_viz_render', action='store_true',
                        help='save 3D plots of poses and rays on the first render call (figures/debug_rays_*.png and basedir)')
    parser.add_argument('--convex_save_meta_files', default=False, action='store_true',
                        help="create a meta/ folder in basedir (nothing is written to it currently)")
    parser.add_argument('--full_volume_mode', default=False, action='store_true',
                        help="train and evaluate on 3D volumes (a stack of D fan slices per sample) instead of single 2D images")
    
    # Spherical (elevation-angle) volume mode: not implemented
    parser.add_argument('--use_spherical_volume', default=False, action='store_true',
                        help="spherical volume mode: rays diverge in elevation instead of stacking D "
                             "parallel fan slices; not implemented (its ray generator is never called)")
    parser.add_argument('--n_rays_elevation', type=int, default=68,
                        help="spherical volume mode only (not implemented): number of elevation rays")
    parser.add_argument('--opening_angle_elevation', type=float, default=20.0,
                        help="spherical volume mode only (not implemented): elevation opening angle in degrees")

    return parser
