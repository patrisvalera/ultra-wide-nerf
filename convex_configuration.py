
# Configuration object shared as global parameters.
# All parameters can be specified in configuration file e.g. ./conf_us_set_l2.txt
# Params in configuration file will override params in this file except:
# convex_use_compiled_cache
# convex_use_compiled_cache_on_the_fly
# convex_save_meta_files
# Those parameters override configuration file.
config = {}

def get_convex_settings():
    """Helper method to get most used convex settings"""
    convex_cx, convex_cy = config['convex_center']
    return (config['convex_center'], convex_cx, convex_cy,
            config['convex_n_rays'], config['convex_angle'],
            config['convex_radius'], config['convex_radius2'])