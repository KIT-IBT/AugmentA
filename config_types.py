from dataclasses import dataclass, field

#RA Configs

@dataclass
class RAThresholds:
    """Laplace field thresholds for RA region segmentation."""
    tao_icv: float = 0.95              # inferior vena cava threshold
    tao_scv: float = 0.10              # superior vena cava threshold
    alpha_RAA: float = 0.30            # interpolation factor for RAA segmentation
    ivc_phie_r_offset: float = 0.2     # offset added to max phie_r of IVC

@dataclass
class RABundleSizes:
    """Bundle radii (before scaling with args.scale)."""
    w_ct: float = 4.62                 # crista terminalis radius
    w_pm: float = 0.66                 # pectinate muscle radius
    w_bb: float = 2.0                  # Bachmann bundle radius
    r_sn: float = 2.5                  # sinus node radius
    tv_ring_radius: float = 4.0        # radius around TV ring points
    ct_band_query_radius: float = 7.0  # kd-tree query radius for CT band
    cs_ring_radius: float = 4.0        # radius around CS ring points
    bb_sphere_factor: float = 1.02     # sphere radius = bb_radius * this

@dataclass
class PectinateMuscleConfig:
    """Parameters controlling pectinate muscle generation."""
    pm_num: int = 15                       # total number of pectinate muscles
    pm_skip_first: int = 2                 # skip first N PMs (IVC side)
    pm_skip_last: int = 2                  # skip last N PMs (edge artifacts)
    pm_trim_fraction: float = 0.03         # trim first X% of path (roof of RA)
    pm_downsample_fraction: float = 0.065  # downsample step as fraction of path length
    pm1_downsample_fraction: float = 0.1   # first PM (to RAA apex)
    pm_min_path_length: int = 3            # minimum points for valid PM
    band_width_fraction: float = 0.15      # adaptive band width in dijkstra_path_on_a_plane

@dataclass
class RABachmannBundleConfig:
    """Parameters for Bachmann bundle generation."""
    bb_downsample_fraction: float = 0.1    # downsample step as fraction of BB path length
    bb_center_fraction: float = 0.45       # where on BB path to place the bridge center
    bb_step: int = 20                      # downsampling step for bridge path

@dataclass
class RAFiberConfig:
    """Main configuration for RA fiber generation."""
    thresholds: RAThresholds = field(default_factory=RAThresholds)
    bundles: RABundleSizes = field(default_factory=RABundleSizes)
    pm: PectinateMuscleConfig = field(default_factory=PectinateMuscleConfig)
    bb: RABachmannBundleConfig = field(default_factory=RABachmannBundleConfig)

#LA Configs

@dataclass
class LAThresholds:
    """Laplace field thresholds for LA region segmentation."""
    lpv_lb: float = 0.0                # LPV binary search lower bound
    lpv_ub: float = 0.4                # LPV binary search upper bound
    rpv_lb: float = 0.6                # RPV binary search lower bound
    rpv_ub: float = 1.0                # RPV binary search upper bound
    laa_bb_offset_low: float = 0.03    # LAA_bb lower offset from max_phie_ab
    laa_bb_offset_high: float = 0.01   # LAA_bb upper offset from max_phie_ab
    laa_shape_thr: float = 0.95        # LAA shape threshold (phie_ab3)
    epi_endo_thr: float = 0.5          # epi/endo separation (phie_phi)

@dataclass
class LABundleSizes:
    """Bundle radii (before scaling with args.scale)."""
    w_bb: float = 2.0                      # Bachmann bundle radius
    mv_ring_radius: float = 4.0            # MV ring element selection radius
    bilayer_normal_offset: float = 0.1      # offset for bilayer surface normals

@dataclass
class LAFiberConfig:
    """Main configuration for LA fiber generation."""
    thresholds: LAThresholds = field(default_factory=LAThresholds)
    bundles: LABundleSizes = field(default_factory=LABundleSizes)

#Bridge Configs

@dataclass
class BridgeConfig:
    """Parameters for interatrial bridge generation."""
    bridge_radius: float = 1.65            # bridge tube radius (before scaling)
    mpb_position: float = 0.6              # middle posterior bridge position on septum path
    upb_position: float = 0.4              # upper posterior bridge position on septum path
    intersection_radius: float = 0.7       # intersection point search radius (before scaling)
    cs_tv_offset: float = 0.1              # CS-TV offset factor for coronary sinus bridge
    bilayer_normal_offset: float = 0.1     # bilayer surface normal offset (before scaling)
    remesh_target_length: float = 0.4      # remeshing target edge length (before scaling)
    bilayer_max_dist: float = 0.12         # max distance for bilayer generation (before scaling)

#Top-level

@dataclass
class AugmentAConfig:
    """Top-level configuration aggregating all sub-configs."""
    ra: RAFiberConfig = field(default_factory=RAFiberConfig)
    la: LAFiberConfig = field(default_factory=LAFiberConfig)
    bridges: BridgeConfig = field(default_factory=BridgeConfig)