from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RAThresholds:
    """Laplace field thresholds for RA region segmentation."""
    tao_tv: float = 0.9  # tricuspid valve threshold
    tao_icv: float = 0.95  # inferior vena cava threshold
    tao_scv: float = 0.10  # superior vena cava threshold
    tao_ct_plus: float = -0.10  # crista terminalis upper bound
    tao_ct_minus: float = -0.13  # crista terminalis lower bound
    tao_ib: float = -0.06  # inter-caval bundle threshold
    tao_ras: float = 0.13  # right atrial septum threshold
    tao_raw: float = 0.55  # right atrial wall threshold
    alpha_RAA: float = 0.30  # interpolation factor for RAA segmentation
    ivc_phie_r_offset: float = 0.2  # offset added to max phie_r of IVC


@dataclass
class BundleSizes:
    """Bundle radii (before scaling with args.scale)."""
    w_ct: float = 4.62  # Crista terminalis radius
    w_pm: float = 0.66  # Pectinate muscle radius
    w_bb: float = 2.0   # Bachmann bundle radius
    r_sn: float = 2.5   # Sinus node radius
    tv_ring_radius: float = 4.0  # radius around TV ring points
    ct_band_query_radius: float = 7.0  # kd-tree query radius for CT band
    cs_ring_radius: float = 4.0  # radius around CS ring points
    bb_sphere_factor: float = 1.02  # sphere radius = bb_radius * this


@dataclass
class PectinateMuscleConfig:
    """Parameters controlling pectinate muscle generation."""
    pm_num: int = 15  # total number of pectinate muscles
    pm_skip_first: int = 2  # skip first N PMs (IVC side)
    pm_skip_last: int = 2  # skip last N PMs (edge artifacts)
    pm_trim_fraction: float = 0.03  # trim first X% of path (roof of RA)
    pm_downsample_fraction: float = 0.065  # downsample step as fraction of path length
    pm1_downsample_fraction: float = 0.1  # first PM (to RAA apex)
    pm_min_path_length: int = 3  # minimum points for valid PM
    band_width_fraction: float = 0.15  # adaptive band width in dijkstra_path_on_a_plane


@dataclass
class BachmannBundleConfig:
    """Parameters for Bachmann bundle generation."""
    bb_downsample_fraction: float = 0.1 # downsample step as fraction of BB path length
    bb_center_fraction: float = 0.45  # where on BB path to place the bridge center
    bb_step: int = 20 # downsampling step for bridge path


@dataclass
class RAFiberConfig:
    """Main configuration for RA fiber generation."""
    thresholds: RAThresholds = field(default_factory=RAThresholds)
    bundles: BundleSizes = field(default_factory=BundleSizes)
    pm: PectinateMuscleConfig = field(default_factory=PectinateMuscleConfig)
    bb: BachmannBundleConfig = field(default_factory=BachmannBundleConfig)


_instance: Optional[RAFiberConfig] = None


def get_config() -> RAFiberConfig:
    global _instance
    if _instance is None:
        _instance = RAFiberConfig()
    return _instance
