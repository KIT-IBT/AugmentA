from pathlib import Path

# Configuration dictionary for all path components and naming conventions.
# Modifying these values will change the output paths and names across the pipeline.
PATH_COMPONENTS = {
    "surface_dir_suffix": "_surf",
    "epi_suffix": "_epi",
    "vol_suffix": "_vol",
    "cut_suffix": "_cutted",
    "fit_suffix": "_fit",
    "resampled_suffix": "_res",
    "fibers_dir_suffix": "_fibers",
    "mesh_data_suffix": "_mesh_data.csv",
    "bilayer_fiber_suffix": "_bilayer_with_fiber",
    "vol_fiber_suffix": "_vol_with_fiber"
}


class WorkflowPaths:
    """
    A stateful class that centrally replicates the exact file and directory paths
    of the procedural AugmentA pipeline.
    """

    def __init__(self, initial_mesh_path: str, atrium: str):
        self.initial_mesh = Path(initial_mesh_path).resolve()
        self.atrium = atrium
        self._active_stage = 'initial'
        self._base_paths = {'initial': self.initial_mesh.with_suffix('')}

    def register_stage_completion(self, stage_name: str, base_path: str):
        self._active_stage = stage_name
        self._base_paths[stage_name] = Path(base_path).with_suffix('')
        print(f"Stage '{stage_name}' completed. Active mesh base is now: {self._base_paths[stage_name]}")

    @property
    def active_mesh_base(self) -> Path:
        """Returns the base path (path without extension) of the last stage."""
        return self._base_paths[self._active_stage]

    @property
    def surf_dir(self) -> Path:
        """
        Replicates the procedural `f"{mesh_name}_surf"` directory.
        The name is based on the currently active mesh base path.
        """
        active_base = self.active_mesh_base
        return active_base.with_name(f"{active_base.name}{PATH_COMPONENTS['surface_dir_suffix']}")

    @property
    def initial_mesh_base(self) -> Path:
        """The base path of the original input mesh."""
        return self.initial_mesh.with_suffix('')

    @property
    def initial_mesh_dir(self) -> Path:
        """The directory of the original input mesh."""
        return self.initial_mesh.parent

    @property
    def initial_mesh_ext(self) -> str:
        """The extension of the original input mesh."""
        return self.initial_mesh.suffix

    @property
    def mesh_data_csv(self) -> Path:
        """Path for the LAA/RAA apex ID csv file, named after the active mesh."""
        return self.active_mesh_base.with_suffix('.csv').with_name(f"{self.active_mesh_base.name}{PATH_COMPONENTS['mesh_data_suffix']}")

    @property
    def closed_surface_epi_mesh(self) -> Path:
        """Path for the separated epicardial mesh in the closed surface workflow."""
        return self.initial_mesh_base.with_name(f"{self.initial_mesh_base.name}_{self.atrium}{PATH_COMPONENTS['epi_suffix']}")

    @property
    def closed_surface_vol_mesh(self) -> Path:
        """Path for the volumetric mesh in the closed surface workflow."""
        return self.initial_mesh_base.with_name(f"{self.initial_mesh_base.name}_{self.atrium}{PATH_COMPONENTS['vol_suffix']}")

    @property
    def cut_mesh(self) -> Path:
        """Path for the mesh after orifices are cut."""
        return self.initial_mesh_dir / f"{self.atrium}{PATH_COMPONENTS['cut_suffix']}"

    @property
    def ssm_target_dir(self) -> Path:
        """Path for the SSM target's surf directory."""
        return self.initial_mesh_dir / f"{self.atrium}{PATH_COMPONENTS['cut_suffix']}{PATH_COMPONENTS['surface_dir_suffix']}"


    @property
    def fit_mesh(self) -> Path:
        """Path for the SSM fitted mesh."""
        return self.ssm_target_dir / f"{self.atrium}{PATH_COMPONENTS['fit_suffix']}"

    @property
    def resampled_mesh(self) -> Path:
        """Path for the resampled mesh."""
        # The base for resampling depends on the active stage (fit or cut).
        return self.active_mesh_base.with_name(f"{self.active_mesh_base.name}{PATH_COMPONENTS['resampled_suffix']}")

    @property
    def fiber_base_dir(self) -> Path:
        """The base directory for fiber generation output, named after the active mesh."""
        return self.active_mesh_base.parent / f"{self.active_mesh_base.name}{PATH_COMPONENTS['fibers_dir_suffix']}"


    def final_bilayer_mesh(self, extension: str) -> Path:
        """
        Path for the final bilayer mesh with fibers.
        The result subdir can be LA or RA depending on the atrium.
        """
        # For LA_RA, the procedural script specifically constructed path using result_RA,
        # even if args.atrium was restored to "LA_RA". We need to respect this.
        # args.atrium at the point of plotting in procedural was its final state.
        if self.atrium == "LA_RA":
            atrium_for_subdir = "RA"
        else:
            atrium_for_subdir = self.atrium

        result_dir = self.fiber_base_dir / f"result_{atrium_for_subdir}"
        return result_dir / f"{self.atrium}{PATH_COMPONENTS['bilayer_fiber_suffix']}.{extension}"

    def final_volumetric_mesh(self, extension: str) -> Path:
        """Path for the final volumetric mesh with fibers."""
        result_dir = self.fiber_base_dir / f"result_{self.atrium}"
        return result_dir / f"{self.atrium}{PATH_COMPONENTS['vol_fiber_suffix']}.{extension}"
