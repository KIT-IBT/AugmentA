import os
import subprocess
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

# Direct imports from original files
from Atrial_LDRBM.Generate_Boundaries.extract_rings import label_atrial_orifices
from Atrial_LDRBM.Generate_Boundaries.extract_rings_TOP_epi_endo import label_atrial_orifices_TOP_epi_endo
from Atrial_LDRBM.Generate_Boundaries.generate_mesh import generate_mesh
from Atrial_LDRBM.Generate_Boundaries.generate_surf_id import generate_surf_id
from Atrial_LDRBM.Generate_Boundaries.separate_epi_endo import separate_epi_endo, load_element_tags

# Optional imports for more direct VTK manipulation if needed
# from vtk.numpy_interface import dataset_adapter as dsa
# from scipy.spatial import cKDTree
# from sklearn.cluster import KMeans
# import vtk

class AtrialBoundaryGenerator:
    """
    A class to handle generation and labeling of atrial boundaries:
      1) Generating a volume mesh from surface files.
      2) Extracting rings from the mesh.
      3) Specialized ring extraction for top epi/endo.
      4) Separating epicardial/endocardial surfaces.
      5) Generating surface IDs.

    Attributes:
        mesh_path (str): Path to the mesh/surface file (e.g., without extension or with a known extension).
        la_apex (int): Index for LA apex (if present).
        ra_apex (int): Index for RA apex (if present).
        la_base (int): Index for LA base (if present).
        ra_base (int): Index for RA base (if present).
        debug (bool): Toggle verbosity or debug outputs.
        ring_info (dict): Holds ring-related data (e.g., centroids, boundary tags).
        element_tags (dict): For storing epi/endo tags loaded from CSV.
    """

    def __init__(self,
                 mesh_path: str,
                 la_apex: int = None,
                 ra_apex: int = None,
                 la_base: int = None,
                 ra_base: int = None,
                 debug: bool = True):
        self.mesh_path = mesh_path
        self.la_apex = la_apex
        self.ra_apex = ra_apex
        self.la_base = la_base
        self.ra_base = ra_base
        self.debug = debug

        # Dictionary to hold ring centroids, boundary tags, etc.
        self.ring_info: Dict[str, Any] = {}

        # Dictionary to hold element tags loaded from CSV (for epi/endo separation)
        self.element_tags: Dict[str, str] = {}

    # Get the base mesh filename (without extension)
    def _get_base_mesh(self) -> str:
        base, ext = os.path.splitext(self.mesh_path)
        return base
    # ----------------------------------------------------------------
    # 1. Generate a Volume Mesh (from generate_mesh.py)
    # ----------------------------------------------------------------
    def generate_mesh(self, la_mesh_scale: float = 1.0) -> None:
        """
        Calls meshtool to generate a volumetric mesh (VTK format) from an .obj surface file.
        The output mesh is written to: "<mesh_path>_vol".

        Args:
            la_mesh_scale (float): Scaling factor (if needed).
        """
        self._run_meshtool(self.mesh_path, la_mesh_scale)

    def _run_meshtool(self, path: str, la_mesh_scale: float) -> None:
        """
        Private method that calls the external meshtool.

        Args:
            path (str): The mesh path (assumed to correspond to an .obj file).
            la_mesh_scale (float): Mesh scale (currently commented out in the call).
        """
        subprocess.run([
            "meshtool",
            "generate",
            "mesh",
            # f"-scale={la_mesh_scale}",
            f"-surf={path}.obj",
            "-ofmt=vtk",
            f"-outmsh={path}_vol"
        ], check=True)

    # ----------------------------------------------------------------
    # 2. Standard Ring Extraction (from extract_rings.py)
    # ----------------------------------------------------------------
    def extract_rings(self) -> None:
        """
        Wraps the standard ring extraction.
        Reads the mesh, detects and marks atrial rings (for LA and/or RA),
        writes output VTK files, and updates self.ring_info.
        """
        self._label_atrial_orifices_standard()

    def _format_id(self, idx: int) -> str:
        """Helper to convert an index to string, or return an empty string if None."""
        return str(idx) if idx is not None else ""

    def _label_atrial_orifices_standard(self) -> None:
        """
        Wraps the procedural function label_atrial_orifices() and updates self.ring_info
        with the resulting ring centroids loaded from CSV.
        """
        LAA_id = self._format_id(self.la_apex)
        RAA_id = self._format_id(self.ra_apex)
        LAA_base_id = self._format_id(self.la_base)
        RAA_base_id = self._format_id(self.ra_base)

        if self.debug:
            print("Running standard ring extraction with parameters:")
            print(f"  Mesh: {self.mesh_path}")
            print(f"  LAA: {LAA_id}, RAA: {RAA_id}, LAA_base: {LAA_base_id}, RAA_base: {RAA_base_id}")

        # Call the original function. This will write output files to a folder.
        label_atrial_orifices(
            mesh=self.mesh_path,
            LAA_id=LAA_id,
            RAA_id=RAA_id,
            LAA_base_id=LAA_base_id,
            RAA_base_id=RAA_base_id,
            debug=int(self.debug)
        )

        # Load ring centroids from CSV into ring_info.
        base = self._get_base_mesh()
        csv_path = os.path.join(f"{base}_surf", "rings_centroids.csv")
        if os.path.exists(csv_path):
            if self.debug:
                print(f"Loading ring centroids from {csv_path}")
            self.ring_info = pd.read_csv(csv_path).to_dict(orient="list")
        else:
            if self.debug:
                print(f"Warning: Ring centroids file not found at {csv_path}")


    # ----------------------------------------------------------------
    # 3. Specialized Ring Extraction for Top Epi/Endo
    #    (from extract_rings_TOP_epi_endo.py)
    # ----------------------------------------------------------------
    def extract_rings_top_epi_endo(self) -> None:
        """
        Wraps the specialized ring extraction for top epi/endo.
        Calls the corresponding procedural function and updates ring_info.
        """
        self._label_orifices_top_endo()

    def _label_orifices_top_endo(self) -> None:
        """
        Private method that wraps label_atrial_orifices_TOP_epi_endo().
        """
        from Atrial_LDRBM.Generate_Boundaries.extract_rings_TOP_epi_endo import label_atrial_orifices_TOP_epi_endo

        LAA_id = str(self.la_apex) if self.la_apex is not None else ""
        RAA_id = str(self.ra_apex) if self.ra_apex is not None else ""
        LAA_base_id = str(self.la_base) if self.la_base is not None else ""
        RAA_base_id = str(self.ra_base) if self.ra_base is not None else ""

        if self.debug:
            print("Running top epi/endo ring extraction with parameters:")
            print(f"  Mesh: {self.mesh_path}")
            print(f"  LAA: {LAA_id}, RAA: {RAA_id}, LAA_base: {LAA_base_id}, RAA_base: {RAA_base_id}")

        label_atrial_orifices_TOP_epi_endo(
            mesh=self.mesh_path,
            LAA_id=LAA_id,
            RAA_id=RAA_id,
            LAA_base_id=LAA_base_id,
            RAA_base_id=RAA_base_id
        )

        # Load ring centroids from CSV (assumed to be in the same outdir)
        base = self._get_base_mesh()
        csv_path = os.path.join(f"{base}_surf", "rings_centroids.csv")
        if os.path.exists(csv_path):
            if self.debug:
                print(f"Loading top epi/endo ring centroids from {csv_path}")
            self.ring_info = pd.read_csv(csv_path).to_dict(orient="list")
        else:
            if self.debug:
                print(f"Warning: Top epi/endo ring centroids file not found at {csv_path}")


    # ----------------------------------------------------------------
    # 4. Separating Epicardial/Endocardial Surfaces (separate_epi_endo.py)
    # ----------------------------------------------------------------
    def separate_epi_endo(self, atrium: str) -> None:
        """
        Separates the epicardial and endocardial surfaces.

        Args:
            atrium (str): Should be "LA" or "RA".
        """
        self._separate_epi_endo_impl(atrium)

    def _separate_epi_endo_impl(self, atrium: str) -> None:
        """
        Private method that wraps the procedural function separate_epi_endo().
        """
        from Atrial_LDRBM.Generate_Boundaries.separate_epi_endo import separate_epi_endo
        if self.debug:
            print(f"Separating epi/endo surfaces for {atrium} using mesh: {self.mesh_path}")
        separate_epi_endo(path=self.mesh_path, atrium=atrium)

    # ----------------------------------------------------------------
    # 5. Generating Surface IDs (from generate_surf_id.py)
    # ----------------------------------------------------------------
    def generate_surf_id(self, atrium: str, resampled: bool = False) -> None:
        """
        Generates surface ID files (in .vtx format) that map the volumetric mesh
        to various boundaries (EPI, ENDO, and ring-based IDs).

        Args:
            atrium (str): "LA" or "RA"
            resampled (bool): If True, uses the resampled epi surface.
        """
        self._generate_surf_id_impl(atrium, resampled)

    def _generate_surf_id_impl(self, atrium: str, resampled: bool) -> None:
        """
        Private method that wraps generate_surf_id().
        """
        from Atrial_LDRBM.Generate_Boundaries.generate_surf_id import generate_surf_id
        if self.debug:
            print(f"Generating surface IDs for {atrium} (resampled={resampled}) using mesh: {self.mesh_path}")
        generate_surf_id(meshname=self.mesh_path, atrium=atrium, resampled=resampled)

    # ----------------------------------------------------------------
    # Optional Tag-Loading or Helper Methods
    # (If needed for thresholding, ring detection, etc.)
    # ----------------------------------------------------------------
    def load_element_tags(self, csv_filepath: str) -> None:
        """
        Loads element tags from a CSV file (for use in epi/endo separation).

        Args:
            csv_filepath (str): Path to the CSV file containing element tag mappings.
        """
        from Atrial_LDRBM.Generate_Boundaries.separate_epi_endo import load_element_tags
        if self.debug:
            print(f"Loading element tags from {csv_filepath}")
        self.element_tags = load_element_tags(csv_filepath)

    # Add as many private helpers as you need
    # to unify repeated code from extract_rings.py & extract_rings_TOP_epi_endo.py.

