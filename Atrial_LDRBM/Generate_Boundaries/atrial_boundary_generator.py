import os
import subprocess
import shutil
from glob import glob
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

from mesh_handler import generate_mesh  # Legacy mesh generation remains here.
from epi_endo_separator import separate_epi_endo
from surface_id_generator import generate_surf_id
from tag_loader import load_element_tags
from file_manager import write_vtk, write_obj, write_csv, write_vtx_file

from vtk_opencarp_helper_methods.vtk_methods.finder import find_closest_point
from vtk_opencarp_helper_methods.vtk_methods.init_objects import init_connectivity_filter, ExtractionModes
from vtk_opencarp_helper_methods.vtk_methods.thresholding import get_threshold_between
from vtk_opencarp_helper_methods.vtk_methods.converters import vtk_to_numpy
from vtk_opencarp_helper_methods.vtk_methods.filters import generate_ids, apply_vtk_geom_filter

# Import the new MeshReader class for reading meshes.
from mesh import MeshReader
# Import the new RingDetector class.
from ring_detector import RingDetector


class AtrialBoundaryGenerator:
    """
    Orchestrates atrial boundary generation using a modular, object‑oriented approach.
    Uses separate modules for:
      - Mesh handling (via MeshReader)
      - Ring detection (via RingDetector)
      - Epi/Endo separation
      - Surface ID generation
      - Tag loading

    :param mesh_path: Path to the mesh file.
    :param la_apex: Index for left atrial apex.
    :param ra_apex: Index for right atrial apex.
    :param la_base: Index for left atrial base.
    :param ra_base: Index for right atrial base.
    :param debug: Enables verbose output.
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

        self.ring_info: Dict[str, Any] = {}
        self.element_tags: Dict[str, str] = {}

    def _get_base_mesh(self) -> str:
        """Returns the base mesh filename (without extension)."""
        base, _ = os.path.splitext(self.mesh_path)
        return base

    def _prepare_output_directory(self, suffix: str = "_surf") -> str:
        """
        Prepares the output directory by creating it if needed and removing any stale ID files.
        """
        base = self._get_base_mesh()
        outdir = f"{base}{suffix}"
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        for file_path in glob(os.path.join(outdir, 'ids_*')):
            os.remove(file_path)
        return outdir

    def _format_id(self, idx: int) -> str:
        """Converts an index to a string (or returns empty if None)."""
        return str(idx) if idx is not None else ""

    def generate_mesh(self, la_mesh_scale: float = 1.0) -> None:
        """
        Generates a volumetric mesh using meshtool.
        """
        from mesh_handler import generate_mesh as gm  # Using legacy function here.
        gm(self.mesh_path, la_mesh_scale)
        if self.debug:
            print("Mesh generated.")

    def _process_LA_region(self, mesh: vtk.vtkPolyData, outdir: str) -> Dict[str, Any]:
        """
        Processes the left atrial region:
          - Retrieves the LA apex point.
          - Applies connectivity filtering and thresholding.
          - Generates IDs and detects rings using RingDetector.
          - Updates boundary tags and centroids.
        """
        LA_ap_point = mesh.GetPoint(int(self.la_apex))

        # Connectivity filtering to isolate the LA region.
        mesh_conn = init_connectivity_filter(mesh, ExtractionModes.ALL_REGIONS, True).GetOutput()
        arr = mesh_conn.GetPointData().GetArray("RegionId")
        arr.SetName("RegionID")
        id_vector = vtk_to_numpy(arr)
        new_LAA_id = find_closest_point(mesh_conn, LA_ap_point)
        LA_tag = id_vector[int(new_LAA_id)]

        la_thresh = get_threshold_between(mesh_conn, LA_tag, LA_tag,
                                          "vtkDataObject::FIELD_ASSOCIATION_POINTS", "RegionID")
        la_poly = apply_vtk_geom_filter(la_thresh.GetOutputPort(), True)
        LA_region = generate_ids(la_poly, "Ids", "Ids")
        write_vtk(os.path.join(outdir, 'LA.vtk'), LA_region)

        # Adjust the apex using the processed region.
        adjusted_LAA = find_closest_point(LA_region, LA_ap_point)
        b_tag = np.zeros((LA_region.GetNumberOfPoints(),))

        # Use RingDetector class for ring detection and marking.
        detector = RingDetector(LA_region, LA_ap_point, outdir)
        rings = detector.detect_rings()
        b_tag, centroids = detector.mark_la_rings(adjusted_LAA, rings, b_tag, {}, LA_region)

        ds = dsa.WrapDataObject(LA_region)
        ds.PointData.append(b_tag, 'boundary_tag')
        write_vtk(os.path.join(outdir, 'LA_boundaries_tagged.vtk'), ds.VTKObject)

        return centroids

    def _process_RA_region(self, mesh: vtk.vtkPolyData, outdir: str) -> Dict[str, Any]:
        """
        Processes the right atrial region:
          - Retrieves the RA apex point.
          - Applies connectivity filtering and thresholding.
          - Generates IDs and detects rings using RingDetector.
          - Updates boundary tags and centroids.
        """
        RA_ap_point = mesh.GetPoint(int(self.ra_apex))

        mesh_conn = init_connectivity_filter(mesh, ExtractionModes.ALL_REGIONS, True).GetOutput()
        arr = mesh_conn.GetPointData().GetArray("RegionId")
        arr.SetName("RegionID")
        id_vector = vtk_to_numpy(arr)
        new_RAA_id = find_closest_point(mesh_conn, RA_ap_point)
        RA_tag = id_vector[int(new_RAA_id)]

        ra_thresh = get_threshold_between(mesh_conn, RA_tag, RA_tag,
                                          "vtkDataObject::FIELD_ASSOCIATION_POINTS", "RegionID")
        ra_poly = apply_vtk_geom_filter(ra_thresh.GetOutputPort(), True)
        RA_region = generate_ids(ra_poly, "Ids", "Ids")
        write_vtk(os.path.join(outdir, 'RA.vtk'), RA_region)

        adjusted_RAA = find_closest_point(RA_region, RA_ap_point)
        b_tag = np.zeros((RA_region.GetNumberOfPoints(),))

        detector = RingDetector(RA_region, RA_ap_point, outdir)
        rings = detector.detect_rings()
        b_tag, centroids, rings = detector.mark_ra_rings(adjusted_RAA, rings, b_tag, {})

        detector.cutting_plane_to_identify_tv_f_tv_s(RA_region, rings, True)

        ds = dsa.WrapDataObject(RA_region)
        ds.PointData.append(b_tag, 'boundary_tag')
        write_vtk(os.path.join(outdir, 'RA_boundaries_tagged.vtk'), ds.VTKObject)

        return centroids

    def extract_rings(self) -> None:
        """
        Orchestrates the ring extraction process by reading the mesh (via MeshReader),
        processing the LA and/or RA regions, and writing centroids to CSV.
        """
        # Use MeshReader to read the mesh.
        mesh_obj = MeshReader(self.mesh_path)
        mesh = mesh_obj.get_polydata()
        outdir = self._prepare_output_directory("_surf")
        centroids = {}

        if self.la_apex is not None and self.ra_apex is not None:
            centroids.update(self._process_LA_region(mesh, outdir))
            centroids.update(self._process_RA_region(mesh, outdir))
        elif self.ra_apex is None and self.la_apex is not None:
            centroids.update(self._process_LA_region(mesh, outdir))
        elif self.la_apex is None and self.ra_apex is not None:
            centroids.update(self._process_RA_region(mesh, outdir))
        else:
            raise ValueError("At least one of LA or RA apex must be provided.")

        write_csv(os.path.join(outdir, "rings_centroids.csv"), pd.DataFrame(centroids))
        self.ring_info = centroids
        if self.debug:
            print("Ring extraction complete. Centroids saved.")

    def separate_epi_endo(self, atrium: str) -> None:
        """
        Delegates epicardial/endocardial separation to the epi_endo_separator module.
        """
        separate_epi_endo(self.mesh_path, atrium, self.element_tags)
        if self.debug:
            print(f"Epi/Endo separation completed for {atrium}.")

    def generate_surf_id(self, atrium: str, resampled: bool = False) -> None:
        """
        Delegates surface ID generation to the surface_id_generator module.
        """
        generate_surf_id(self.mesh_path, atrium, resampled)
        if self.debug:
            print(f"Surface ID generation completed for {atrium}.")

    def load_element_tags(self, csv_filepath: str) -> None:
        """
        Loads element tags using tag_loader.
        """
        self.element_tags = load_element_tags(csv_filepath)
        if self.debug:
            print("Element tags loaded.")

    def extract_rings_top_epi_endo(self) -> None:
        """
        Delegates top epi/endo ring extraction to the top_epi_endo module,
        then loads computed centroids from CSV.
        """
        LAA_id = self._format_id(self.la_apex)
        RAA_id = self._format_id(self.ra_apex)
        LAA_base_id = self._format_id(self.la_base)
        RAA_base_id = self._format_id(self.ra_base)
        print("Running top epi/endo ring extraction with parameters:")
        print(f"  Mesh: {self.mesh_path}")
        print(f"  LAA: {LAA_id}, RAA: {RAA_id}, LAA_base: {LAA_base_id}, RAA_base: {RAA_base_id}")
        from top_epi_endo import label_atrial_orifices_TOP_epi_endo
        label_atrial_orifices_TOP_epi_endo(
            mesh=self.mesh_path,
            LAA_id=LAA_id,
            RAA_id=RAA_id,
            LAA_base_id=LAA_base_id,
            RAA_base_id=RAA_base_id
        )
        base = self._get_base_mesh()
        outdir = f"{base}_surf"
        csv_path = os.path.join(outdir, "rings_centroids.csv")
        if os.path.exists(csv_path):
            print(f"Loading top epi/endo ring centroids from {csv_path}")
            self.ring_info = pd.read_csv(csv_path).to_dict(orient="list")
        else:
            print(f"Warning: Top epi/endo ring centroids file not found at {csv_path}")
