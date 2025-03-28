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

from mesh_handler import load_mesh, generate_mesh
from ring_detector import Ring, detect_and_mark_rings, mark_LA_rings, mark_RA_rings, cutting_plane_to_identify_UAC, cutting_plane_to_identify_RSPV, cutting_plane_to_identify_tv_f_tv_s
from epi_endo_separator import separate_epi_endo
from surface_id_generator import generate_surf_id
from tag_loader import load_element_tags
from file_manager import write_vtk, write_obj, write_csv, write_vtx_file

from vtk_opencarp_helper_methods.vtk_methods.finder import find_closest_point
from vtk_opencarp_helper_methods.vtk_methods.init_objects import init_connectivity_filter, ExtractionModes
from vtk_opencarp_helper_methods.vtk_methods.thresholding import get_threshold_between
from vtk_opencarp_helper_methods.vtk_methods.converters import vtk_to_numpy
from vtk_opencarp_helper_methods.vtk_methods.filters import generate_ids, apply_vtk_geom_filter

# Import the top epi/endo extraction function from our new module.
from top_epi_endo import label_atrial_orifices_TOP_epi_endo


class AtrialBoundaryGenerator:
    """
    Orchestrates atrial boundary generation using a modular, object‑oriented approach.
    Uses separate modules for:
      - Mesh handling
      - Ring detection (including TV splitting and UAC identification)
      - Epi/Endo separation
      - Surface ID generation
      - Tag loading

    :param mesh_path: Path to the mesh file.
    :param la_apex: Index for left atrial apex.
    :param ra_apex: Index for right atrial apex.
    :param la_base: Index for left atrial base.
    :param ra_base: Index for right atrial base.
    :param debug: Enables verbose output.
    :return: An instance of AtrialBoundaryGenerator.
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

    # Helper Methods
    def _get_base_mesh(self) -> str:
        """
        Returns the base mesh filename (without extension).
        :return: Base mesh filename as a string.
        """
        base, _ = os.path.splitext(self.mesh_path)
        return base

    def _prepare_output_directory(self, suffix: str = "_surf") -> str:
        """
        Prepares and cleans the output directory by creating it if needed and
        removing any previously generated ID files.

        :param suffix: Suffix to append to the base mesh filename.
        :return: The output directory path.
        """
        base = self._get_base_mesh()
        outdir = f"{base}{suffix}"
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        # Remove previous ID files to avoid stale data.
        for file_path in glob(os.path.join(outdir, 'ids_*')):
            os.remove(file_path)
        return outdir

    def _format_id(self, idx: int) -> str:
        """
        Converts an index to a string.

        :param idx: An integer index.
        :return: The index as a string, or empty if None.
        """
        return str(idx) if idx is not None else ""

    # Mesh Generation
    def generate_mesh(self, la_mesh_scale: float = 1.0) -> None:
        """
        Generates a volumetric mesh using meshtool. The mesh is saved with the suffix "_vol".

        :param la_mesh_scale: A scaling factor for the mesh (if needed).
        :return: None.
        """
        generate_mesh(self.mesh_path, la_mesh_scale)
        if self.debug:
            print("Mesh generated.")

    # Ring Extraction: Helper for Left Atrial Region
    def _process_LA_region(self, mesh: vtk.vtkPolyData, outdir: str) -> Dict[str, Any]:
        """
        Processes the left atrial region:
          - Retrieves the LA apex point.
          - Applies connectivity filtering and thresholding.
          - Generates IDs and detects rings.
          - Marks rings using clustering.
          - Writes out the processed LA region and its boundaries.

        :param mesh: The input mesh as a VTK polydata object.
        :param outdir: The directory where LA-related files will be saved.
        :return: A dictionary of computed centroids for the LA region.
        """
        # Retrieve the LA apex point from the mesh.
        LA_ap_point = mesh.GetPoint(int(self.la_apex))

        # Apply connectivity filtering to segment the mesh.
        mesh_conn = init_connectivity_filter(mesh, ExtractionModes.ALL_REGIONS, True).GetOutput()
        arr = mesh_conn.GetPointData().GetArray("RegionId")
        arr.SetName("RegionID")
        id_vector = vtk_to_numpy(arr)
        new_LAA_id = find_closest_point(mesh_conn, LA_ap_point)
        LA_tag = id_vector[int(new_LAA_id)]

        # Apply thresholding to isolate the LA region.
        la_thresh = get_threshold_between(mesh_conn, LA_tag, LA_tag,
                                          "vtkDataObject::FIELD_ASSOCIATION_POINTS", "RegionID")
        la_poly = apply_vtk_geom_filter(la_thresh.GetOutputPort(), True)
        # Generate IDs to ensure consistency.
        LA_region = generate_ids(la_poly, "Ids", "Ids")

        # Write out the raw LA region.
        write_vtk(os.path.join(outdir, 'LA.vtk'), LA_region)

        # Adjust the apex point to the processed region.
        adjusted_LAA = find_closest_point(LA_region, LA_ap_point)
        boundary_tag = np.zeros((LA_region.GetNumberOfPoints(),))

        # Detect rings within the LA region.
        centroids = {}
        rings = detect_and_mark_rings(LA_region, LA_ap_point, outdir)

        # Mark the rings using clustering and update boundary tags and centroids.
        boundary_tag, centroids = mark_LA_rings(adjusted_LAA, rings, boundary_tag, centroids, outdir, LA_region)

        # Wrap the LA region for adding extra point data and write boundaries.
        ds = dsa.WrapDataObject(LA_region)
        ds.PointData.append(boundary_tag, 'boundary_tag')
        write_vtk(os.path.join(outdir, 'LA_boundaries_tagged.vtk'), ds.VTKObject)

        return centroids

    # Ring Extraction: Helper for Right Atrial Region
    def _process_RA_region(self, mesh: vtk.vtkPolyData, outdir: str) -> Dict[str, Any]:
        """
        Processes the right atrial region:
          - Retrieves the RA apex point.
          - Applies connectivity filtering and thresholding.
          - Generates IDs and detects rings.
          - Marks rings using clustering.
          - Performs TV splitting to segment the tricuspid valve.
          - Writes out the processed RA region and its boundaries.

        :param mesh: The input mesh as a VTK polydata object.
        :param outdir: The directory where RA-related files will be saved.
        :return: A dictionary of computed centroids for the RA region.
        """
        # Retrieve the RA apex point.
        RA_ap_point = mesh.GetPoint(int(self.ra_apex))

        # Apply connectivity filtering to isolate the RA region.
        mesh_conn = init_connectivity_filter(mesh, ExtractionModes.ALL_REGIONS, True).GetOutput()
        arr = mesh_conn.GetPointData().GetArray("RegionId")
        arr.SetName("RegionID")
        id_vector = vtk_to_numpy(arr)
        new_RAA_id = find_closest_point(mesh_conn, RA_ap_point)
        RA_tag = id_vector[int(new_RAA_id)]

        # Thresholding for the RA region.
        ra_thresh = get_threshold_between(mesh_conn, RA_tag, RA_tag,
                                          "vtkDataObject::FIELD_ASSOCIATION_POINTS", "RegionID")
        ra_poly = apply_vtk_geom_filter(ra_thresh.GetOutputPort(), True)
        RA_region = generate_ids(ra_poly, "Ids", "Ids")

        # Write out the raw RA region.
        write_vtk(os.path.join(outdir, 'RA.vtk'), RA_region)

        # Adjust the RA apex point.
        adjusted_RAA = find_closest_point(RA_region, RA_ap_point)
        boundary_tag = np.zeros((RA_region.GetNumberOfPoints(),))

        # Detect rings within the RA region.
        centroids = {}
        rings = detect_and_mark_rings(RA_region, RA_ap_point, outdir)
        boundary_tag, centroids, rings = mark_RA_rings(adjusted_RAA, rings, boundary_tag, centroids, outdir)

        # Perform TV splitting to separate tricuspid valve segments.
        cutting_plane_to_identify_tv_f_tv_s(RA_region, rings, outdir, self.debug)

        # Wrap the RA region to attach boundary tags and write the boundaries.
        ds = dsa.WrapDataObject(RA_region)
        ds.PointData.append(boundary_tag, 'boundary_tag')
        write_vtk(os.path.join(outdir, 'RA_boundaries_tagged.vtk'), ds.VTKObject)

        return centroids

    # Main Ring Extraction Orchestration
    def extract_rings(self) -> None:
        """
        Orchestrates the ring extraction process.

        This method loads the mesh, prepares the output directory, and then based on the provided apex values,
        processes the LA region, RA region, or both. It then writes a CSV file containing the computed centroids.

        :return: None
        """
        mesh = load_mesh(self.mesh_path)
        outdir = self._prepare_output_directory("_surf")
        centroids = {}

        # Check for biatrial, LA-only, or RA-only processing:
        if self.la_apex is not None and self.ra_apex is not None:
            centroids.update(self._process_LA_region(mesh, outdir))
            centroids.update(self._process_RA_region(mesh, outdir))
        elif self.ra_apex is None and self.la_apex is not None:
            centroids.update(self._process_LA_region(mesh, outdir))
        elif self.la_apex is None and self.ra_apex is not None:
            centroids.update(self._process_RA_region(mesh, outdir))
        else:
            raise ValueError("At least one of LA or RA apex must be provided.")

        # Write the centroids to a CSV file using our centralized I/O function.
        write_csv(os.path.join(outdir, "rings_centroids.csv"), pd.DataFrame(centroids))
        self.ring_info = centroids
        if self.debug:
            print("Ring extraction complete. Centroids saved.")

    # Epi/Endo Separation
    def separate_epi_endo(self, atrium: str) -> None:
        """
        Delegates epicardial and endocardial separation to the epi_endo_separate module.

        :param atrium: A string ("LA" or "RA") indicating which atrium to process.
        :return: None.
        """
        separate_epi_endo(self.mesh_path, atrium, self.element_tags)
        if self.debug:
            print(f"Epi/Endo separation completed for {atrium}.")

    # Surface ID Generation
    def generate_surf_id(self, atrium: str, resampled: bool = False) -> None:
        """
        Delegates surface ID generation to the surface_id_generator module.

        :param atrium: A string ("LA" or "RA") indicating which atrium to process.
        :param resampled: Boolean flag to indicate if resampling is used.
        :return: None.
        """
        generate_surf_id(self.mesh_path, atrium, resampled)
        if self.debug:
            print(f"Surface ID generation completed for {atrium}.")

    # Tag Loading
    def load_element_tags(self, csv_filepath: str) -> None:
        """
        Loads element tags from a CSV file using the tag_loader module.

        :param csv_filepath: The file path to the CSV containing tag mappings.
        :return: None.
        """
        self.element_tags = load_element_tags(csv_filepath)
        if self.debug:
            print("Element tags loaded.")

    # Top Epi/Endo Extraction
    def extract_rings_top_epi_endo(self) -> None:
        """
        Delegates top epi/endo ring extraction to the top_epi_endo module,
        and then loads the computed centroids from the output CSV.

        :return: None.
        """
        LAA_id = self._format_id(self.la_apex)
        RAA_id = self._format_id(self.ra_apex)
        LAA_base_id = self._format_id(self.la_base)
        RAA_base_id = self._format_id(self.ra_base)
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
        base = self._get_base_mesh()
        outdir = f"{base}_surf"
        csv_path = os.path.join(outdir, "rings_centroids.csv")
        if os.path.exists(csv_path):
            print(f"Loading top epi/endo ring centroids from {csv_path}")
            self.ring_info = pd.read_csv(csv_path).to_dict(orient="list")
        else:
            print(f"Warning: Top epi/endo ring centroids file not found at {csv_path}")
