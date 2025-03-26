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

# Module-level imports from VTK helper modules
from vtk_opencarp_helper_methods.vtk_methods.reader import smart_reader
from vtk_opencarp_helper_methods.vtk_methods.filters import (
    apply_vtk_geom_filter, clean_polydata, generate_ids, get_center_of_mass,
    get_feature_edges, get_elements_above_plane
)
from vtk_opencarp_helper_methods.vtk_methods.exporting import vtk_polydata_writer, write_to_vtx, vtk_obj_writer
from vtk_opencarp_helper_methods.vtk_methods.finder import find_closest_point
from vtk_opencarp_helper_methods.vtk_methods.init_objects import (
    init_connectivity_filter, ExtractionModes, initialize_plane_with_points, initialize_plane
)
from vtk_opencarp_helper_methods.vtk_methods.converters import vtk_to_numpy, numpy_to_vtk
from vtk_opencarp_helper_methods.mathematical_operations.vector_operations import get_normalized_cross_product
from vtk_opencarp_helper_methods.vtk_methods.thresholding import get_lower_threshold, get_threshold_between

# For tag loading from the procedural module
from Atrial_LDRBM.Generate_Boundaries.separate_epi_endo import load_element_tags
# For TV splitting, import the procedural split_tv function
from Atrial_LDRBM.Generate_Boundaries.extract_rings import split_tv
# Import ring detector module functions and Ring class
from ring_detector import (
    Ring, detect_and_mark_rings, mark_LA_rings, mark_RA_rings,
    cutting_plane_to_identify_UAC, cutting_plane_to_identify_RSPV,
    cutting_plane_to_identify_tv_f_tv_s
)
# Import mesh handling functions
from mesh_handler import load_mesh, generate_mesh


class AtrialBoundaryGenerator:
    """
    This class implements:
      1. Mesh generation via meshtool.
      2. Ring extraction (standard and top epi/endo) from the mesh.
      3. Epicardial/endocardial surface separation.
      4. Surface ID generation.
      5. Tag loading.

    Attributes:
        mesh_path (str): Path to the mesh file.
        la_apex (int): Index for left atrial apex.
        ra_apex (int): Index for right atrial apex.
        la_base (int): Index for left atrial base.
        ra_base (int): Index for right atrial base.
        debug (bool): Enables verbose output.
        ring_info (dict): Stores computed ring centroids and related data.
        element_tags (dict): Stores element tags for epi/endo separation.
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

    # --------------------------
    # Helper Methods
    # --------------------------
    def _get_base_mesh(self) -> str:
        base, _ = os.path.splitext(self.mesh_path)
        return base

    def _prepare_output_directory(self, suffix: str = "_surf") -> str:
        base = self._get_base_mesh()
        outdir = f"{base}{suffix}"
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        for f in glob(os.path.join(outdir, 'ids_*')):
            os.remove(f)
        return outdir

    def _format_id(self, idx: int) -> str:
        return str(idx) if idx is not None else ""

    # --------------------------
    # 1. Mesh Generation
    # --------------------------
    def generate_mesh(self, la_mesh_scale: float = 1.0) -> None:
        generate_mesh(self.mesh_path, la_mesh_scale)
        if self.debug:
            print("Mesh generated.")

    # --------------------------
    # 2. Standard Ring Extraction
    # --------------------------
    def extract_rings(self) -> None:
        mesh = load_mesh(self.mesh_path)
        outdir = self._prepare_output_directory("_surf")
        centroids = {}

        if self.la_apex is not None and self.ra_apex is not None:
            # Biatrial workflow:
            LA_ap_point = mesh.GetPoint(int(self.la_apex))
            RA_ap_point = mesh.GetPoint(int(self.ra_apex))
            centroids["LAA"] = LA_ap_point
            centroids["RAA"] = RA_ap_point
            if self.la_base is not None and self.ra_base is not None:
                centroids["LAA_base"] = mesh.GetPoint(int(self.la_base))
                centroids["RAA_base"] = mesh.GetPoint(int(self.ra_base))

            mesh_conn = init_connectivity_filter(mesh, ExtractionModes.ALL_REGIONS, True).GetOutput()
            arr = mesh_conn.GetPointData().GetArray("RegionId")
            arr.SetName("RegionID")
            id_vec = vtk_to_numpy(arr)
            new_LAA_id = find_closest_point(mesh_conn, LA_ap_point)
            new_RAA_id = find_closest_point(mesh_conn, RA_ap_point)
            LA_tag = id_vec[int(new_LAA_id)]
            RA_tag = id_vec[int(new_RAA_id)]

            # Process LA region:
            la_thresh = get_threshold_between(mesh_conn, LA_tag, LA_tag,
                                              "vtkDataObject::FIELD_ASSOCIATION_POINTS", "RegionID")
            la_poly = apply_vtk_geom_filter(la_thresh.GetOutputPort(), True)
            LA_region = generate_ids(la_poly, "Ids", "Ids")
            vtk_polydata_writer(os.path.join(outdir, 'LA.vtk'), LA_region)
            adjusted_LAA = find_closest_point(LA_region, LA_ap_point)
            b_tag = np.zeros((LA_region.GetNumberOfPoints(),))
            rings = detect_and_mark_rings(LA_region, LA_ap_point, outdir)
            b_tag, centroids = mark_LA_rings(adjusted_LAA, rings, b_tag, centroids, outdir, LA_region)
            ds = dsa.WrapDataObject(LA_region)
            ds.PointData.append(b_tag, 'boundary_tag')
            vtk_polydata_writer(os.path.join(outdir, 'LA_boundaries_tagged.vtk'), ds.VTKObject)

            # Process RA region:
            ra_thresh = get_threshold_between(mesh_conn, RA_tag, RA_tag,
                                              "vtkDataObject::FIELD_ASSOCIATION_POINTS", "RegionID")
            ra_poly = apply_vtk_geom_filter(ra_thresh.GetOutputPort(), True)
            RA_region = generate_ids(ra_poly, "Ids", "Ids")
            vtk_polydata_writer(os.path.join(outdir, 'RA.vtk'), RA_region)
            adjusted_RAA = find_closest_point(RA_region, RA_ap_point)
            b_tag_ra = np.zeros((RA_region.GetNumberOfPoints(),))
            rings_ra = detect_and_mark_rings(RA_region, RA_ap_point, outdir)
            b_tag_ra, centroids, rings_ra = mark_RA_rings(adjusted_RAA, rings_ra, b_tag_ra, centroids, outdir)
            # Invoke TV splitting for RA:
            cutting_plane_to_identify_tv_f_tv_s(RA_region, rings_ra, outdir, self.debug)
            ds_ra = dsa.WrapDataObject(RA_region)
            ds_ra.PointData.append(b_tag_ra, 'boundary_tag')
            vtk_polydata_writer(os.path.join(outdir, 'RA_boundaries_tagged.vtk'), ds_ra.VTKObject)

        elif self.ra_apex is None and self.la_apex is not None:
            # LA-only processing:
            LA_ap_point = mesh.GetPoint(int(self.la_apex))
            centroids["LAA"] = LA_ap_point
            if self.la_base is not None:
                centroids["LAA_base"] = mesh.GetPoint(int(self.la_base))
            mesh_conn = init_connectivity_filter(mesh, ExtractionModes.ALL_REGIONS, True).GetOutput()
            arr = mesh_conn.GetPointData().GetArray("RegionId")
            arr.SetName("RegionID")
            id_vec = vtk_to_numpy(arr)
            new_LAA_id = find_closest_point(mesh_conn, LA_ap_point)
            LA_tag = id_vec[int(new_LAA_id)]
            la_thresh = get_threshold_between(mesh_conn, LA_tag, LA_tag,
                                              "vtkDataObject::FIELD_ASSOCIATION_POINTS", "RegionID")
            la_poly = apply_vtk_geom_filter(la_thresh.GetOutputPort(), True)
            LA_region = generate_ids(la_poly, "Ids", "Ids")
            vtk_polydata_writer(os.path.join(outdir, 'LA.vtk'), LA_region)
            adjusted_LAA = find_closest_point(LA_region, LA_ap_point)
            b_tag = np.zeros((LA_region.GetNumberOfPoints(),))
            rings = detect_and_mark_rings(LA_region, LA_ap_point, outdir)
            b_tag, centroids = mark_LA_rings(adjusted_LAA, rings, b_tag, centroids, outdir, LA_region)
            ds = dsa.WrapDataObject(LA_region)
            ds.PointData.append(b_tag, 'boundary_tag')
            vtk_polydata_writer(os.path.join(outdir, 'LA_boundaries_tagged.vtk'), ds.VTKObject)
        elif self.la_apex is None and self.ra_apex is not None:
            # RA-only processing:
            RA_ap_point = mesh.GetPoint(int(self.ra_apex))
            centroids["RAA"] = RA_ap_point
            if self.ra_base is not None:
                centroids["RAA_base"] = mesh.GetPoint(int(self.ra_base))
            mesh_conn = init_connectivity_filter(mesh, ExtractionModes.ALL_REGIONS, True).GetOutput()
            arr = mesh_conn.GetPointData().GetArray("RegionId")
            arr.SetName("RegionID")
            id_vec = vtk_to_numpy(arr)
            new_RAA_id = find_closest_point(mesh_conn, RA_ap_point)
            RA_tag = id_vec[int(new_RAA_id)]
            ra_thresh = get_threshold_between(mesh_conn, RA_tag, RA_tag,
                                              "vtkDataObject::FIELD_ASSOCIATION_POINTS", "RegionID")
            ra_poly = apply_vtk_geom_filter(ra_thresh.GetOutputPort(), True)
            RA_region = generate_ids(ra_poly, "Ids", "Ids")
            vtk_polydata_writer(os.path.join(outdir, 'RA.vtk'), RA_region)
            adjusted_RAA = find_closest_point(RA_region, RA_ap_point)
            b_tag = np.zeros((RA_region.GetNumberOfPoints(),))
            rings = detect_and_mark_rings(RA_region, RA_ap_point, outdir)
            b_tag, centroids, rings = mark_RA_rings(adjusted_RAA, rings, b_tag, centroids, outdir)
            # Invoke TV splitting for RA:
            cutting_plane_to_identify_tv_f_tv_s(RA_region, rings, outdir, self.debug)
            ds = dsa.WrapDataObject(RA_region)
            ds.PointData.append(b_tag, 'boundary_tag')
            vtk_polydata_writer(os.path.join(outdir, 'RA_boundaries_tagged.vtk'), ds.VTKObject)
        else:
            raise ValueError("At least one of LA or RA apex must be provided.")

        df = pd.DataFrame(centroids)
        csv_path = os.path.join(outdir, "rings_centroids.csv")
        df.to_csv(csv_path, float_format="%.2f", index=False)
        self.ring_info = centroids
        if self.debug:
            print("Ring extraction complete. Centroids saved.")

    # --------------------------
    # Epi/Endo Separation
    # --------------------------
    def separate_epi_endo(self, atrium: str) -> None:
        if atrium not in ["LA", "RA"]:
            raise ValueError("Atrium must be 'LA' or 'RA'.")
        model = smart_reader(self.mesh_path)
        if not self.element_tags:
            self.element_tags = load_element_tags('path/to/element_tag.csv')
        if atrium == "LA":
            epi_tag = int(self.element_tags.get('left_atrial_wall_epi', 0))
            endo_tag = int(self.element_tags.get('left_atrial_wall_endo', 0))
        else:
            epi_tag = int(self.element_tags.get('right_atrial_wall_epi', 0))
            endo_tag = int(self.element_tags.get('right_atrial_wall_endo', 0))
        combined_thresh = get_threshold_between(model, endo_tag, epi_tag,
                                                "vtkDataObject::FIELD_ASSOCIATION_CELLS", "tag")
        filtered_combined = apply_vtk_geom_filter(combined_thresh.GetOutput())
        outdir = self._prepare_output_directory("_vol_surf")
        vtk_polydata_writer(os.path.join(outdir, f"{atrium}.vtk"), filtered_combined)
        vtk_obj_writer(os.path.join(outdir, f"{atrium}.obj"), filtered_combined)
        epi_thresh = get_threshold_between(model, epi_tag, epi_tag,
                                           "vtkDataObject::FIELD_ASSOCIATION_CELLS", "tag")
        filtered_epi = apply_vtk_geom_filter(epi_thresh.GetOutput())
        vtk_polydata_writer(os.path.join(outdir, f"{atrium}_epi.vtk"), filtered_epi)
        vtk_obj_writer(os.path.join(outdir, f"{atrium}_epi.obj"), filtered_epi)
        endo_thresh = get_threshold_between(model, endo_tag, endo_tag,
                                            "vtkDataObject::FIELD_ASSOCIATION_CELLS", "tag")
        filtered_endo = apply_vtk_geom_filter(endo_thresh.GetOutput())
        vtk_polydata_writer(os.path.join(outdir, f"{atrium}_endo.vtk"), filtered_endo)
        vtk_obj_writer(os.path.join(outdir, f"{atrium}_endo.obj"), filtered_endo)
        if self.debug:
            print(f"Epi/Endo separation completed for {atrium}.")

    # --------------------------
    # Surface ID Generation
    # --------------------------
    def generate_surf_id(self, atrium: str, resampled: bool = False) -> None:
        base = self.mesh_path
        vol = smart_reader(f"{base}_{atrium}_vol.vtk")
        coords = vtk_to_numpy(vol.GetPoints().GetData())
        tree = cKDTree(coords)
        epi_obj = smart_reader(f"{base}_{atrium}_epi.obj")
        epi_pts = vtk_to_numpy(epi_obj.GetPoints().GetData())
        _, epi_indices = tree.query(epi_pts)
        epi_ids = np.array(epi_indices)
        outdir = f"{base}_{atrium}_vol_surf"
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        shutil.copyfile(f"{base}_{atrium}_vol.vtk", os.path.join(outdir, f"{atrium}.vtk"))
        res_str = "_res" if resampled else ""
        shutil.copyfile(f"{base}_{atrium}_epi{res_str}_surf/rings_centroids.csv",
                        os.path.join(outdir, "rings_centroids.csv"))
        write_to_vtx(os.path.join(outdir, "EPI.vtx"), epi_indices)
        endo_obj = smart_reader(f"{base}_{atrium}_endo.obj")
        endo_pts = vtk_to_numpy(endo_obj.GetPoints().GetData())
        _, endo_indices = tree.query(endo_pts)
        endo_indices = np.setdiff1d(endo_indices, epi_ids)
        write_to_vtx(os.path.join(outdir, "ENDO.vtx"), endo_indices)
        if self.debug:
            print(f"Surface ID generation completed for {atrium}.")

    # --------------------------
    # Tag Loading
    # --------------------------
    def load_element_tags(self, csv_filepath: str) -> None:
        self.element_tags = load_element_tags(csv_filepath)
        if self.debug:
            print("Element tags loaded.")

    # --------------------------
    # Top Epi/Endo Extraction
    # --------------------------
    def extract_rings_top_epi_endo(self) -> None:
        LAA_id = self._format_id(self.la_apex)
        RAA_id = self._format_id(self.ra_apex)
        LAA_base_id = self._format_id(self.la_base)
        RAA_base_id = self._format_id(self.ra_base)
        if self.debug:
            print("Running top epi/endo ring extraction with parameters:")
            print(f"  Mesh: {self.mesh_path}")
            print(f"  LAA: {LAA_id}, RAA: {RAA_id}, LAA_base: {LAA_base_id}, RAA_base: {RAA_base_id}")
        from Atrial_LDRBM.Generate_Boundaries.extract_rings_TOP_epi_endo import label_atrial_orifices_TOP_epi_endo
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
            if self.debug:
                print(f"Loading top epi/endo ring centroids from {csv_path}")
            self.ring_info = pd.read_csv(csv_path).to_dict(orient="list")
        else:
            if self.debug:
                print(f"Warning: Top epi/endo ring centroids file not found at {csv_path}")


# --------------------------
# Example usage:
# --------------------------
if __name__ == '__main__':
    generator = AtrialBoundaryGenerator(mesh_path="path/to/mesh.vtk",
                                        la_apex=123,
                                        ra_apex=456,
                                        la_base=789,
                                        ra_base=1011,
                                        debug=True)
    generator.generate_mesh()
    generator.extract_rings()
    generator.extract_rings_top_epi_endo()
    generator.separate_epi_endo("LA")
    generator.generate_surf_id("LA", resampled=False)
    generator.load_element_tags("path/to/element_tag.csv")
    print("Ring information:", generator.ring_info)
