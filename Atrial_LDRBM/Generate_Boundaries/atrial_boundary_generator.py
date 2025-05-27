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

from Atrial_LDRBM.Generate_Boundaries.epi_endo_separator import separate_epi_endo
from Atrial_LDRBM.Generate_Boundaries.surface_id_generator import generate_surf_id
from Atrial_LDRBM.Generate_Boundaries.tag_loader import load_element_tags
from Atrial_LDRBM.Generate_Boundaries.file_manager import write_vtk, write_obj, write_csv, write_vtx_file
from Atrial_LDRBM.Generate_Boundaries.surface_id_generator import generate_surf_id as gen_surf_id_func
from Atrial_LDRBM.Generate_Boundaries.mesh import MeshReader
from Atrial_LDRBM.Generate_Boundaries.ring_detector import RingDetector

from vtk_opencarp_helper_methods.vtk_methods.finder import find_closest_point
from vtk_opencarp_helper_methods.vtk_methods.init_objects import init_connectivity_filter, ExtractionModes
from vtk_opencarp_helper_methods.vtk_methods.thresholding import get_threshold_between
from vtk_opencarp_helper_methods.vtk_methods.converters import vtk_to_numpy
from vtk_opencarp_helper_methods.vtk_methods.filters import generate_ids, apply_vtk_geom_filter

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

        self.polydata = None
        if os.path.exists(self.mesh_path):
            try:
                reader = MeshReader(self.mesh_path)
                self.polydata = reader.get_polydata()

                if self.debug and (self.polydata is None or
                                   self.polydata.GetNumberOfPoints() == 0):
                    print(f"Warning: Initial mesh loaded from {self.mesh_path} "
                          f"is empty or invalid.")
            except Exception as e:
                if self.debug:
                    print(f"Warning: Could not load initial mesh "
                          f"{self.mesh_path} in AtrialBoundaryGenerator "
                          f"constructor: {e}")
        else:
            if self.debug:
                print(f"Warning: Initial mesh_path {self.mesh_path} provided to "
                      f"AtrialBoundaryGenerator does not exist.")

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

        os.makedirs(outdir, exist_ok=True)

        for file_path in glob(os.path.join(outdir, 'ids_*')):
            os.remove(file_path)

        return outdir

    def _format_id(self, idx: int) -> str:
        """Converts an index to a string (or returns empty if None)."""
        return str(idx) if idx is not None else ""

    def _run_meshtool_volume_generation(self, input_obj_path: str, output_base_path: str) -> None:
        """
        Internal method to execute the meshtool command for volume generation.

        Args:
            input_obj_path (str): Path to the input surface mesh in OBJ format.
            output_base_path (str): Base path for the output volumetric mesh (e.g., /path/to/mesh_RA_vol).
                                     Meshtool will append the format extension (.vtk).
        """
        command = ["meshtool",
                   "generate",
                   "mesh",
                   "-surf=" + input_obj_path,
                   "-ofmt=vtk",
                   "-outmsh=" + output_base_path
                   ]

        if self.debug:
            print(f"Running meshtool command: {' '.join(command)}")

        try:
            subprocess.run(command, check=True, capture_output=True, text=True)

            if self.debug:
                print(f"Meshtool completed. Output expected at ~{output_base_path}.vtk")

        except FileNotFoundError:
            print("ERROR: 'meshtool' command not found. Ensure it is installed and in the system PATH.")
            raise

        except subprocess.CalledProcessError as e:
            # Error if meshtool returns a non-zero exit code
            print(f"ERROR: meshtool execution failed with code {e.returncode}")
            print(f"Meshtool stdout:\n{e.stdout}")
            print(f"Meshtool stderr:\n{e.stderr}")
            raise RuntimeError("meshtool volume generation failed") from e

    def generate_mesh(self, input_surface_path: str) -> None:
        """
        Generates a volumetric mesh from the given input surface path using meshtool.
        Ensures the input to meshtool is in OBJ format, converting if necessary.

        Args:
            input_surface_path (str): Path to the input surface mesh (any format readable by MeshReader).
        """
        base_path, input_ext = os.path.splitext(input_surface_path)

        # Define the standard output naming convention relative to the input base path
        output_base_path = f"{base_path}_vol"

        # Define the required OBJ input path for meshtool
        input_obj_path = f"{base_path}.obj"

        # Ensure OBJ input exists for meshtool
        obj_input_ready = False

        if os.path.exists(input_obj_path):
            # If the target OBJ file already exists, it's usable
            obj_input_ready = True

            if self.debug:
                print(f"Using existing OBJ file: {input_obj_path}")

        elif os.path.exists(input_surface_path):
            try:
                reader = MeshReader(input_surface_path)
                polydata = reader.get_polydata()

                if not polydata or polydata.GetNumberOfPoints() == 0:
                    raise ValueError("MeshReader returned empty polydata.")

                write_obj(input_obj_path, polydata)

                if self.debug:
                    print(f"Successfully converted to {input_obj_path}")

                obj_input_ready = True

            except Exception as e:
                print(f"ERROR: Failed to convert {input_surface_path} to OBJ for meshtool. {e}")
                raise RuntimeError(f"Failed OBJ conversion required for meshtool") from e
        else:
            # Neither the input path nor the derived obj path exists
            raise FileNotFoundError(f"Input surface file not found: {input_surface_path}")

        # Run meshtool if OBJ input is ready
        if obj_input_ready:
            self._run_meshtool_volume_generation(input_obj_path, output_base_path)
        else:
            print("ERROR: OBJ input for meshtool could not be prepared.")
            raise RuntimeError("Failed to prepare OBJ input for meshtool")

    # ---------------------------------------------------------------------
    #  Modified LA processing – now adds LAA / LAA_base to centroid output
    # ---------------------------------------------------------------------
    def _process_LA_region(self,
                           input_mesh_polydata: vtk.vtkPolyData,
                           outdir: str,
                           is_biatrial: bool) -> dict:
        """
        Processes the LA region and returns a centroid dictionary that now
        also includes the LAA apex (“LAA”) and the LAA base (“LAA_base”)
        coordinates for the final CSV.
        """
        if self.la_apex is None:
            return {}

        # -------------------------------------------------- CSV coordinates
        la_ap_point_coord_for_csv = None
        la_bs_point_coord_for_csv = None
        if hasattr(self, "polydata") and self.polydata is not None:
            if (self.la_apex is not None and
                    0 <= self.la_apex < self.polydata.GetNumberOfPoints()):
                la_ap_point_coord_for_csv = self.polydata.GetPoint(self.la_apex)

            if (self.la_base is not None and
                    0 <= self.la_base < self.polydata.GetNumberOfPoints()):
                la_bs_point_coord_for_csv = self.polydata.GetPoint(self.la_base)

        # -------------------------------------------------- isolate region
        if is_biatrial:
            try:
                la_ap_point_coord = input_mesh_polydata.GetPoint(self.la_apex)

                mesh_conn = init_connectivity_filter(
                    input_mesh_polydata, ExtractionModes.ALL_REGIONS, True
                ).GetOutput()

                arr = mesh_conn.GetPointData().GetArray("RegionId")
                arr.SetName("RegionID")
                id_vector = vtk_to_numpy(arr)

                temp_LAA_id = find_closest_point(mesh_conn, la_ap_point_coord)
                LA_tag_val = id_vector[temp_LAA_id]

                la_thresh = get_threshold_between(
                    mesh_conn, LA_tag_val, LA_tag_val,
                    "vtkDataObject::FIELD_ASSOCIATION_POINTS", "RegionID"
                )

                la_poly_ug = la_thresh.GetOutput()
                la_poly = apply_vtk_geom_filter(la_poly_ug)

                if la_poly and la_poly.GetNumberOfPoints() > 0:
                    la_region_polydata = generate_ids(la_poly, "Ids", "Ids")
                else:
                    print("Warning: LA region extraction resulted in empty mesh.")
                    return {}

            except Exception as e:
                print(f"ERROR during LA region extraction: {e}")
                return {}
        else:
            la_region_polydata = input_mesh_polydata
            try:
                la_ap_point_coord = la_region_polydata.GetPoint(self.la_apex)
            except IndexError:
                print(f"ERROR: LA apex ID {self.la_apex} out of bounds.")
                return {}

        # -------------------------------------------------- ring detection
        if la_region_polydata and la_ap_point_coord:
            write_vtk(os.path.join(outdir, "LA.vtk"), la_region_polydata)

            adjusted_LAA = find_closest_point(la_region_polydata, la_ap_point_coord)
            if adjusted_LAA < 0:
                adjusted_LAA = self.la_apex

            apex_coord_for_detector = la_region_polydata.GetPoint(adjusted_LAA)

            try:
                detector_LA = RingDetector(la_region_polydata,
                                           apex_coord_for_detector,
                                           outdir)
                rings_la = detector_LA.detect_rings(debug=self.debug)

                b_tag_la, la_centroids = detector_LA.mark_la_rings(
                    adjusted_LAA,
                    rings_la,
                    np.zeros(la_region_polydata.GetNumberOfPoints()),
                    {},
                    la_region_polydata,
                    debug=self.debug
                )

                ds_la = dsa.WrapDataObject(la_region_polydata)
                ds_la.PointData.append(b_tag_la, "boundary_tag")
                write_vtk(os.path.join(outdir, "LA_boundaries_tagged.vtk"),
                          ds_la.VTKObject)

                # -------------------- add apex / base --------------------
                final_la_centroids = la_centroids.copy()
                if la_ap_point_coord_for_csv is not None:
                    final_la_centroids["LAA"] = la_ap_point_coord_for_csv
                if la_bs_point_coord_for_csv is not None:
                    final_la_centroids["LAA_base"] = la_bs_point_coord_for_csv

                return final_la_centroids

            except Exception as e:
                print(f"ERROR during LA ring detection/marking: {e}")
                return {}

        return {}

    # ---------------------------------------------------------------------
    #  Modified RA processing – now adds RAA / RAA_base to centroid output
    # ---------------------------------------------------------------------
    def _process_RA_region(self,
                           input_mesh_polydata: vtk.vtkPolyData,
                           outdir: str,
                           is_biatrial: bool) -> dict:
        """
        Processes the RA region and returns a centroid dictionary that now
        also includes the RAA apex (“RAA”) and the RAA base (“RAA_base”)
        coordinates for the final CSV.
        """
        if self.ra_apex is None:
            return {}

        # -------------------------------------------------- CSV coordinates
        ra_ap_point_coord_for_csv = None
        ra_bs_point_coord_for_csv = None
        if hasattr(self, "polydata") and self.polydata is not None:
            if (self.ra_apex is not None and
                    0 <= self.ra_apex < self.polydata.GetNumberOfPoints()):
                ra_ap_point_coord_for_csv = self.polydata.GetPoint(self.ra_apex)

            if (self.ra_base is not None and
                    0 <= self.ra_base < self.polydata.GetNumberOfPoints()):
                ra_bs_point_coord_for_csv = self.polydata.GetPoint(self.ra_base)

        # -------------------------------------------------- isolate region
        if is_biatrial:
            try:
                ra_ap_point_coord = input_mesh_polydata.GetPoint(self.ra_apex)

                mesh_conn = init_connectivity_filter(
                    input_mesh_polydata, ExtractionModes.ALL_REGIONS, True
                ).GetOutput()

                arr = mesh_conn.GetPointData().GetArray("RegionId")
                arr.SetName("RegionID")
                id_vector = vtk_to_numpy(arr)

                temp_RAA_id = find_closest_point(mesh_conn, ra_ap_point_coord)
                RA_tag_val = id_vector[temp_RAA_id]

                ra_thresh = get_threshold_between(
                    mesh_conn, RA_tag_val, RA_tag_val,
                    "vtkDataObject::FIELD_ASSOCIATION_POINTS", "RegionID"
                )

                ra_poly_ug = ra_thresh.GetOutput()
                ra_poly = apply_vtk_geom_filter(ra_poly_ug)

                if ra_poly and ra_poly.GetNumberOfPoints() > 0:
                    ra_region_polydata = generate_ids(ra_poly, "Ids", "Ids")
                else:
                    print("Warning: RA region extraction resulted in empty mesh.")
                    return {}
            except Exception as e:
                print(f"ERROR during RA region extraction: {e}")
                return {}
        else:
            ra_region_polydata = input_mesh_polydata
            try:
                ra_ap_point_coord = ra_region_polydata.GetPoint(self.ra_apex)
            except IndexError:
                print(f"ERROR: RA apex ID {self.ra_apex} out of bounds.")
                return {}

        # -------------------------------------------------- ring detection
        if ra_region_polydata and ra_ap_point_coord:
            write_vtk(os.path.join(outdir, "RA.vtk"), ra_region_polydata)

            adjusted_RAA = find_closest_point(ra_region_polydata, ra_ap_point_coord)
            if adjusted_RAA < 0:
                adjusted_RAA = self.ra_apex

            apex_coord_for_detector = ra_region_polydata.GetPoint(adjusted_RAA)

            try:
                detector_RA = RingDetector(ra_region_polydata,
                                           apex_coord_for_detector,
                                           outdir)
                rings_ra = detector_RA.detect_rings(debug=self.debug)

                b_tag_ra, ra_centroids, rings_ra_obj = detector_RA.mark_ra_rings(
                    adjusted_RAA,
                    rings_ra,
                    np.zeros(ra_region_polydata.GetNumberOfPoints()),
                    {},
                    debug=self.debug
                )

                detector_RA.cutting_plane_to_identify_tv_f_tv_s(
                    ra_region_polydata, rings_ra_obj, debug=self.debug
                )

                ds_ra = dsa.WrapDataObject(ra_region_polydata)
                ds_ra.PointData.append(b_tag_ra, "boundary_tag")
                write_vtk(os.path.join(outdir, "RA_boundaries_tagged.vtk"),
                          ds_ra.VTKObject)

                # -------------------- add apex / base --------------------
                final_ra_centroids = ra_centroids.copy()
                if ra_ap_point_coord_for_csv is not None:
                    final_ra_centroids["RAA"] = ra_ap_point_coord_for_csv
                if ra_bs_point_coord_for_csv is not None:
                    final_ra_centroids["RAA_base"] = ra_bs_point_coord_for_csv

                return final_ra_centroids

            except Exception as e:
                print(f"ERROR during RA ring detection/marking: {e}")
                return {}

        return {}

    def extract_rings(self, surface_mesh_path: str) -> None:
        """
        Orchestrates the ring extraction process by reading the mesh (via MeshReader),
        processing the LA and/or RA regions, and writing centroids to CSV.
        """
        if self.debug:
            print(f"Initiating standard ring extraction on: {surface_mesh_path}")

        # Prepare output directory
        # outdir = self._prepare_output_directory("_surf")
        surface_base = os.path.splitext(surface_mesh_path)[0]
        outdir = f"{surface_base}_surf"
        os.makedirs(outdir, exist_ok=True)

        # Clear previous ids_* from this specific outdir
        for f_path in glob(os.path.join(outdir, 'ids_*')):
            os.remove(f_path)

        # Read the specified surface mesh
        try:
            mesh_obj = MeshReader(surface_mesh_path)
            input_mesh_polydata = mesh_obj.get_polydata()

            if not input_mesh_polydata.GetPointData() or not input_mesh_polydata.GetPointData().GetArray("Ids"):
                input_mesh_polydata = generate_ids(input_mesh_polydata, "Ids", "Ids")

        except Exception as e:
            print(f"ERROR reading surface mesh {surface_mesh_path}: {e}")
            raise

        centroids = {}

        if self.la_apex is not None and self.ra_apex is not None:
            # Biatrial case: process both from the input mesh
            la_centroids = self._process_LA_region(input_mesh_polydata, outdir, is_biatrial=True)
            centroids.update(la_centroids)
            ra_centroids = self._process_RA_region(input_mesh_polydata, outdir, is_biatrial=True)
            centroids.update(ra_centroids)

        elif self.la_apex is not None:
            # LA only case: process input mesh as LA
            la_centroids = self._process_LA_region(input_mesh_polydata, outdir, is_biatrial=False)
            centroids.update(la_centroids)

        elif self.ra_apex is not None:
            # RA only case: process input mesh as RA
            ra_centroids = self._process_RA_region(input_mesh_polydata, outdir, is_biatrial=False)
            centroids.update(ra_centroids)
        else:
            raise ValueError("No apex ID provided for extract_rings.")

        processed_centroids_for_df = {key: list(value) for key, value in centroids.items()}
        df = pd.DataFrame(processed_centroids_for_df)
        csv_path = os.path.join(outdir, "rings_centroids.csv")
        write_csv(csv_path, df)

        self.ring_info = centroids


    def separate_epi_endo(self, tagged_volume_mesh_path: str, atrium: str) -> None:
        """
        Delegates epi/endo separation using the epi_endo_separator module.
        Requires element tags loaded via load_element_tags() beforehand.

        Args:
            tagged_volume_mesh_path (str): Path to the tagged volume mesh.
            atrium (str): 'LA' or 'RA'.
        """
        if not self.element_tags:
            raise RuntimeError("Element tags missing in AtrialBoundaryGenerator. Call load_element_tags() before separate_epi_endo().")

        try:
            separate_epi_endo(mesh_path=tagged_volume_mesh_path,
                              atrium=atrium,
                              element_tags=self.element_tags)

        except Exception as e:
            print(f"ERROR during epi/endo separation call for {atrium} on {tagged_volume_mesh_path}: {e}")
            raise

    def generate_surf_id(self, volumetric_mesh_path: str, atrium: str, resampled: bool = False) -> None:
        """
        Delegates surface ID generation to the surface_id_generator module.
        """
        if self.debug:
            print(f"Initiating surface ID generation for {atrium} using volume mesh: {volumetric_mesh_path}")

        try:
            gen_surf_id_func(vol_mesh_path=volumetric_mesh_path,
                             atrium=atrium,
                             resampled=resampled,
                             debug=self.debug)
            if self.debug:
                print(f"Surface ID generation call completed for {atrium}.")

        except Exception as e:
            print(f"ERROR during surface ID generation for {atrium}: {e}")

    def load_element_tags(self, csv_filepath: str) -> None:
        self.element_tags = load_element_tags(csv_filepath)

    def extract_rings_top_epi_endo(self, surface_mesh_path: str) -> None:
        """
        Specialized workflow (TOP_EPI/ENDO).  Now also guarantees that “RAA”
        and “RAA_base” coordinates—taken from the **original** reference mesh
        via `self.polydata`—are written to rings_centroids.csv, matching the
        behaviour of the legacy script.
        """
        # -------- set-up & load ------------------------------------------------
        surface_base = os.path.splitext(surface_mesh_path)[0]
        outdir = f"{surface_base}_surf"
        os.makedirs(outdir, exist_ok=True)

        for f_path in glob(os.path.join(outdir, "ids_*")):
            os.remove(f_path)

        reader = MeshReader(surface_mesh_path)
        mesh_pd = reader.get_polydata()
        if mesh_pd.GetPointData().GetArray("Ids") is None:
            mesh_pd = generate_ids(mesh_pd, "Ids", "Ids")

        centroids: Dict[str, Tuple[float, float, float]] = {}

        # --------------------- LA (optional) ----------------------------------
        if self.la_apex is not None:
            la_cents = self._process_LA_region(mesh_pd, outdir, is_biatrial=True)
            centroids.update(la_cents)

        # --------------------- RA base workflow -------------------------------
        if self.ra_apex is None:
            raise ValueError("RA apex index (ra_apex) is required for TOP_EPI/ENDO")

        # isolate RA region
        ra_ap_pt = mesh_pd.GetPoint(self.ra_apex)
        conn = init_connectivity_filter(mesh_pd, ExtractionModes.ALL_REGIONS, True).GetOutput()
        arr = conn.GetPointData().GetArray("RegionId");
        arr.SetName("RegionID")
        tags = vtk_to_numpy(arr)
        tag_val = int(tags[find_closest_point(conn, ra_ap_pt)])

        thr = get_threshold_between(conn, tag_val, tag_val,
                                    "vtkDataObject::FIELD_ASSOCIATION_POINTS", "RegionID")
        ra_region = generate_ids(apply_vtk_geom_filter(thr.GetOutputPort()), "Ids", "Ids")
        write_vtk(os.path.join(outdir, "RA.vtp"), ra_region)

        # detect + mark rings
        detector = RingDetector(ra_region, ra_ap_pt, outdir)
        rings = detector.detect_rings(debug=self.debug)
        adj_id = find_closest_point(ra_region, ra_ap_pt)
        b_tag, ra_cents, rings = detector.mark_ra_rings(
            adj_id,
            rings,
            np.zeros(ra_region.GetNumberOfPoints(), dtype=int),
            {},
            debug=self.debug
        )
        centroids.update(ra_cents)

        # --------------------- ADD RAA / RAA_base -----------------------------
        if (hasattr(self, "polydata") and self.polydata is not None and
                self.polydata.GetNumberOfPoints() > 0):

            # RAA
            if self.ra_apex is not None and 0 <= self.ra_apex < self.polydata.GetNumberOfPoints():
                raa_coord = self.polydata.GetPoint(self.ra_apex)
                centroids["RAA"] = raa_coord
                if self.debug:
                    print(f"Added RAA coordinate {raa_coord} to TOP_EPI/ENDO centroids.")
            elif self.ra_apex is not None and self.debug:
                print(f"Warning: ra_apex ID {self.ra_apex} invalid for self.polydata.")

            # RAA_base
            if self.ra_base is not None and 0 <= self.ra_base < self.polydata.GetNumberOfPoints():
                raa_base_coord = self.polydata.GetPoint(self.ra_base)
                centroids["RAA_base"] = raa_base_coord
                if self.debug:
                    print(f"Added RAA_base coordinate {raa_base_coord} to TOP_EPI/ENDO centroids.")
            elif self.ra_base is not None and self.debug:
                print(f"Warning: ra_base ID {self.ra_base} invalid for self.polydata.")
        elif self.debug:
            print("Warning: self.polydata unavailable—RAA/RAA_base not added.")

        # --------------------- continue workflow ------------------------------
        ds_ra = dsa.WrapDataObject(ra_region)
        ds_ra.PointData.append(b_tag, "boundary_tag")
        write_vtk(os.path.join(outdir, "RA_boundaries_tagged.vtk"), ds_ra.VTKObject)

        endo_path = f"{self._get_base_mesh()}_endo.obj"
        if not os.path.exists(endo_path):
            raise FileNotFoundError(f"Expected endo mesh at {endo_path}; run separate_epi_endo first")

        detector.perform_tv_split_and_find_top_epi_endo(
            model_epi=ra_region,
            endo_mesh_path=endo_path,
            rings=rings,
            debug=self.debug
        )

        # --------------------- write CSV --------------------------------------
        df = pd.DataFrame.from_dict(centroids, orient="index", columns=["X", "Y", "Z"])
        df.index.name = "RingName"
        write_csv(os.path.join(outdir, "rings_centroids.csv"), df)

        self.ring_info = centroids
        if self.debug:
            print("TOP_EPI/ENDO centroids:", self.ring_info)
