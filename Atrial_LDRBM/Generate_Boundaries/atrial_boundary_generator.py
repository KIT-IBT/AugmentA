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

    def _process_LA_region(self, input_mesh_polydata: vtk.vtkPolyData, outdir: str, is_biatrial: bool) -> dict:
        """
        Processes the LA region (extracts if biatrial) and performs ring analysis.
        """
        if self.la_apex is None:
            # Cannot process without LA apex
            return {}

        if is_biatrial:
            try:
                # Get initial apex coordinate from the input mesh
                la_ap_point_coord = input_mesh_polydata.GetPoint(self.la_apex)

                mesh_conn = init_connectivity_filter(input_mesh_polydata, ExtractionModes.ALL_REGIONS, True).GetOutput()

                if not mesh_conn.GetPointData() or not mesh_conn.GetPointData().GetArray("RegionId"):
                    raise RuntimeError("Connectivity filter failed.")

                arr = mesh_conn.GetPointData().GetArray("RegionId")
                arr.SetName("RegionID")
                id_vector = vtk_to_numpy(arr)

                temp_LAA_id: int = find_closest_point(mesh_conn, la_ap_point_coord)
                if temp_LAA_id < 0:
                    raise ValueError("Could not re-locate LA apex after connectivity.")

                LA_tag_val: int = id_vector[temp_LAA_id]

                la_thresh = get_threshold_between(mesh_conn, LA_tag_val, LA_tag_val,"vtkDataObject::FIELD_ASSOCIATION_POINTS", "RegionID")
                la_poly_ug = la_thresh.GetOutput()
                la_poly = apply_vtk_geom_filter(la_poly_ug)

                if la_poly and la_poly.GetNumberOfPoints() > 0:
                    # Generate unique IDs for the isolated LA region
                    la_region_polydata = generate_ids(la_poly, "Ids", "Ids")
                else:
                    print("Warning: LA region extraction resulted in empty mesh.")
                    return {}
            except Exception as e:
                print(f"ERROR during LA region extraction: {e}")
                return {}

        else:
            # Assume the input polydata is already the LA region
            la_region_polydata = input_mesh_polydata
            try:
                # Still need the apex coordinate
                la_ap_point_coord = la_region_polydata.GetPoint(self.la_apex)
            except IndexError:
                print(f"ERROR: LA apex ID {self.la_apex} out of bounds for the provided LA mesh.")
                return {}

        # Perform Ring Detection on la_region_polydata
        if la_region_polydata and la_ap_point_coord:
            write_vtk(os.path.join(outdir, 'LA.vtk'), la_region_polydata)

            # Adjust apex ID for the isolated LA region polydata
            adjusted_LAA = find_closest_point(la_region_polydata, la_ap_point_coord)
            if adjusted_LAA < 0:
                print("Warning: Could not find closest point for LA apex in isolated region. Using original ID.")
                adjusted_LAA = self.la_apex  # Use original as fallback

            try:
                detector_LA = RingDetector(la_region_polydata, la_ap_point_coord, outdir)
                rings_la = detector_LA.detect_rings(debug=self.debug)

                # Marking rings and get centroids
                b_tag_la, la_centroids = detector_LA.mark_la_rings(adjusted_LAA,
                                                                   rings_la,
                                                                   np.zeros(la_region_polydata.GetNumberOfPoints()),
                                                                   {},
                                                                   la_region_polydata,
                                                                   debug=self.debug)
                # Save tagged LA mesh
                ds_la = dsa.WrapDataObject(la_region_polydata)
                ds_la.PointData.append(b_tag_la, 'boundary_tag')
                write_vtk(os.path.join(outdir, 'LA_boundaries_tagged.vtk'), ds_la.VTKObject)

                return la_centroids
            except Exception as e:
                print(f"ERROR during LA ring detection/marking: {e}")
                return {}
        else:
            # Return empty if region extraction failed
            return {}

    def _process_RA_region(self, input_mesh_polydata: vtk.vtkPolyData, outdir: str, is_biatrial: bool) -> dict:
        """
        Processes the right atrial region:
          - Retrieves the RA apex point.
          - Applies connectivity filtering and thresholding.
          - Generates IDs and detects rings using RingDetector.
          - Updates boundary tags and centroids.
        """
        if self.ra_apex is None:
            return {}

        if is_biatrial:
            try:
                ra_ap_point_coord = input_mesh_polydata.GetPoint(self.ra_apex)

                # Perform connectivity filtering and thresholding to isolate RA
                mesh_conn = init_connectivity_filter(input_mesh_polydata, ExtractionModes.ALL_REGIONS, True).GetOutput()
                if not mesh_conn.GetPointData() or not mesh_conn.GetPointData().GetArray("RegionId"):
                    raise RuntimeError("Connectivity filter failed.")

                arr = mesh_conn.GetPointData().GetArray("RegionId")
                arr.SetName("RegionID")

                id_vector = vtk_to_numpy(arr)

                temp_RAA_id = find_closest_point(mesh_conn, ra_ap_point_coord)
                if temp_RAA_id < 0:
                    raise ValueError("Could not re-locate RA apex after connectivity.")

                RA_tag_val = id_vector[temp_RAA_id]

                ra_thresh = get_threshold_between(mesh_conn,
                                                  RA_tag_val,
                                                  RA_tag_val,
                                                  "vtkDataObject::FIELD_ASSOCIATION_POINTS",
                                                  "RegionID")

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
                print(f"ERROR: RA apex ID {self.ra_apex} out of bounds for the provided RA mesh.")
                return {}

        # Perform Ring Detection on ra_region_polydata
        if ra_region_polydata and ra_ap_point_coord:
            write_vtk(os.path.join(outdir, 'RA.vtk'), ra_region_polydata)

            adjusted_RAA = find_closest_point(ra_region_polydata, ra_ap_point_coord)

            if adjusted_RAA < 0:
                print("Warning: Could not find closest point for RA apex in isolated region. Using original ID.")
                adjusted_RAA = self.ra_apex  # Fallback

            try:
                detector_RA = RingDetector(ra_region_polydata, ra_ap_point_coord, outdir)
                rings_ra = detector_RA.detect_rings(debug=self.debug)

                # Pass adjusted_RAA, empty initial centroids {}
                b_tag_ra, ra_centroids, rings_ra_obj = detector_RA.mark_ra_rings(adjusted_RAA,
                                                                                 rings_ra,
                                                                                 np.zeros(ra_region_polydata.GetNumberOfPoints()),
                                                                                 {},
                                                                                 debug=self.debug)

                # Perform standard TOP_ENDO identification using the detector method
                detector_RA.cutting_plane_to_identify_tv_f_tv_s(ra_region_polydata,
                                                                rings_ra_obj,
                                                                debug=self.debug)

                # Write the tagged mesh
                ds_ra = dsa.WrapDataObject(ra_region_polydata)
                ds_ra.PointData.append(b_tag_ra, 'boundary_tag')
                write_vtk(os.path.join(outdir, 'RA_boundaries_tagged.vtk'), ds_ra.VTKObject)

                return ra_centroids
            except Exception as e:
                print(f"ERROR during RA ring detection/marking: {e}")
                return {}
        else:
            return {}

    def extract_rings(self, surface_mesh_path: str) -> None:
        """
        Orchestrates the ring extraction process by reading the mesh (via MeshReader),
        processing the LA and/or RA regions, and writing centroids to CSV.
        """
        if self.debug:
            print(f"Initiating standard ring extraction on: {surface_mesh_path}")

        # Prepare output directory
        outdir = self._prepare_output_directory("_surf")

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

        df = pd.DataFrame.from_dict(centroids, orient='index', columns=['X', 'Y', 'Z'])
        df.index.name = 'RingName'
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
        Orchestrates the specialized TOP_EPI/ENDO ring extraction workflow
        by calling the function from the top_epi_endo module. Loads the
        centroids generated by that function.

        Args:
            surface_mesh_path (str): Path to the input surface mesh (e.g., epi).
        """
        # prepare output
        outdir = self._prepare_output_directory("_surf")

        # load surface mesh
        reader = MeshReader(surface_mesh_path)
        mesh_pd = reader.get_polydata()

        if mesh_pd.GetPointData().GetArray("Ids") is None:
            mesh_pd = generate_ids(mesh_pd, "Ids", "Ids")

        centroids = {}

        if self.la_apex is not None:
            la_cents = self._process_LA_region(mesh_pd, outdir, is_biatrial=True)
            centroids.update(la_cents)

        if self.ra_apex is None:
            raise ValueError("RA apex index (ra_apex) is required for TOP_EPI/ENDO")

        # isolate RA region via connectivity + threshold
        ra_ap_pt = mesh_pd.GetPoint(self.ra_apex)
        conn = init_connectivity_filter(mesh_pd, ExtractionModes.ALL_REGIONS, True).GetOutput()
        arr = conn.GetPointData().GetArray("RegionId")
        arr.SetName("RegionID")
        tags = vtk_to_numpy(arr)
        idx = find_closest_point(conn, ra_ap_pt)
        tag_value = int(tags[idx])
        thr = get_threshold_between(conn, tag_value, tag_value,"vtkDataObject::FIELD_ASSOCIATION_POINTS", "RegionID")

        ra_poly = apply_vtk_geom_filter(thr.GetOutputPort())

        ra_region = generate_ids(ra_poly, "Ids", "Ids")

        # write RA.vtp for downstream
        write_vtk(os.path.join(outdir, "RA.vtp"), ra_region)

        # 3) detect & mark RA rings
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

        ds_ra = dsa.WrapDataObject(ra_region)
        ds_ra.PointData.append(b_tag, "boundary_tag")
        write_vtk(os.path.join(outdir, "RA_boundaries_tagged.vtk"), ds_ra.VTKObject)

        endo_path = f"{self._get_base_mesh()}_endo.obj"
        if not os.path.exists(endo_path):
            raise FileNotFoundError(f"Expected endo mesh at {endo_path}; run separate_epi_endo first")

        detector.perform_tv_split_and_find_top_epi_endo(model_epi=ra_region,
                                                        endo_mesh_path=endo_path,
                                                        rings=rings,
                                                        debug=self.debug)

        df = pd.DataFrame.from_dict(centroids, orient="index", columns=["X", "Y", "Z"])
        df.index.name = "RingName"
        write_csv(os.path.join(outdir, "rings_centroids.csv"), df)

        self.ring_info = centroids
        if self.debug:
            print("TOP_EPI/ENDO centroids:", self.ring_info)