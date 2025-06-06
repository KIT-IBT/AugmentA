import os
import subprocess
import shutil
from glob import glob
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

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

        self.polydata = None  # Initial mesh polydata

        if os.path.exists(self.mesh_path):
            try:
                reader = MeshReader(self.mesh_path)
                self.polydata = reader.get_polydata()

                if self.debug:
                    if self.polydata is None or self.polydata.GetNumberOfPoints() == 0:
                        print(f"Warning: Initial mesh loaded from {self.mesh_path} is empty or invalid.")

            except Exception as e:
                if self.debug:
                    print(
                        f"Warning: Could not load initial mesh {self.mesh_path} in AtrialBoundaryGenerator constructor: {e}")
        else:
            if self.debug:
                print(
                    f"Warning: Initial mesh_path {self.mesh_path} provided to AtrialBoundaryGenerator does not exist.")

        self.active_surface_for_rings: vtk.vtkPolyData = None

        self.epi_surface_polydata: vtk.vtkPolyData = None
        self.endo_surface_polydata: vtk.vtkPolyData = None
        self.combined_wall_polydata: vtk.vtkPolyData = None
        self.epi_surface_path: str = None
        self.endo_surface_path: str = None
        self.combined_wall_path: str = None

        self.la_isolated_region_polydata: vtk.vtkPolyData = None
        self.la_ring_detector: RingDetector = None
        self.la_tagged_boundaries_polydata: vtk.vtkPolyData = None

        self.ra_isolated_region_polydata: vtk.vtkPolyData = None
        self.ra_ring_detector: RingDetector = None
        self.ra_tagged_boundaries_polydata: vtk.vtkPolyData = None

        self.active_ra_epi_for_top_rings: vtk.vtkPolyData = None
        self.ra_isolated_region_for_top_epi_endo: vtk.vtkPolyData = None
        self.ra_ring_detector_top_epi_endo: RingDetector = None

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

    def _process_LA_region(
            self,
            input_mesh_polydata: vtk.vtkPolyData,
            outdir: str,
            is_biatrial: bool
    ) -> dict:
        """
        Processes the LA region and returns a centroid dictionary that now
        also includes the LAA apex (“LAA”) and the LAA base (“LAA_base”)
        coordinates for the final CSV.
        """
        if self.la_apex is None:
            print("DEBUG: _process_LA_region skipped as self.la_apex is None.")
            return {}

        final_centroids = {}
        laa_coord_on_input = None  # Store the LAA coordinate for reuse

        # Validate and fetch the LAA apex coordinate
        try:
            num_points = input_mesh_polydata.GetNumberOfPoints()
            if not (0 <= self.la_apex and self.la_apex < num_points):
                raise IndexError(
                    f"LAA ID {self.la_apex} is out of bounds for current input mesh "
                    f"(size: {num_points})."
                )

            laa_coord_on_input = input_mesh_polydata.GetPoint(self.la_apex)
            final_centroids["LAA"] = laa_coord_on_input

        except IndexError as e:
            print(
                f"ERROR_ABG_LA: Invalid LAA ID {self.la_apex} for current input mesh. "
                f"Cannot proceed. {e}"
            )
            return {}

        # If a base ID for the LAA was provided, validate and fetch its coordinate
        if self.la_base is not None:
            try:
                num_points = input_mesh_polydata.GetNumberOfPoints()
                if not (0 <= self.la_base and self.la_base < num_points):
                    raise IndexError(f"LAA_base ID {self.la_base} is out of bounds for current input mesh.")

                final_centroids["LAA_base"] = input_mesh_polydata.GetPoint(self.la_base)

            except IndexError as e:
                if self.debug:
                    print(
                        f"WARN_LA: LAA_base ID {self.la_base} invalid for current input. "
                        f"Not added to centroids. {e}"
                    )

        la_region_polydata: vtk.vtkPolyData = None
        # Coordinate used for isolation if biatrial

        if is_biatrial:
            # For biatrial meshes, isolate LA region from combined RA/LA mesh
            try:
                # Get the actual LA apex coordinate from the full mesh
                la_ap_point_coord = input_mesh_polydata.GetPoint(self.la_apex)

                # Run a connectivity filter over all regions, then fetch the output
                mesh_conn = init_connectivity_filter(input_mesh_polydata, ExtractionModes.ALL_REGIONS, True).GetOutput()

                # Ensure connectivity filter output is valid
                if mesh_conn is None or mesh_conn.GetNumberOfPoints() == 0:
                    print(f"ERROR: Connectivity filter produced no output or empty mesh for processing.")
                    raise RuntimeError(f"Connectivity filter failed for")

                # Fetch the RegionId array to determine which points belong to LA
                arr = mesh_conn.GetPointData().GetArray("RegionId")
                if arr is None:
                    msg = f"LA_RA: 'RegionId' array not found on mesh_conn for input {input_mesh_polydata}"
                    print(f"ERROR: {msg}")
                    raise RuntimeError(msg)

                arr.SetName("RegionID")
                id_vector = vtk_to_numpy(arr)

                # Find the closest point index in mesh_conn to the LA apex coordinate
                temp_LAA_id = find_closest_point(mesh_conn, la_ap_point_coord)
                if temp_LAA_id < 0:
                    msg = (f"LA: find_closest_point failed to find LA apex on mesh_conn (returned {temp_LAA_id}). "
                           f"Input mesh for connectivity: {self.mesh_path}, Processed: path was "
                           f"{input_mesh_polydata.GetObjectName() if hasattr(input_mesh_polydata, 'GetObjectName') else 'N/A'}")
                    print(f"ERROR: {msg}")
                    raise RuntimeError(msg)

                # Ensure the found index is within the bounds of the RegionID array
                if temp_LAA_id >= len(id_vector):  # Check if the ID is too large for the region ID array
                    msg = (f"LA: Point index {temp_LAA_id} (from find_closest_point on mesh_conn) "
                           f"is out of bounds for RegionID array (id_vector) of size {len(id_vector)}. "
                           f"Mesh_conn points: {mesh_conn.GetNumberOfPoints()}.")
                    print(f"ERROR: {msg}")
                    raise IndexError(msg)

                # Determine which region tag corresponds to LA
                LA_tag_val = id_vector[temp_LAA_id]

                # Threshold the mesh to isolate only points with the LA tag value
                la_thresh = get_threshold_between(
                    mesh_conn,
                    LA_tag_val,
                    LA_tag_val,
                    "vtkDataObject::FIELD_ASSOCIATION_POINTS",
                    "RegionID"
                )
                la_poly_ug = la_thresh.GetOutput()
                la_poly = apply_vtk_geom_filter(la_poly_ug)

                # If the isolated LA polydata is valid and non-empty, regenerate 'Ids' array
                if la_poly and la_poly.GetNumberOfPoints() > 0:
                    if la_poly.GetPointData().GetArray("Ids"):
                        if self.debug:
                            print("DEBUG_LA: Removing existing 'Ids' from isolated LA part before local generation.")

                        la_poly.GetPointData().RemoveArray("Ids")

                    la_region_polydata = generate_ids(la_poly, "Ids", "Ids")

                else:
                    print("Warning: LA region extraction resulted in empty mesh.")
                    return {}

            except Exception as e:
                print(f"ERROR during LA region extraction: {e}")
                return {}

        else:
            # Non-biatrial case: use the input mesh directly for LA processing
            la_region_polydata = input_mesh_polydata
            try:
                # Validate and fetch the LA apex coordinate from the LA-only mesh
                la_ap_point_coord = la_region_polydata.GetPoint(self.la_apex)

            except IndexError:
                print(f"ERROR: LA apex ID {self.la_apex} out of bounds for LA-only mesh.")
                return {}

            # If an 'Ids' array already exists, remove it before regenerating
            if la_region_polydata.GetPointData().GetArray("Ids"):
                if self.debug:
                    print(f"DEBUG_LA: LA-only: Removing existing 'Ids' before local generation for consistency.")
                la_region_polydata.GetPointData().RemoveArray("Ids")  # Ensure it's fresh if we regenerate

            la_region_polydata = generate_ids(la_region_polydata, "Ids", "Ids")

        self.la_isolated_region_polydata = la_region_polydata

        # === Ring detection on the LA region ===
        if self.la_isolated_region_polydata and la_ap_point_coord:
            # Write the LA-only mesh to disk
            write_vtk(os.path.join(outdir, "LA.vtk"), self.la_isolated_region_polydata)

            # Find the index of the closest point to the LA apex on the LA-only mesh
            adjusted_LAA = find_closest_point(self.la_isolated_region_polydata, la_ap_point_coord)
            if adjusted_LAA < 0:
                adjusted_LAA = self.la_apex

            apex_coord_for_detector = self.la_isolated_region_polydata.GetPoint(adjusted_LAA)

            try:
                self.la_ring_detector = RingDetector(
                    self.la_isolated_region_polydata,
                    apex_coord_for_detector,
                    outdir
                )

                rings_la = self.la_ring_detector.detect_rings(debug=self.debug)

                b_tag_la, la_centroids = self.la_ring_detector.mark_la_rings(
                    adjusted_LAA,
                    rings_la,
                    np.zeros(la_region_polydata.GetNumberOfPoints()),
                    {},
                    self.la_isolated_region_polydata,
                    debug=self.debug
                )

                # Attach boundary tags to point data and write to disk
                ds_la = dsa.WrapDataObject(self.la_isolated_region_polydata)
                ds_la.PointData.append(b_tag_la, "boundary_tag")

                self.la_tagged_boundaries_polydata = ds_la.VTKObject
                write_vtk(os.path.join(outdir, "LA_boundaries_tagged.vtk"), self.la_tagged_boundaries_polydata)

                # Copy centroids and ensure LAA coordinate is included
                final_la_centroids = la_centroids.copy()
                if 'la_ap_point_coord' in locals() and la_ap_point_coord is not None:
                    final_la_centroids["LAA"] = la_ap_point_coord

                else:
                    try:
                        if self.la_apex is not None:
                            num_points = input_mesh_polydata.GetNumberOfPoints()

                            if 0 <= self.la_apex and self.la_apex < num_points:
                                final_la_centroids["LAA"] = input_mesh_polydata.GetPoint(self.la_apex)

                            if self.debug:
                                print(f"DEBUG_LA: Fallback used for 'LAA' coordinate in CSV output.")
                        else:
                            if self.debug:
                                print(f"WARN_LA: 'LAA' coordinate could not be determined for CSV (self.la_apex: {self.la_apex}).")

                    except IndexError:
                        if self.debug:
                            print(f"WARN_LA: Error fetching 'LAA' coordinate for CSV via fallback (self.la_apex: {self.la_apex}).")

                # If a base ID was provided, attempt to include its coordinate
                if self.la_base is not None:
                    try:
                        num_points = input_mesh_polydata.GetNumberOfPoints()

                        if 0 <= self.la_base and self.la_base < num_points:
                            final_la_centroids["LAA_base"] = input_mesh_polydata.GetPoint(self.la_base)

                        else:
                            if self.debug:
                                print(f"WARN_ABG_LA: LAA_base ID {self.la_base} is out of bounds for current mesh; "
                                      f"'LAA_base' not added to centroids for CSV.")

                    except IndexError:  # Should be caught by the bounds check, but as a safeguard
                        if self.debug:
                            print(
                                f"WARN_LA: Error fetching 'LAA_base' coordinate for CSV (self.la_base: {self.la_base}).")

                return final_la_centroids

            except Exception as e:
                print(f"ERROR during LA ring detection/marking: {e}")
                return {}

        return {}

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

        # Prepare apex/base coordinates from cached polydata (if present)
        ra_ap_point_coord_for_csv = None
        ra_bs_point_coord_for_csv = None

        if hasattr(self, "polydata") and self.polydata is not None:
            num_pts = self.polydata.GetNumberOfPoints()

            # Fetch RAA apex coordinate if within bounds
            if 0 <= self.ra_apex and self.ra_apex < num_pts:
                ra_ap_point_coord_for_csv = self.polydata.GetPoint(self.ra_apex)

            # Fetch RAA base coordinate if within bounds
            if 0 <= self.ra_base and self.ra_base < num_pts:
                ra_bs_point_coord_for_csv = self.polydata.GetPoint(self.ra_base)

        # Isolate the RA region (biatrial vs. RA-only)
        if is_biatrial:
            try:
                # Get the RA apex coordinate from the combined mesh
                ra_ap_point_coord = input_mesh_polydata.GetPoint(self.ra_apex)

                # Apply connectivity filter over all regions
                mesh_conn = init_connectivity_filter(input_mesh_polydata, ExtractionModes.ALL_REGIONS, True).GetOutput()

                # Retrieve the RegionId array and convert to numpy
                arr = mesh_conn.GetPointData().GetArray("RegionId")
                arr.SetName("RegionID")
                id_vector = vtk_to_numpy(arr)

                # Find the index closest to the RA apex
                temp_RAA_id = find_closest_point(mesh_conn, ra_ap_point_coord)
                RA_tag_val = id_vector[temp_RAA_id]

                # Threshold the mesh to isolate RA region by tag
                ra_thresh = get_threshold_between(
                    mesh_conn,
                    RA_tag_val,
                    RA_tag_val,
                    "vtkDataObject::FIELD_ASSOCIATION_POINTS",
                    "RegionID"
                )
                ra_poly_ug = ra_thresh.GetOutput()
                ra_poly = apply_vtk_geom_filter(ra_poly_ug)

                # If RA submesh is non-empty, regenerate its 'Ids' array
                if ra_poly and ra_poly.GetNumberOfPoints() > 0:
                    # Explicitly remove any pre-existing "Ids" array from ra_poly
                    if ra_poly.GetPointData().GetArray("Ids"):
                        if self.debug:
                            print("DEBUG_RA: Removing existing 'Ids' from isolated RA part before local generation.")
                        ra_poly.GetPointData().RemoveArray("Ids")

                    ra_region_polydata = generate_ids(ra_poly, "Ids", "Ids")

                else:
                    print("Warning: RA region extraction resulted in empty mesh.")
                    return {}

            except Exception as e:
                print(f"ERROR during RA region extraction: {e}")
                return {}

        else:
            # RA-only case: use the input mesh directly
            ra_region_polydata = input_mesh_polydata

            try:
                # Validate and fetch RA apex coordinate
                ra_ap_point_coord = ra_region_polydata.GetPoint(self.ra_apex)
            except IndexError:
                print(f"ERROR: RA apex ID {self.ra_apex} out of bounds for RA-only mesh.")
                return {}

            # Remove any existing 'Ids' array before regenerating
            if ra_region_polydata.GetPointData().GetArray("Ids"):
                if self.debug:
                    print("DEBUG_RA: RA-only: Removing existing 'Ids' before local generation for consistency.")
                ra_region_polydata.GetPointData().RemoveArray("Ids")

            ra_region_polydata = generate_ids(ra_region_polydata, "Ids", "Ids")

        # Caching isolated RA mesh for future reference
        self.ra_isolated_region_polydata = ra_region_polydata

        # Ring detection on the isolated RA region
        if self.ra_isolated_region_polydata and ra_ap_point_coord:
            # Write the isolated RA mesh to disk
            write_vtk(os.path.join(outdir, "RA.vtk"), self.ra_isolated_region_polydata)

            # Find the index closest to the RA apex on the isolated mesh
            adjusted_RAA = find_closest_point(self.ra_isolated_region_polydata, ra_ap_point_coord)
            if adjusted_RAA < 0:
                adjusted_RAA = self.ra_apex

            apex_coord_for_detector = self.ra_isolated_region_polydata.GetPoint(adjusted_RAA)

            try:
                detector_RA = RingDetector(
                    self.ra_isolated_region_polydata,
                    apex_coord_for_detector,
                    outdir
                )

                rings_ra = detector_RA.detect_rings(debug=self.debug)

                b_tag_ra, ra_centroids, rings_ra_obj = detector_RA.mark_ra_rings(
                    adjusted_RAA,
                    rings_ra,
                    np.zeros(self.ra_isolated_region_polydata.GetNumberOfPoints()),
                    {},
                    debug=self.debug
                )

                detector_RA.cutting_plane_to_identify_tv_f_tv_s(
                    self.ra_isolated_region_polydata,
                    rings_ra_obj,
                    debug=self.debug
                )

                # Attach boundary tags and write tagged mesh to disk
                ds_ra = dsa.WrapDataObject(self.ra_isolated_region_polydata)
                ds_ra.PointData.append(b_tag_ra, "boundary_tag")
                self.ra_tagged_boundaries_polydata = ds_ra.VTKObject
                write_vtk(os.path.join(outdir, "RA_boundaries_tagged.vtk"), self.ra_tagged_boundaries_polydata)

                # Prepare final centroids dictionary
                final_ra_centroids = ra_centroids.copy()

                # Add "RAA" coordinate from the isolated mesh (fallback to original mesh)
                if 'ra_ap_point_coord' in locals() and ra_ap_point_coord is not None:
                    final_ra_centroids["RAA"] = ra_ap_point_coord
                else:
                    try:
                        num_pts_input = input_mesh_polydata.GetNumberOfPoints()

                        if 0 <= self.ra_apex and self.ra_apex < num_pts_input:
                            final_ra_centroids["RAA"] = input_mesh_polydata.GetPoint(self.ra_apex)
                            if self.debug:
                                print(f"DEBUG_RA: Fallback used for 'RAA' coordinate in CSV output.")
                        else:
                            if self.debug:
                                print(f"WARN_RA: 'RAA' coordinate could not be determined for CSV (self.ra_apex: {self.ra_apex}).")
                    except IndexError:
                        if self.debug:
                            print(f"WARN_RA: Error fetching 'RAA' coordinate for CSV via fallback (self.ra_apex: {self.ra_apex}).")

                # Add "RAA_base" coordinate if valid in the original mesh
                if self.ra_base is not None:
                    try:
                        num_pts_input = input_mesh_polydata.GetNumberOfPoints()
                        if 0 <= self.ra_base and self.ra_base < num_pts_input:
                            final_ra_centroids["RAA_base"] = input_mesh_polydata.GetPoint(self.ra_base)
                        else:
                            if self.debug:
                                print(f"WARN_ABG_RA: RAA_base ID {self.ra_base} is out of bounds for current mesh "
                                      f"'RAA_base' not added to centroids for CSV.")
                    except IndexError:
                        if self.debug:
                            print(f"WARN_ABG_RA: Error fetching 'RAA_base' coordinate for CSV (self.ra_base: {self.ra_base}).")

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

        surface_base = os.path.splitext(surface_mesh_path)[0]
        outdir = f"{surface_base}_surf"
        os.makedirs(outdir, exist_ok=True)

        # Remove any existing 'ids_*' files in the output directory
        for f_path in glob(os.path.join(outdir, 'ids_*')):
            if os.path.isfile(f_path):
                os.remove(f_path)

        # Read the surface mesh file and ensure it has an "Ids" array on its points
        try:
            mesh_obj = MeshReader(surface_mesh_path)
            self.active_surface_for_rings = mesh_obj.get_polydata()

            # If the mesh has no point data or no "Ids" array, generate one
            pt_data = self.active_surface_for_rings.GetPointData()
            if not pt_data or not pt_data.GetArray("Ids"):
                self.active_surface_for_rings = generate_ids(self.active_surface_for_rings,
                                                             "Ids",
                                                             "Ids")

        except Exception as e:
            print(f"ERROR reading surface mesh {surface_mesh_path}: {e}")
            raise

        # Collect centroids from LA and RA regions (biatrial or single chamber)
        centroids: Dict[Any, Any] = {}

        if self.la_apex is not None and self.ra_apex is not None:
            # Biatrial case: process both LA and RA from the same mesh
            la_centroids = self._process_LA_region(
                self.active_surface_for_rings,
                outdir,
                is_biatrial=True
            )
            centroids.update(la_centroids)

            ra_centroids = self._process_RA_region(
                self.active_surface_for_rings,
                outdir,
                is_biatrial=True
            )
            centroids.update(ra_centroids)

        elif self.la_apex is not None:
            # LA-only case: process LA region
            la_centroids = self._process_LA_region(
                self.active_surface_for_rings,
                outdir,
                is_biatrial=False
            )
            centroids.update(la_centroids)

        elif self.ra_apex is not None:
            # RA-only case: process RA region
            ra_centroids = self._process_RA_region(
                self.active_surface_for_rings,
                outdir,
                is_biatrial=False
            )
            centroids.update(ra_centroids)

        else:
            # Neither apex provided: cannot proceed
            raise ValueError("No apex ID provided for extract_rings.")

        # Convert each centroid’s values into a list for DataFrame construction
        processed_centroids_for_df: Dict[Any, List[Any]] = {}
        for key, value in centroids.items():
            processed_centroids_for_df[key] = list(value)

        # Build a DataFrame and write it to CSV
        df = pd.DataFrame(processed_centroids_for_df)
        csv_path = os.path.join(outdir, "rings_centroids.csv")
        write_csv(csv_path, df)

        # Store the raw centroids for later use
        self.ring_info = centroids

    def separate_epi_endo(self, tagged_volume_mesh_path: str, atrium: str) -> None:
        """
        Delegates epi/endo separation using the epi_endo_separator module.
        Requires element tags loaded via load_element_tags() beforehand.

        Args:
            tagged_volume_mesh_path (str): Path to the tagged volume mesh.
            atrium (str): 'LA' or 'RA'.
        """
        # Ensure element tags are loaded before proceeding
        if not self.element_tags:
            raise RuntimeError(
                "Element tags missing in AtrialBoundaryGenerator. "
                "Call load_element_tags() before separate_epi_endo()."
            )

        try:
            separate_epi_endo(
                mesh_path=tagged_volume_mesh_path,
                atrium=atrium,
                element_tags=self.element_tags
            )

            # Extract base name (without extension) from the input path
            base_vol_path_no_ext, _ = os.path.splitext(tagged_volume_mesh_path)
            original_base_name = base_vol_path_no_ext

            # If the filename ends with "_vol", strip that suffix
            if base_vol_path_no_ext.endswith("_vol"):
                original_base_name = base_vol_path_no_ext[:-4]

            # Construct expected output file paths
            self.epi_surface_path = f"{original_base_name}_{atrium}_epi.vtk"
            self.endo_surface_path = f"{original_base_name}_{atrium}_endo.vtk"
            self.combined_wall_path = f"{original_base_name}_{atrium}.vtk"

            # Load the epicardial surface if it exists
            if os.path.exists(self.epi_surface_path):
                self.epi_surface_polydata = MeshReader(self.epi_surface_path).get_polydata()
            else:
                self.epi_surface_polydata = None
                if self.debug:
                    print(f"Warning: Epi surface file not found after separation: {self.epi_surface_path}")

            # Load the endocardial surface if it exists
            if os.path.exists(self.endo_surface_path):
                self.endo_surface_polydata = MeshReader(self.endo_surface_path).get_polydata()
            else:
                self.endo_surface_polydata = None
                if self.debug: print(f"Warning: Endo surface file not found after separation: {self.endo_surface_path}")

            # Load the combined wall surface if it exists
            if os.path.exists(self.combined_wall_path):
                self.combined_wall_polydata = MeshReader(self.combined_wall_path).get_polydata()
            else:
                self.combined_wall_polydata = None
                if self.debug: print(
                    f"Warning: Combined wall file not found after separation: {self.combined_wall_path}")

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

        for f_path in glob(os.path.join(outdir, 'ids_*')):
            if os.path.isfile(f_path):
                os.remove(f_path)

        reader = MeshReader(surface_mesh_path)

        self.active_ra_epi_for_top_rings = reader.get_polydata()

        if not self.active_ra_epi_for_top_rings or self.active_ra_epi_for_top_rings.GetNumberOfPoints() == 0:
            print(f"ERROR_TOP: Mesh loaded from '{surface_mesh_path}' is empty or invalid. Aborting TOP_EPI_ENDO.")
            return

        if self.active_ra_epi_for_top_rings.GetPointData().GetArray("Ids") is None:
            if self.debug:
                print(f"DEBUG_TOP: Generating 'Ids' for mesh_pd from '{surface_mesh_path}'.")
            self.active_ra_epi_for_top_rings = generate_ids(self.active_ra_epi_for_top_rings, "Ids", "Ids")

        centroids: Dict[str, Any] = {}  # TODO: Check and change to Dict[str, Tuple[float,float,float]]

        if self.la_apex is not None:
            if self.debug:
                print(f"DEBUG_TOP: LAA ID (self.la_apex) is set. Processing LA region from '{surface_mesh_path}'.")
            la_cents = self._process_LA_region(self.active_ra_epi_for_top_rings, outdir, is_biatrial=True)
            centroids.update(la_cents)

        if self.ra_apex is None:
            # If only LA was processed, write its centroids (if any) and raise error.
            raise ValueError("RA apex index (ra_apex) is required for TOP_EPI/ENDO")

        # Isolate RA region
        try:
            if not self.ra_apex >= 0:
                if not self.active_ra_epi_for_top_rings.GetNumberOfPoints():
                    raise IndexError(f"RAA ID {self.ra_apex} is out of bounds for surface_mesh_path '{surface_mesh_path}'.")

            ra_ap_pt_on_mesh_pd = self.active_ra_epi_for_top_rings.GetPoint(self.ra_apex)

        except IndexError as e:
            print(f"ERROR_TOP: Invalid RAA ID {self.ra_apex} for surface_mesh_path '{surface_mesh_path}'. {e}")
            return

        conn = init_connectivity_filter(self.active_ra_epi_for_top_rings, ExtractionModes.ALL_REGIONS, True).GetOutput()
        if not conn or conn.GetNumberOfPoints() == 0:
            print(f"ERROR_TOP: Connectivity filter on '{surface_mesh_path}' (for RA) yielded empty result.")

        arr = conn.GetPointData().GetArray("RegionId")
        if not arr:
            print(f"ERROR_TOP: 'RegionId' not found after connectivity on '{surface_mesh_path}' for RA.")

        arr.SetName("RegionID")
        tags = vtk_to_numpy(arr)
        temp_raa_id_on_conn = find_closest_point(conn, ra_ap_pt_on_mesh_pd)

        tag_val = int(tags[temp_raa_id_on_conn])

        thr = get_threshold_between(
            conn,
            tag_val,
            tag_val,
            "vtkDataObject::FIELD_ASSOCIATION_POINTS",
            "RegionID"
        )

        ra_region_ug = thr.GetOutput()
        ra_poly_geom_filtered = apply_vtk_geom_filter(ra_region_ug)

        if not ra_poly_geom_filtered or ra_poly_geom_filtered.GetNumberOfPoints() == 0:
            print(f"ERROR_TOP: RA region from thresholding (in TOP_EPI_ENDO) is empty for '{surface_mesh_path}'.")
            return

        if ra_poly_geom_filtered.GetPointData().GetArray("Ids"):
            if self.debug:
                print(
                    f"DEBUG_TOP: Removing existing 'Ids' from isolated RA part (TOP_EPI_ENDO) before local generation.")
            ra_poly_geom_filtered.GetPointData().RemoveArray("Ids")

        self.ra_isolated_region_for_top_epi_endo = generate_ids(ra_poly_geom_filtered, "Ids", "Ids")

        if not self.ra_isolated_region_for_top_epi_endo or self.ra_isolated_region_for_top_epi_endo.GetNumberOfPoints() == 0:
            print(f"ERROR_TOP: RA region became empty after generate_ids for '{surface_mesh_path}'.")
            return

        write_vtk(os.path.join(outdir, "RA.vtp"), self.ra_isolated_region_for_top_epi_endo, xml_format=True)

        adj_id_on_ra_region = find_closest_point(self.ra_isolated_region_for_top_epi_endo, ra_ap_pt_on_mesh_pd)
        apex_coord_for_detector_on_ra_region = None
        if adj_id_on_ra_region < 0:
            print(
                f"WARNING_TOP: RAA point (from mesh_pd coord: {ra_ap_pt_on_mesh_pd}) not found on isolated ra_region. "
                "Using coordinate from mesh_pd directly for RingDetector, but this may be suboptimal if not on the "
                "surface.")
            apex_coord_for_detector_on_ra_region = ra_ap_pt_on_mesh_pd
        else:
            apex_coord_for_detector_on_ra_region = self.ra_isolated_region_for_top_epi_endo.GetPoint(adj_id_on_ra_region)

        detector = RingDetector(self.ra_isolated_region_for_top_epi_endo, apex_coord_for_detector_on_ra_region, outdir)
        detected_ra_ring_objects = detector.detect_rings(debug=self.debug)

        if adj_id_on_ra_region < 0 and self.debug:
            print(
                f'WARNING_TOP: RAA ID for mark_ra_rings is invalid ({adj_id_on_ra_region}). File ids_RAA.vtx may be affected.')

        b_tag_numpy, ra_ring_centroids, updated_ra_ring_objects_list = detector.mark_ra_rings(
            adjusted_RAA_id=adj_id_on_ra_region,
            rings=detected_ra_ring_objects,
            b_tag=np.zeros(self.ra_isolated_region_for_top_epi_endo.GetNumberOfPoints(), dtype=int),
            centroids={},
            debug=self.debug
        )

        centroids.update(ra_ring_centroids)

        if self.ra_apex is not None:
            try:
                if 0 <= self.ra_apex < self.active_ra_epi_for_top_rings.GetNumberOfPoints():
                    centroids["RAA"] = ra_ap_pt_on_mesh_pd
                    if self.debug:
                        print(
                            f"DEBUG_TOP: Added RAA coordinate {ra_ap_pt_on_mesh_pd} (ID {self.ra_apex} on input '{surface_mesh_path}') to local centroids.")
            except Exception as e_raa_csv:
                if self.debug:
                    print(
                        f"WARN_TOP: Error ensuring 'RAA' for local centroids (ID: {self.ra_apex}) from '{surface_mesh_path}'. {e_raa_csv}")
        elif self.debug:
            print(f"DEBUG_TOP: self.ra_apex is None (re-check). 'RAA' coordinate not added.")

        if self.ra_base is not None:
            try:
                if 0 <= self.ra_base and self.ra_base < self.active_ra_epi_for_top_rings.GetNumberOfPoints():
                    raa_base_coord_on_surface_mesh = self.active_ra_epi_for_top_rings.GetPoint(self.ra_base)
                    centroids["RAA_base"] = raa_base_coord_on_surface_mesh

                    if self.debug:
                        print(
                            f"DEBUG_TOP: Added RAA_base coordinate {raa_base_coord_on_surface_mesh} (ID {self.ra_base} on input '{surface_mesh_path}') to local centroids.")

                elif self.debug:
                    print(
                        f"WARN_TOP: RAA_base ID {self.ra_base} out of bounds for '{surface_mesh_path}'; 'RAA_base' not added.")

            except Exception as e_raabase_csv:
                if self.debug:
                    print(
                        f"WARN_TOP: Error fetching 'RAA_base' for local centroids (ID: {self.ra_base}) from '{surface_mesh_path}'. {e_raabase_csv}")

        elif self.debug:
            print(f"DEBUG_TOP: self.ra_base is None. 'RAA_base' coordinate not added.")

        ds_ra = dsa.WrapDataObject(self.ra_isolated_region_for_top_epi_endo)
        ds_ra.PointData.append(b_tag_numpy, "boundary_tag")
        write_vtk(os.path.join(outdir, "RA_boundaries_tagged.vtp"), ds_ra.VTKObject, xml_format=True)

        original_input_base = self._get_base_mesh()
        # This method is contextually for RA in the pipeline for TOP_EPI_ENDO.
        endo_atrium_specifier = "RA"
        endo_path = f"{original_input_base}_{endo_atrium_specifier}_endo.obj"

        if not os.path.exists(endo_path):
            # Fallback: Try to derive from surface_mesh_path (e.g., if it's "..._RA_epi.obj")
            current_surface_base_name = os.path.splitext(surface_mesh_path)[0]

            # Check if the surface_mesh_path explicitly mentions the atrium (e.g., "_RA_epi")
            if current_surface_base_name.endswith(f"_{endo_atrium_specifier}_epi"):
                potential_endo_path = current_surface_base_name[:-len("_epi")] + "_endo.obj"
                if os.path.exists(potential_endo_path):
                    endo_path = potential_endo_path
                    if self.debug:
                        print(f"DEBUG_ABG_TOP: Used fallback for endo_path construction: {endo_path}")

                else:
                    raise FileNotFoundError(
                        f"Endocardial mesh not found. Primary: '{endo_path}'. Fallback: '{potential_endo_path}'. "
                        f"Ensure 'separate_epi_endo' created the correct file for atrium '{endo_atrium_specifier}'.")

            else:
                raise FileNotFoundError(
                    f"Endocardial mesh not found at primary path: '{endo_path}'. Cannot derive from '{surface_mesh_path}'. "
                    f"Ensure 'separate_epi_endo' created '{original_input_base}_{endo_atrium_specifier}_endo.obj'.")

            if self.debug:
                print(f"DEBUG_TOP: Using endocardial mesh path: {endo_path} for TOP_EPI/ENDO cuts.")

        detector.perform_tv_split_and_find_top_epi_endo(
            model_epi=self.ra_isolated_region_for_top_epi_endo,
            endo_mesh_path=endo_path,
            rings=updated_ra_ring_objects_list,
            debug=self.debug
        )

        df = pd.DataFrame.from_dict(centroids, orient="index", columns=["X", "Y", "Z"])
        df.index.name = "RingName"
        write_csv(os.path.join(outdir, "rings_centroids.csv"), df)

        self.ring_info = centroids
        if self.debug:
            print(
                f"DEBUG_TOP: TOP_EPI/ENDO processing complete. Final centroids for '{surface_base}': {self.ring_info}")
