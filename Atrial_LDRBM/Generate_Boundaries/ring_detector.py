import os

import numpy as np
import vtk
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from vtk.numpy_interface import dataset_adapter as dsa

from Atrial_LDRBM.Generate_Boundaries.file_manager import write_vtk, write_vtx_file
from Atrial_LDRBM.Generate_Boundaries.mesh import MeshReader
from vtk_opencarp_helper_methods.mathematical_operations.vector_operations import get_normalized_cross_product
from vtk_opencarp_helper_methods.vtk_methods.converters import vtk_to_numpy, numpy_to_vtk
from vtk_opencarp_helper_methods.vtk_methods.filters import apply_vtk_geom_filter, clean_polydata, generate_ids, \
    get_center_of_mass, get_feature_edges, get_elements_above_plane
from vtk_opencarp_helper_methods.vtk_methods.finder import find_closest_point
from vtk_opencarp_helper_methods.vtk_methods.init_objects import init_connectivity_filter, ExtractionModes, \
    initialize_plane_with_points, initialize_plane
from vtk_opencarp_helper_methods.vtk_methods.thresholding import get_lower_threshold


class Ring:
    """
    Represents a detected anatomical ring from the atrial surface.
    Each ring is defined by its connectivity region, center of mass, and its point count.
    """

    def __init__(self, index, name, points_num, center_point, distance, polydata):
        # The unique identifier for the connectivity region (i.e., the ring)
        self.id: int = index

        self.name: str = name
        # Total number of points in this ring's geometry
        self.np: int = points_num

        # Center of mass (x, y, z) calculated from the ring's geometry
        self.center: tuple = center_point

        # Euclidean distance from the given apex to the ring's center
        self.ap_dist: float = distance

        # The VTK PolyData object that represents the ring's geometry
        self.vtk_polydata: vtk.vtkPolyData = polydata


class RingDetector:
    """
    The RingDetector class encapsulates the complete workflow for detecting,
    classifying, and marking rings on atrial surface meshes.

    It supports both the standard workflow (e.g., identifying TOP_ENDO) and a specialized
    workflow for separate epi/endo surfaces (TOP_EPI/ENDO). All outputs (VTX files and debug files)
    are saved in the provided output directory.
    """

    def __init__(self, surface: vtk.vtkPolyData, apex: tuple, outdir: str):
        # Validate the input surface. If it lacks the required 'Ids' array, try to generate it.
        if not self._validate_vtk_data(surface, check_ids=True):
            print("Warning: Input surface missing 'Ids'. Generating IDs...")
            surface = generate_ids(surface, "Ids", "CellIds")

            if not self._validate_vtk_data(surface, check_ids=True):
                raise ValueError("Failed to get/generate 'Ids' on input surface.")

        self.surface = surface

        # Ensure the provided apex is a valid 3D coordinate.
        if not apex or len(apex) != 3:
            raise ValueError("Invalid apex point (must be tuple/list of 3 numbers).")
        self.apex = apex

        # Check that the output directory's parent exists; if not, throw an error.
        parent_dir = os.path.dirname(outdir) or '.'
        if not os.path.exists(parent_dir):
            raise FileNotFoundError(f"Parent directory does not exist: {parent_dir}")
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)

    def _validate_vtk_data(self, vtk_object, check_points=True, check_cells=False, check_ids=False) -> bool:
        """
        Verifies that a VTK object is valid and optionally checks for the existence
        of points, cells, and the 'Ids' array in its point data.
        """
        if not vtk_object:
            return False

        if check_points and vtk_object.GetNumberOfPoints() == 0:
            return False

        if check_cells:
            cells_are_empty = hasattr(vtk_object, 'GetNumberOfCells') and vtk_object.GetNumberOfCells() == 0
            if cells_are_empty:
                return False

        if check_ids and (not vtk_object.GetPointData() or not vtk_object.GetPointData().GetArray("Ids")):
            return False

        return True

    # Ring Detection Workflow
    def detect_rings(self, debug: bool = False) -> list[Ring]:
        """
        Detects rings from the input surface by extracting the boundary edges andthen finding connected components.
        """
        # Extract the boundary edges of the surface
        boundary_edges = get_feature_edges(self.surface,
                                           boundary_edges_on=True,
                                           feature_edges_on=False,
                                           manifold_edges_on=False,
                                           non_manifold_edges_on=False)

        # Validate the boundary edges output to ensure there are lines/cells.
        if not self._validate_vtk_data(boundary_edges, check_cells=True):
            if debug:
                print("Warning: No boundary edge segments found on the surface.")
            return []

        # Ensure that the boundary edges have the original point IDs
        if not self._validate_vtk_data(boundary_edges, check_ids=True):
            if debug:
                print("Warning: Boundary edges missing 'Ids'. Attempting to generate...")

            boundary_edges = generate_ids(boundary_edges, "Ids", "CellIds")

            if not self._validate_vtk_data(boundary_edges, check_ids=True):
                print("  Error: Failed to obtain 'Ids' on boundary edges; aborting ring detection.")
                return []

        if debug:
            print(f"Found {boundary_edges.GetNumberOfLines()} boundary segments.")
            print("Extracting connected components to form individual rings.")

        # Use connectivity filtering to separate the boundary edges into connected components.
        connect_filter = init_connectivity_filter(boundary_edges, ExtractionModes.ALL_REGIONS)
        num_regions = connect_filter.GetNumberOfExtractedRegions()
        connect_filter.SetExtractionModeToSpecifiedRegions()

        # Process each connected component (potential ring)
        detected_rings = []
        for region_index in range(num_regions):
            ring_obj = self._process_detected_ring_region(connect_filter, region_index, debug)

            if ring_obj:
                detected_rings.append(ring_obj)

        if debug:
            print(f"Detected {len(detected_rings)} rings.")
        return detected_rings

    def _process_detected_ring_region(self, connect_filter, region_index, debug=False) -> Ring | None:
        """
        Processes a single connected component from the connectivity filter, cleaning and converting it to a Ring object.
        """
        connect_filter.AddSpecifiedRegion(region_index)
        connect_filter.Update()
        region_ug = connect_filter.GetOutput()

        # Clean unused points
        region_pd = apply_vtk_geom_filter(region_ug)
        region_pd = clean_polydata(region_pd)

        # Validate that the processed region contains points and has the required 'Ids'
        if not self._validate_vtk_data(region_pd, check_points=True, check_ids=True):
            if debug:
                print(f"Warning: Region {region_index} is empty or missing 'Ids'. Skipping.")

            connect_filter.DeleteSpecifiedRegion(region_index)
            return None

        if debug:
            # Save a debug version of the ring for inspection
            debug_path = os.path.join(self.outdir, f'debug_ring_{region_index}.vtk')
            write_vtk(debug_path, region_pd)

        ring_poly_copy = vtk.vtkPolyData()
        ring_poly_copy.DeepCopy(region_pd)

        # Compute the center of mass and distance from the apex for classification
        center = get_center_of_mass(region_pd, set_use_scalars_as_weights=False)
        distance = np.sqrt(np.sum((np.array(self.apex) - np.array(center)) ** 2))
        num_pts = region_pd.GetNumberOfPoints()

        # Create the Ring object with the computed properties
        ring_obj = Ring(region_index, "", num_pts, center, distance, ring_poly_copy)

        connect_filter.DeleteSpecifiedRegion(region_index)
        connect_filter.Update()
        return ring_obj

    # -------------------- Ring Classification and Marking --------------------
    def _cluster_rings(self, rings: list[Ring], n_clusters: int = 2, debug: bool = False) -> tuple[
        list, list, np.ndarray | None]:
        """
        Clusters non-MV rings using KMeans to separate anatomical groups.
        """
        # Calculating pv indices
        non_mv_indices = [idx for idx, r in enumerate(rings) if r.name != "MV"]

        if not non_mv_indices:
            if debug:
                print("_cluster_rings: No rings available for clustering (all may be MV).")
            return [], [], None

        non_mv_centers = [rings[idx].center for idx in non_mv_indices]
        actual_clusters = min(n_clusters, len(non_mv_centers))

        if actual_clusters < 1:
            return [], [], None

        if actual_clusters == 1:
            if debug:
                print(f"_cluster_rings: Only one ring available; assigning it to group 1.")
            return non_mv_indices, [], np.array([0] * len(non_mv_centers))

        try:
            estimator = KMeans(n_clusters=actual_clusters, n_init='auto', random_state=0)
            labels = estimator.fit_predict(non_mv_centers)

        except Exception as e:
            print(f"Error during clustering: {e}")
            return [], [], None

        # Determine which cluster has the ring closest to the apex
        ap_dists = [rings[idx].ap_dist for idx in non_mv_indices]
        closest_idx = np.argmin(ap_dists)
        label_closest = labels[closest_idx]

        group1 = [non_mv_indices[i] for i, lab in enumerate(labels) if lab == label_closest]
        group2 = [non_mv_indices[i] for i, lab in enumerate(labels) if lab != label_closest]

        if debug:
            print(f"_cluster_rings: Group1 (size {len(group1)}), Group2 (size {len(group2)})")
        return group1, group2, labels

    def _name_pv_subgroup(self, group_indices: list, group_name_prefix: str,
                          rings: list[Ring], apex_dist_is_superior: bool,
                          ref_superior_idx: int | None = None, debug: bool = False):
        """
        Helper method to assign anatomical names (superior/inferior) to a subgroup of PV rings.
        The group_name_prefix is either "L" for left or "R" for right.
        """
        if not group_indices:
            return
        if len(group_indices) == 1:
            rings[group_indices[
                0]].name = f"{group_name_prefix}SPV" if apex_dist_is_superior else f"{group_name_prefix}IPV"
            if debug:
                print(f"Named single {group_name_prefix} ring: {rings[group_indices[0]].name}")
            return

        subgroup_centers = [rings[idx].center for idx in group_indices]

        try:
            estimator = KMeans(n_clusters=2, n_init='auto', random_state=0)
            labels = estimator.fit_predict(subgroup_centers)
        except Exception as e:
            if debug:
                print(f"Warning: Clustering failed for {group_name_prefix} subgroup: {e}")
            return

        superior_label = -1
        if ref_superior_idx is not None and ref_superior_idx in group_indices:
            try:
                local_ref_idx = group_indices.index(ref_superior_idx)
                superior_label = labels[local_ref_idx]
                if debug:
                    print(f"Using reference index {ref_superior_idx} for Superior label {superior_label}.")
            except ValueError:
                if debug:
                    print(f"Warning: Reference index {ref_superior_idx} not found in group. Falling back.")
                ref_superior_idx = None

        if superior_label == -1:
            subgroup_apex_dists = [rings[idx].ap_dist for idx in group_indices]
            local_idx = np.argmin(subgroup_apex_dists) if apex_dist_is_superior else np.argmax(subgroup_apex_dists)
            superior_label = labels[local_idx]
            if debug:
                print(f"Using apex distance to set Superior label {superior_label} for {group_name_prefix}PVs.")

        for local_idx, original_idx in enumerate(group_indices):
            is_superior = (labels[local_idx] == superior_label)
            rings[original_idx].name = f"{group_name_prefix}SPV" if is_superior else f"{group_name_prefix}IPV"
            if debug:
                print(f"Named ring at index {original_idx}: {rings[original_idx].name}")

    def mark_la_rings(self, adjusted_LAA_id: int, rings: list[Ring], b_tag: np.ndarray,
                      centroids: dict, LA_region: vtk.vtkPolyData, debug: bool = False) -> tuple[np.ndarray, dict]:
        """
        Marks the left atrial rings (MV and PVs) by assigning anatomical names, updating
        boundary tags in the b_tag array, and recording centroids.
        """
        if not rings:
            return b_tag, centroids

        # Identify the MV ring by choosing the ring with the most points
        mv_ring_index = np.argmax([r.np for r in rings])
        rings[mv_ring_index].name = "MV"
        if debug:
            print(f"Identified MV: Ring {mv_ring_index} (Size: {rings[mv_ring_index].np})")

        # Cluster non-MV rings into left and right groups.
        LPV_indices, RPV_indices, _ = self._cluster_rings(rings, n_clusters=2, debug=debug)

        RSPV_idx = None
        if LPV_indices and RPV_indices:
            RSPV_idx = self._cutting_plane_to_identify_RSPV(LPV_indices, RPV_indices, rings, debug=debug)

        self._name_pv_subgroup(LPV_indices, "L", rings, apex_dist_is_superior=True, debug=debug)
        self._name_pv_subgroup(RPV_indices, "R", rings, apex_dist_is_superior=False, ref_superior_idx=RSPV_idx,
                               debug=debug)

        LPV_point_ids = []
        RPV_point_ids = []
        final_centroids = centroids.copy()
        tag_map = {"MV": 1, "LIPV": 2, "LSPV": 3, "RIPV": 4, "RSPV": 5}

        # Assign boundary tags and write ring ID files
        for ring_obj in rings:
            if not ring_obj.name or ring_obj.name not in tag_map:
                continue

            if not self._validate_vtk_data(ring_obj.vtk_polydata, check_points=True, check_ids=True):
                if debug:
                    print(f"Skipping ring {ring_obj.id} ('{ring_obj.name}') due to invalid data.")
                continue

            point_ids = vtk_to_numpy(ring_obj.vtk_polydata.GetPointData().GetArray("Ids"))
            if point_ids.size == 0:
                continue

            tag_value = tag_map[ring_obj.name]
            try:
                b_tag[point_ids] = tag_value
            except IndexError:
                print(f"Error: Point IDs for ring {ring_obj.name} out of bounds. Skipping tagging.")
                continue

            write_vtx_file(os.path.join(self.outdir, f'ids_{ring_obj.name}.vtx'), point_ids)
            final_centroids[ring_obj.name] = ring_obj.center
            if ring_obj.name.startswith("L"):
                LPV_point_ids.extend(list(point_ids))
            if ring_obj.name.startswith("R"):
                RPV_point_ids.extend(list(point_ids))

        # Perform additional UAC analysis on the left atrial region.
        self._cutting_plane_to_identify_UAC(LPV_indices, RPV_indices, rings, LA_region, debug=debug)
        write_vtx_file(os.path.join(self.outdir, 'ids_LAA.vtx'), adjusted_LAA_id)
        if LPV_point_ids:
            write_vtx_file(os.path.join(self.outdir, 'ids_LPV.vtx'), LPV_point_ids)
        if RPV_point_ids:
            write_vtx_file(os.path.join(self.outdir, 'ids_RPV.vtx'), RPV_point_ids)

        if debug:
            print("LA ring marking complete.")
        return b_tag, final_centroids

    def mark_ra_rings(self, adjusted_RAA_id: int, rings: list[Ring], b_tag: np.ndarray,
                      centroids: dict, debug: bool = False) -> tuple[np.ndarray, dict, list[Ring]]:
        """
        Marks the right atrial rings (TV, SVC, IVC, CS) by assigning anatomical names,
        updating the b_tag array with boundary tags, and recording centroids.
        """
        if not rings:
            return b_tag, centroids, rings

        # Identify the TV ring by choosing the closer of the two largest rings.
        tv_ring = None
        if len(rings) >= 2:
            largest_two = sorted(rings, key=lambda r: r.np, reverse=True)[:2]

            if largest_two[0].ap_dist < largest_two[1].ap_dist:
                tv_ring = largest_two[0]
            else:
                tv_ring = largest_two[1]

        elif len(rings) == 1:
            tv_ring = rings[0]

        if not tv_ring:
            return b_tag, centroids, rings

        tv_ring.name = "TV"
        if debug:
            print(f"  Identified TV: Ring {tv_ring.id} (Size: {tv_ring.np}, Dist: {tv_ring.ap_dist:.2f})")

        # Cluster remaining rings to identify SVC, IVC, and CS.
        other_indices = [i for i, r in enumerate(rings) if r.name != "TV"]
        if other_indices:
            SVC_indices, IVC_CS_indices, _ = self._cluster_rings(rings, n_clusters=2, debug=debug)
            if SVC_indices:
                svc_idx = max({idx: rings[idx].np for idx in SVC_indices}, key=lambda k: rings[k].np)
                rings[svc_idx].name = "SVC"
                if debug:
                    print(f"  Identified SVC: Ring {svc_idx} (Size: {rings[svc_idx].np})")

            if IVC_CS_indices:
                ivc_idx = max({idx: rings[idx].np for idx in IVC_CS_indices}, key=lambda k: rings[k].np)
                rings[ivc_idx].name = "IVC"

                if debug:
                    print(f"  Identified IVC: Ring {ivc_idx} (Size: {rings[ivc_idx].np})")
                cs_indices = [idx for idx in IVC_CS_indices if idx != ivc_idx]

                if cs_indices:
                    rings[cs_indices[0]].name = "CS"

                    if debug:
                        print(f"  Identified CS: Ring {cs_indices[0]} (Size: {rings[cs_indices[0]].np})")

                        if len(cs_indices) > 1:
                            print(f"    Warning: Multiple CS candidates {cs_indices}, assigned first.")

        final_centroids = centroids.copy()
        tag_map = {"TV": 6, "SVC": 7, "IVC": 8, "CS": 9}
        for ring_obj in rings:
            if not ring_obj.name or ring_obj.name not in tag_map:
                continue

            if not self._validate_vtk_data(ring_obj.vtk_polydata, check_points=True, check_ids=True):
                if debug:
                    print(f"  Skipping RA ring {ring_obj.id} ('{ring_obj.name}') due to invalid data.")
                continue

            point_ids = vtk_to_numpy(ring_obj.vtk_polydata.GetPointData().GetArray("Ids"))

            if point_ids.size == 0:
                continue

            try:
                b_tag[point_ids] = tag_map[ring_obj.name]
            except IndexError:
                print(f"  Error: Point IDs for ring {ring_obj.name} out of bounds. Skipping tagging.")
                continue

            write_vtx_file(os.path.join(self.outdir, f'ids_{ring_obj.name}.vtx'), point_ids)
            final_centroids[ring_obj.name] = ring_obj.center

        write_vtx_file(os.path.join(self.outdir, 'ids_RAA.vtx'), adjusted_RAA_id)
        if debug:
            print("RA ring marking complete.")
        return b_tag, final_centroids, rings

    # -------------------- UAC Path and RSPV Identification --------------------
    def _cutting_plane_to_identify_UAC(self, LPVs_indices: list, RPVs_indices: list,
                                       rings: list[Ring], LA_region: vtk.vtkPolyData, debug: bool = False):
        """
        Identifies MV anterior/posterior boundaries and calculates UAC geodesic paths
        by applying a cutting plane, then using Dijkstra's algorithm on the boundary network.
        """
        # Ensure groups exist; if not, we cannot compute UAC paths.
        if not LPVs_indices or not RPVs_indices:
            return

        try:
            # Calculate the mean centers for LPV and RPV groups.
            LPVs_c = np.array([rings[i].center for i in LPVs_indices])
            lpv_mean = np.mean(LPVs_c, axis=0)
            RPVs_c = np.array([rings[i].center for i in RPVs_indices])
            rpv_mean = np.mean(RPVs_c, axis=0)
            # Find the MV ring's center from the ring with name "MV"
            mv_mean = next(r.center for r in rings if r.name == "MV")
            # Create a cutting plane using MV center and the average PV centers.
            plane = initialize_plane_with_points(mv_mean, rpv_mean, lpv_mean, mv_mean)
        except Exception as e:
            if debug:
                print(f"Error calculating UAC plane: {e}")
            return

        # Extract the geometry from LA_region that lies above the cutting plane.
        surface_above_plane = get_elements_above_plane(LA_region, plane)
        surface_pd = apply_vtk_geom_filter(surface_above_plane)
        # Extract boundary edges from the resulting geometry.
        boundary_edges = get_feature_edges(
            surface_pd,
            boundary_edges_on=True,
            feature_edges_on=False,
            manifold_edges_on=False,
            non_manifold_edges_on=False
        )
        if not self._validate_vtk_data(boundary_edges, check_points=True, check_ids=True):
            if debug:
                print("Warning: Could not extract valid boundary edges for UAC path calculation.")
            return

        # Build a KDTree for boundary points to facilitate geodesic path mapping.
        boundary_points = vtk_to_numpy(boundary_edges.GetPoints().GetData())
        tree = cKDTree(boundary_points)
        ids_on_boundary = vtk_to_numpy(boundary_edges.GetPointData().GetArray("Ids"))

        # For filtering paths later, get all 'Ids' from all rings.
        MV_ids_all = set(
            vtk_to_numpy(next(r.vtk_polydata.GetPointData().GetArray("Ids") for r in rings if r.name == "MV")))
        MV_ant = set(ids_on_boundary).intersection(MV_ids_all)
        MV_post = MV_ids_all - MV_ant
        if MV_ant:
            write_vtx_file(os.path.join(self.outdir, 'ids_MV_ant.vtx'), list(MV_ant))
        if MV_post:
            write_vtx_file(os.path.join(self.outdir, 'ids_MV_post.vtx'), list(MV_post))
        if debug:
            print(f"  MV_ant: {len(MV_ant)} points, MV_post: {len(MV_post)} points.")

        if debug:
            print("  Calculating UAC geodesic paths using Dijkstra's algorithm...")
        # Find start and end vertices on the boundary edges for the paths.
        lpv_bb_idx = find_closest_point(boundary_edges, lpv_mean)
        rpv_bb_idx = find_closest_point(boundary_edges, rpv_mean)
        # Use the MV ring polydata to project the LPV and RPV means onto MV.
        mv_poly = next(r.vtk_polydata for r in rings if r.name == "MV")
        lpv_mv_proj_idx = find_closest_point(mv_poly, lpv_mean)
        rpv_mv_proj_idx = find_closest_point(mv_poly, rpv_mean)
        if lpv_mv_proj_idx < 0 or rpv_mv_proj_idx < 0:
            if debug:
                print("  Warning: Failed to project PV points onto MV ring for UAC calculation.")
            return
        # Get the coordinates from the MV ring for these projections.
        lpv_mv_idx = find_closest_point(boundary_edges, mv_poly.GetPoint(lpv_mv_proj_idx))
        rpv_mv_idx = find_closest_point(boundary_edges, mv_poly.GetPoint(rpv_mv_proj_idx))
        if lpv_bb_idx < 0 or rpv_bb_idx < 0 or lpv_mv_idx < 0 or rpv_mv_idx < 0:
            if debug:
                print("  Warning: Invalid start/end indices for Dijkstra's algorithm.")
            return

        # Gather all ring IDs to later filter out any paths that are part of existing rings.
        all_ring_ids = set()
        for r in rings:
            if r.name and self._validate_vtk_data(r.vtk_polydata, check_ids=True):
                try:
                    all_ring_ids.update(vtk_to_numpy(r.vtk_polydata.GetPointData().GetArray("Ids")))
                except Exception:
                    pass

        # Initialize Dijkstra's algorithm on the boundary edges network.
        dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
        dijkstra.SetInputData(boundary_edges)

        def _compute_and_write_geodesic_path(start_idx, end_idx, out_name):
            """Helper function to compute a geodesic path and write its point IDs to file."""
            if debug:
                print(f"    Computing geodesic path for {out_name}...")
            dijkstra.SetStartVertex(start_idx)
            dijkstra.SetEndVertex(end_idx)
            dijkstra.Update()
            path_poly = dijkstra.GetOutput()
            if self._validate_vtk_data(path_poly, check_points=True):
                path_coords = vtk_to_numpy(path_poly.GetPoints().GetData())

                try:
                    _, path_indices = tree.query(path_coords)
                    path_ids_raw = set(ids_on_boundary[path_indices])
                    path_ids_filtered = list(path_ids_raw - all_ring_ids)
                    if path_ids_filtered:
                        write_vtx_file(os.path.join(self.outdir, out_name), path_ids_filtered)
                        if debug:
                            print(f"Wrote {out_name} with {len(path_ids_filtered)} points.")
                    else:
                        if debug:
                            print(f"Warning: {out_name} path resulted in zero points after filtering.")

                except Exception as e:
                    if debug:
                        print(f"Error during KDTree query for {out_name}: {e}")
            else:
                if debug:
                    print(f"Warning: Failed to compute {out_name} path or path is empty.")

        # Calculate and write three UAC geodesic paths.
        _compute_and_write_geodesic_path(lpv_bb_idx, lpv_mv_idx, 'ids_MV_LPV.vtx')
        _compute_and_write_geodesic_path(rpv_bb_idx, rpv_mv_idx, 'ids_MV_RPV.vtx')
        _compute_and_write_geodesic_path(lpv_bb_idx, rpv_bb_idx, 'ids_RPV_LPV.vtx')

        if debug:
            print("  UAC path calculation finished.")

    def _cutting_plane_to_identify_RSPV(self, LPVs_indices: list, RPVs_indices: list,
                                        rings: list[Ring], debug: bool = False) -> int | None:
        """
        Uses a cutting plane on RPV candidate rings to identify the RSPV.

        Returns:
            The ring ID of the identified RSPV, or None on failure.
        """
        if not LPVs_indices or not RPVs_indices:
            return None
        mv_rings = [r for r in rings if r.name == "MV"]
        if not mv_rings:
            return None
        mv_mean = mv_rings[0].center

        try:
            LPVs_c = np.array([rings[i].center for i in LPVs_indices])
            lpv_mean = np.mean(LPVs_c, axis=0)
            RPVs_c = np.array([rings[i].center for i in RPVs_indices])
            rpv_mean = np.mean(RPVs_c, axis=0)
            plane = initialize_plane_with_points(mv_mean, rpv_mean, lpv_mean, mv_mean)
        except Exception:
            return None

        append_filter = vtk.vtkAppendPolyData()
        valid_rpvs_appended = False
        original_ids = set()
        for i in RPVs_indices:
            ring_obj = rings[i]

            if self._validate_vtk_data(ring_obj.vtk_polydata, check_points=True):
                temp_poly = vtk.vtkPolyData()
                temp_poly.DeepCopy(ring_obj.vtk_polydata)
                tag_data = numpy_to_vtk(np.full((ring_obj.np,), ring_obj.id, dtype=int), deep=True,
                                        array_type=vtk.VTK_INT)
                tag_data.SetName("temp_ring_id")
                temp_poly.GetPointData().AddArray(tag_data)
                append_filter.AddInputData(temp_poly)
                original_ids.add(ring_obj.id)
                valid_rpvs_appended = True
            else:
                if debug:
                    print(f"Warning: Skipping RPV ring {ring_obj.id} due to invalid data.")

        if not valid_rpvs_appended:
            return None

        append_filter.Update()

        extracted_mesh = get_elements_above_plane(append_filter.GetOutput(), plane)
        if self._validate_vtk_data(extracted_mesh, check_points=True) and extracted_mesh.GetPointData().GetArray(
                "temp_ring_id"):
            extracted_ids = vtk_to_numpy(extracted_mesh.GetPointData().GetArray("temp_ring_id"))
            if extracted_ids.size > 0:
                potential_rspv = int(extracted_ids[0])
                if potential_rspv in original_ids:
                    if debug:
                        print(f"RSPV identified: Ring {potential_rspv}")
                    return potential_rspv
                else:
                    if debug:
                        print(f"Warning: Extracted ID {potential_rspv} not in original RPV set.")
                    return None
            else:
                if debug:
                    print("Warning: RSPV extraction yielded no valid IDs.")
                return None
        else:
            if debug:
                print("Warning: RSPV extraction failed or returned no points/IDs.")
            return None

    # -------------------- TV Splitting --------------------
    def _split_tv(self, tv_polydata: vtk.vtkPolyData, tv_center: tuple,
                  ivc_center: tuple, svc_center: tuple, debug: bool = False):
        """
        Splits the tricuspid valve (TV) ring into two regions: free wall (TV_F)
        and septal wall (TV_S). Writes separate VTX files for each.
        """
        if not self._validate_vtk_data(tv_polydata, check_points=True, check_ids=True):
            if debug:
                print("Warning (_split_tv): Invalid TV polydata or missing IDs.")
            return
        if not all([tv_center, ivc_center, svc_center]):
            if debug:
                print("Warning (_split_tv): Missing center points.")
            return

        try:
            # Calculate the normal for the free wall using the cross product
            norm_free = -get_normalized_cross_product(tv_center, svc_center, ivc_center)
            tv_f_plane = initialize_plane(norm_free, tv_center)
        except Exception as e:
            if debug:
                print(f"Error calculating TV_F plane: {e}")
            return

        # Extract TV_F region using the cutting plane
        tv_f_ug = get_elements_above_plane(tv_polydata, tv_f_plane)
        tv_f_pd = apply_vtk_geom_filter(tv_f_ug)
        if self._validate_vtk_data(tv_f_pd, check_points=True, check_ids=True):
            tv_f_ids = vtk_to_numpy(tv_f_pd.GetPointData().GetArray("Ids"))
            if tv_f_ids.size > 0:
                write_vtx_file(os.path.join(self.outdir, 'ids_TV_F.vtx'), tv_f_ids)
                if debug:
                    print(f"Wrote ids_TV_F.vtx ({len(tv_f_ids)} points)")
        else:
            if debug:
                print("Warning: TV_F splitting failed or produced empty results.")

        # Invert the free wall normal to get the septal wall normal
        norm_septal = -norm_free
        tv_s_plane = initialize_plane(norm_septal, tv_center)
        tv_s_ug = get_elements_above_plane(tv_polydata, tv_s_plane, extract_boundary_cells_on=True)
        tv_s_pd = apply_vtk_geom_filter(tv_s_ug)
        if self._validate_vtk_data(tv_s_pd, check_points=True, check_ids=True):
            tv_s_ids = vtk_to_numpy(tv_s_pd.GetPointData().GetArray("Ids"))
            if tv_s_ids.size > 0:
                write_vtx_file(os.path.join(self.outdir, 'ids_TV_S.vtx'), tv_s_ids)
                if debug:
                    print(f"Wrote ids_TV_S.vtx ({len(tv_s_ids)} points)")
        else:
            if debug:
                print("Warning: TV_S splitting failed or produced empty results.")

    # -------------------- Boundary Loop and Region Extraction --------------------
    def _find_top_boundary_loop(self, boundary_edges_polydata: vtk.vtkPolyData,
                                ivc_ring_ids: np.ndarray, svc_ring_ids: np.ndarray,
                                debug: bool = False) -> vtk.vtkPolyData | None:
        """
        Searches through connected boundary regions to find the loop that connects
        points from both the IVC and SVC rings.
        """
        if not self._validate_vtk_data(boundary_edges_polydata, check_points=True, check_ids=True):
            if debug:
                print("_find_top_boundary_loop: Invalid input data.")
            return None

        if ivc_ring_ids.size == 0 or svc_ring_ids.size == 0:
            if debug:
                print("_find_top_boundary_loop: Empty IVC or SVC IDs provided.")
            return None

        connect_filter = init_connectivity_filter(boundary_edges_polydata, ExtractionModes.SPECIFIED_REGIONS)
        num_regions = connect_filter.GetNumberOfExtractedRegions()

        top_cut_loop = None
        ivc_ids_set = set(ivc_ring_ids)
        svc_ids_set = set(svc_ring_ids)
        if debug:
            print(f"_find_top_boundary_loop: Evaluating {num_regions} regions.")

        # Iterate through each connected region to find one that contains points from both rings.
        for region_index in range(num_regions):
            connect_filter.AddSpecifiedRegion(region_index)
            connect_filter.Update()
            loop_ug = connect_filter.GetOutput()
            loop_pd = apply_vtk_geom_filter(loop_ug)
            loop_pd = clean_polydata(loop_pd)
            if self._validate_vtk_data(loop_pd, check_points=True, check_ids=True):
                loop_ids = set(vtk_to_numpy(loop_pd.GetPointData().GetArray("Ids")))
                if loop_ids.intersection(ivc_ids_set) and loop_ids.intersection(svc_ids_set):
                    if debug:
                        print(f"Found connecting loop in region {region_index}.")
                    top_cut_loop = vtk.vtkPolyData()
                    top_cut_loop.DeepCopy(loop_pd)
                    connect_filter.DeleteSpecifiedRegion(region_index)
                    break
            connect_filter.DeleteSpecifiedRegion(region_index)
        if top_cut_loop is None and debug:
            print("Warning: No connecting boundary loop found.")
        return top_cut_loop

    def _get_region_excluding_ids(self, mesh: vtk.vtkPolyData, ids_to_exclude: list[int],
                                  debug: bool = False) -> vtk.vtkPolyData | None:
        """
        Returns the connected region in the mesh that does NOT contain any of the specified IDs.
        This is used to isolate the final TOP_ENDO or TOP_EPI region.
        """
        if not self._validate_vtk_data(mesh, check_points=True, check_ids=True):
            return None
        if not ids_to_exclude:
            return None

        connect_filter = init_connectivity_filter(mesh, ExtractionModes.SPECIFIED_REGIONS)
        num_regions = connect_filter.GetNumberOfExtractedRegions()
        exclude_region_id = -1
        ids_to_exclude_set = set(ids_to_exclude)
        if debug:
            print(f"_get_region_excluding_ids: Evaluating {num_regions} regions for exclusion.")

        # Find which region contains any of the IDs to be excluded.
        for region_index in range(num_regions):
            connect_filter.AddSpecifiedRegion(region_index)
            connect_filter.Update()
            region_ug = connect_filter.GetOutput()
            region_pd = apply_vtk_geom_filter(region_ug)
            region_pd = clean_polydata(region_pd)
            if self._validate_vtk_data(region_pd, check_points=True, check_ids=True):
                region_ids = set(vtk_to_numpy(region_pd.GetPointData().GetArray("Ids")))
                if not region_ids.isdisjoint(ids_to_exclude_set):
                    exclude_region_id = region_index
                    if debug:
                        print(f"Excluding region {region_index} (contains IDs to exclude).")
                    connect_filter.DeleteSpecifiedRegion(region_index)
                    break
            connect_filter.DeleteSpecifiedRegion(region_index)

        # Reinitialize filter to add all regions except the excluded one.
        connect_filter = init_connectivity_filter(mesh, ExtractionModes.SPECIFIED_REGIONS)
        found_target = False
        for region_index in range(num_regions):
            if region_index != exclude_region_id:
                connect_filter.AddSpecifiedRegion(region_index)
                found_target = True
        if not found_target:
            if debug:
                print("No target region found after exclusion.")
            return None
        connect_filter.Update()
        target_ug = connect_filter.GetOutput()
        target_pd = apply_vtk_geom_filter(target_ug)
        target_pd = clean_polydata(target_pd)
        if self._validate_vtk_data(target_pd, check_points=True):
            return target_pd
        else:
            if debug:
                print("Final region after exclusion is empty or invalid.")
            return None

    # -------------------- Standard TOP_ENDO Workflow --------------------
    def cutting_plane_to_identify_tv_f_tv_s(self, model: vtk.vtkPolyData, rings: list[Ring],
                                            debug: bool = True) -> None:
        """
        Implements the standard workflow to split the TV ring and then identify the TOP_ENDO region.
        It uses a cutting plane derived from the TV, SVC, and IVC rings, filters out unwanted points,
        and isolates the region of interest.
        """
        # Retrieve required rings (TV, SVC, IVC) from the provided list.
        tv_ring = next((r for r in rings if r.name == "TV"), None)
        svc_ring = next((r for r in rings if r.name == "SVC"), None)
        ivc_ring = next((r for r in rings if r.name == "IVC"), None)
        if not all([tv_ring, svc_ring, ivc_ring]):
            if debug:
                print("Warning: Missing required rings for TOP_ENDO workflow. Aborting.")
            return

        # Split the TV ring into its free and septal parts.
        self._split_tv(tv_ring.vtk_polydata, tv_ring.center, ivc_ring.center, svc_ring.center, debug=debug)
        if debug:
            print("TV splitting completed.")

        try:
            # Create a cutting plane using the negative normalized cross product.
            norm_1 = -get_normalized_cross_product(tv_ring.center, svc_ring.center, ivc_ring.center)
            tv_f_plane = initialize_plane(norm_1, tv_ring.center)
        except Exception as e:
            if debug:
                print(f"Error creating cutting plane: {e}")
            return

        # Extract geometry above the cutting plane from the input model.
        surface_over = apply_vtk_geom_filter(get_elements_above_plane(model, tv_f_plane))
        if not self._validate_vtk_data(surface_over, check_points=True):
            if debug:
                print("Warning: No geometry extracted above the TV cutting plane.")
            return

        # Extract boundary edges from the extracted surface.
        gamma_top = get_feature_edges(surface_over, boundary_edges_on=True, feature_edges_on=False,
                                      manifold_edges_on=False, non_manifold_edges_on=False)
        if not self._validate_vtk_data(gamma_top, check_points=True, check_ids=True):
            if debug:
                print("Warning: Feature edges extraction failed or missing 'Ids'.")
            return

        # Find the boundary loop that connects the IVC and SVC rings.
        ivc_ids = vtk_to_numpy(ivc_ring.vtk_polydata.GetPointData().GetArray("Ids"))
        svc_ids = vtk_to_numpy(svc_ring.vtk_polydata.GetPointData().GetArray("Ids"))
        top_cut_loop = self._find_top_boundary_loop(gamma_top, ivc_ids, svc_ids, debug=debug)
        if not self._validate_vtk_data(top_cut_loop, check_points=True, check_ids=True):
            if debug:
                print("Warning: Failed to isolate the connecting boundary loop.")
            return

        # Remove points from the loop that belong to the IVC and SVC rings.
        pts_in_loop = vtk_to_numpy(top_cut_loop.GetPointData().GetArray("Ids"))
        svc_ivc_set = set(ivc_ids).union(set(svc_ids))
        to_delete = np.array([1 if pt in svc_ivc_set else 0 for pt in pts_in_loop], dtype=int)
        if debug:
            print(f"Marked {np.sum(to_delete)} SVC/IVC points for deletion from the loop.")
        top_cut_ds = dsa.WrapDataObject(top_cut_loop)
        top_cut_ds.PointData.append(to_delete, "delete")
        thresh = get_lower_threshold(top_cut_ds.VTKObject, 0, "vtkDataObject::FIELD_ASSOCIATION_POINTS", "delete")
        threshed = apply_vtk_geom_filter(thresh.GetOutput())
        threshed = clean_polydata(threshed)
        if not self._validate_vtk_data(threshed, check_points=True, check_ids=True):
            if debug:
                print("Warning: Thresholding/cleaning failed for TOP_ENDO region.")
            return

        # Determine which TV point (from the TV ring) to use as the exclusion marker.
        tv_ids = vtk_to_numpy(tv_ring.vtk_polydata.GetPointData().GetArray("Ids"))
        threshed_ids = vtk_to_numpy(threshed.GetPointData().GetArray("Ids"))
        tv_on_boundary = list(set(tv_ids).intersection(set(threshed_ids)))
        ids_to_exclude = tv_on_boundary[:1] if tv_on_boundary else []
        if not ids_to_exclude:
            if debug:
                print("Warning: No TV points found on the thresholded loop; using fallback.")
            closest_idx = find_closest_point(threshed, tv_ring.center)
            if closest_idx >= 0:
                try:
                    ids_to_exclude = [int(threshed.GetPointData().GetArray("Ids").GetValue(closest_idx))]
                except Exception:
                    return
            else:
                return
        if debug:
            print(f"Using exclude ID(s): {ids_to_exclude}")

        # Isolate the region that does not contain the excluded ID.
        top_endo_region = self._get_region_excluding_ids(threshed, ids_to_exclude, debug=debug)
        if self._validate_vtk_data(top_endo_region, check_points=True, check_ids=True):
            top_endo_ids = vtk_to_numpy(top_endo_region.GetPointData().GetArray("Ids"))
            if top_endo_ids.size > 0:
                write_vtx_file(os.path.join(self.outdir, 'ids_TOP_ENDO.vtx'), top_endo_ids)
                if debug:
                    print(f"TOP_ENDO identification complete with {len(top_endo_ids)} points.")
            else:
                if debug:
                    print("Warning: Final TOP_ENDO region has empty IDs.")
        else:
            if debug:
                print("Warning: Failed to isolate final TOP_ENDO region.")

    # -------------------- Specialized TOP_EPI/ENDO Workflow --------------------
    def _process_top_surface(self, surface_model: vtk.vtkPolyData, plane: vtk.vtkPlane,
                             ivc_ring_ids: np.ndarray, svc_ring_ids: np.ndarray,
                             tv_ring_center: tuple, tv_ring_ids: np.ndarray,
                             surface_name: str, vtx_filename: str, debug: bool = False) -> np.ndarray | None:
        """
        Helper function to process a top surface (either EPI or ENDO) by applying a cutting plane,
        extracting boundary edges, filtering out unwanted regions, and isolating the final TOP region.

        Returns:
            A NumPy array of point IDs for the identified TOP surface, or None on failure.
        """
        if debug:
            print(f"  Starting TOP_{surface_name} identification...")

        if not self._validate_vtk_data(surface_model, check_points=True, check_ids=True):
            if debug:
                print(f"    Warning: Invalid {surface_name} model or missing IDs.")
            return None

        # Extract geometry above the given plane.
        surface_over = apply_vtk_geom_filter(get_elements_above_plane(surface_model, plane))
        if not self._validate_vtk_data(surface_over, check_points=True):
            if debug:
                print(f"    Warning: Failed to extract geometry above the plane for {surface_name}.")
            return None

        # Get the boundary edges (feature edges) from the extracted geometry.
        gamma_top = get_feature_edges(surface_over, boundary_edges_on=True, feature_edges_on=False,
                                      manifold_edges_on=False, non_manifold_edges_on=False)
        if not self._validate_vtk_data(gamma_top, check_points=True, check_ids=True):
            if debug:
                print(f"    Warning: Failed to extract valid feature edges for {surface_name}.")
            return None

        # Find the connecting boundary loop using the provided IVC and SVC IDs.
        top_cut_loop = self._find_top_boundary_loop(gamma_top, ivc_ring_ids, svc_ring_ids, debug=debug)
        if not self._validate_vtk_data(top_cut_loop, check_points=True, check_ids=True):
            if debug:
                print(f"    Warning: Could not isolate the boundary loop for {surface_name}.")
            return None

        # Filter out points from the loop that belong to the IVC/SVC rings.
        pts_in_loop = vtk_to_numpy(top_cut_loop.GetPointData().GetArray("Ids"))
        svc_ivc_set = set(ivc_ring_ids).union(set(svc_ring_ids))
        to_delete = np.array([1 if pt in svc_ivc_set else 0 for pt in pts_in_loop], dtype=int)
        top_cut_ds = dsa.WrapDataObject(top_cut_loop)
        top_cut_ds.PointData.append(to_delete, "delete")
        thresh = get_lower_threshold(top_cut_ds.VTKObject, 0, "vtkDataObject::FIELD_ASSOCIATION_POINTS", "delete")
        threshed = apply_vtk_geom_filter(thresh.GetOutput())
        threshed = clean_polydata(threshed)
        if not self._validate_vtk_data(threshed, check_points=True, check_ids=True):
            if debug:
                print(f"    Warning: Thresholding/cleaning failed for {surface_name} boundary.")
            return None

        # Determine which TV point to exclude by checking for intersection with TV ring IDs.
        threshed_ids = vtk_to_numpy(threshed.GetPointData().GetArray("Ids"))
        tv_ids_on_boundary = list(set(tv_ring_ids).intersection(set(threshed_ids)))
        ids_to_exclude = tv_ids_on_boundary[:1] if tv_ids_on_boundary else []
        if not ids_to_exclude:
            if debug:
                print(f"    Warning: No TV points found on {surface_name} boundary; using fallback.")
            closest_idx = find_closest_point(threshed, tv_ring_center)
            if closest_idx >= 0:
                try:
                    ids_to_exclude = [int(threshed.GetPointData().GetArray("Ids").GetValue(closest_idx))]
                except Exception:
                    return None
            else:
                return None
        if debug:
            print(f"    Exclude ID for {surface_name}: {ids_to_exclude}")

        # Isolate the region that does not contain the exclude ID.
        top_region = self._get_region_excluding_ids(threshed, ids_to_exclude, debug=debug)
        if self._validate_vtk_data(top_region, check_points=True, check_ids=True):
            top_ids = vtk_to_numpy(top_region.GetPointData().GetArray("Ids"))
            if top_ids.size > 0:
                write_vtx_file(os.path.join(self.outdir, vtx_filename), top_ids)
                if debug:
                    print(f"  TOP_{surface_name} identification complete with {len(top_ids)} points.")
                return top_ids
            else:
                if debug:
                    print(f"    Warning: {surface_name} region identified but has empty IDs.")
        else:
            if debug:
                print(f"    Warning: Failed to isolate final {surface_name} region.")
        return None

    # -------------------- Specialized TOP_EPI/ENDO Workflow --------------------
    def perform_tv_split_and_find_top_epi_endo(self, model_epi: vtk.vtkPolyData, endo_mesh_path: str,
                                               rings: list[Ring], debug: bool = True):
        """
        Specialized workflow for separate epi/endo surfaces.
        Splits the TV ring and then identifies separate TOP_EPI and TOP_ENDO regions.

        Args:
            model_epi (vtk.vtkPolyData): The input RA model representing the epi or combined surface.
            endo_mesh_path (str): File path to the separate endocardial mesh (.obj).
            rings (list[Ring]): List of detected RA rings.
            debug (bool): Enables verbose debug output.
        """
        # Retrieve required rings from the list.
        tv_ring = next((r for r in rings if r.name == "TV"), None)
        svc_ring = next((r for r in rings if r.name == "SVC"), None)
        ivc_ring = next((r for r in rings if r.name == "IVC"), None)
        if not all([tv_ring, svc_ring, ivc_ring]):
            if debug:
                print("Warning (TOP_EPI/ENDO): Missing required rings. Aborting specialized workflow.")
            return

        # Step 1: Perform TV splitting using the TV ring polydata.
        self._split_tv(tv_ring.vtk_polydata, tv_ring.center, ivc_ring.center, svc_ring.center, debug=debug)
        if debug:
            print("TV splitting completed.")

        # Step 2: Load the separate endocardial mesh using MeshReader.
        try:
            endo_loader = MeshReader(endo_mesh_path)
            endo_mesh = endo_loader.get_polydata()
            if not self._validate_vtk_data(endo_mesh, check_points=True):
                raise ValueError("Endocardial mesh is empty.")
            if not self._validate_vtk_data(endo_mesh, check_ids=True):
                if debug:
                    print(f"Warning: Endo mesh missing 'Ids'. Generating IDs...")
                endo_mesh = generate_ids(endo_mesh, "Ids", "Ids")
            if not self._validate_vtk_data(endo_mesh, check_ids=True):
                raise ValueError("Failed to generate 'Ids' on endo mesh.")
        except Exception as e:
            if debug:
                print(f"Error: Failed to load/prepare endo mesh: {e}")
            return

        # Step 3: Define the cutting plane using the TV, SVC, and IVC ring centers.
        norm_1 = -get_normalized_cross_product(tv_ring.center, svc_ring.center, ivc_ring.center)
        tv_f_plane = initialize_plane(norm_1, tv_ring.center)

        # Step 4: Process the epi surface using the helper function.
        self._process_top_surface(
            surface_model=model_epi,
            plane=tv_f_plane,
            ivc_ring_ids=vtk_to_numpy(ivc_ring.vtk_polydata.GetPointData().GetArray("Ids")),
            svc_ring_ids=vtk_to_numpy(svc_ring.vtk_polydata.GetPointData().GetArray("Ids")),
            tv_ring_center=tv_ring.center,
            tv_ring_ids=vtk_to_numpy(tv_ring.vtk_polydata.GetPointData().GetArray("Ids")),
            surface_name="EPI",
            vtx_filename="ids_TOP_EPI.vtx",
            debug=debug
        )

        # Step 5: Map epi ring points to the endocardial mesh using KDTree.
        if debug:
            print("Mapping epi ring coordinates to endo mesh IDs...")
        try:
            svc_coords_epi = vtk_to_numpy(svc_ring.vtk_polydata.GetPoints().GetData())
            ivc_coords_epi = vtk_to_numpy(ivc_ring.vtk_polydata.GetPoints().GetData())
            tv_coords_epi = vtk_to_numpy(tv_ring.vtk_polydata.GetPoints().GetData())
            endo_coords = vtk_to_numpy(endo_mesh.GetPoints().GetData())
            endo_ids_all = vtk_to_numpy(endo_mesh.GetPointData().GetArray("Ids"))
            endo_tree = cKDTree(endo_coords)
            _, ii_svc = endo_tree.query(svc_coords_epi)
            pts_in_svc_endo = set(endo_ids_all[ii_svc])
            _, ii_ivc = endo_tree.query(ivc_coords_epi)
            pts_in_ivc_endo = set(endo_ids_all[ii_ivc])
            _, ii_tv = endo_tree.query(tv_coords_epi)
            pts_in_tv_endo = set(endo_ids_all[ii_tv])
            if debug:
                print("Mapping complete.")
        except Exception as e:
            if debug:
                print(f"Error during KDTree mapping: {e}")
            return

        # Step 6: Process the endo surface using the helper function with the mapped IDs.
        self._process_top_surface(
            surface_model=endo_mesh,
            plane=tv_f_plane,
            ivc_ring_ids=np.array(list(pts_in_ivc_endo)),
            svc_ring_ids=np.array(list(pts_in_svc_endo)),
            tv_ring_center=tv_ring.center,
            tv_ring_ids=np.array(list(pts_in_tv_endo)),
            surface_name="ENDO",
            vtx_filename="ids_TOP_ENDO.vtx",
            debug=debug
        )
