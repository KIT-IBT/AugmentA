import os
from typing import Tuple, List, Any

import numpy as np
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

from vtk_opencarp_helper_methods.vtk_methods.filters import apply_vtk_geom_filter, clean_polydata, generate_ids, get_center_of_mass, get_feature_edges, get_elements_above_plane
from vtk_opencarp_helper_methods.vtk_methods.finder import find_closest_point
from vtk_opencarp_helper_methods.vtk_methods.init_objects import init_connectivity_filter, ExtractionModes, initialize_plane_with_points, initialize_plane
from vtk_opencarp_helper_methods.vtk_methods.converters import vtk_to_numpy, numpy_to_vtk
from vtk_opencarp_helper_methods.mathematical_operations.vector_operations import get_normalized_cross_product
from vtk_opencarp_helper_methods.vtk_methods.thresholding import get_lower_threshold, get_threshold_between
from vtk_opencarp_helper_methods.vtk_methods.reader import smart_reader

from Atrial_LDRBM.Generate_Boundaries.file_manager import write_vtk, write_vtx_file
from Atrial_LDRBM.Generate_Boundaries.mesh import MeshReader


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

    def _validate_vtk_data(
        self,
        vtk_object,
        check_points: bool = True,
        check_cells: bool = False,
        check_ids: bool = False
    ) -> bool:
        """True if the VTK object contains the requested information."""
        if not vtk_object:
            return False

        # points -----------------------------------------------------------------
        if check_points and vtk_object.GetNumberOfPoints() == 0:
            return False

        if check_cells:
            # Accept any dataset that actually contains cells (lines **or** polys)
            if hasattr(vtk_object, "GetNumberOfCells") and vtk_object.GetNumberOfCells() == 0:
                return False

        # Ids array --------------------------------------------------------------
        if check_ids:
            pd = vtk_object.GetPointData()
            if pd is None or pd.GetArray("Ids") is None or pd.GetArray("Ids").GetNumberOfTuples() == 0:
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

        # Use connectivity filtering to separate the boundary edges into connected components
        connect_filter = init_connectivity_filter(boundary_edges, ExtractionModes.ALL_REGIONS)
        num_regions = connect_filter.GetNumberOfExtractedRegions()
        connect_filter.SetExtractionModeToSpecifiedRegions()

        # Process each connected component (potential ring)
        detected_rings = []
        for region_index in range(num_regions):
            ring_obj = self._process_detected_ring_region(connect_filter, region_index, debug)

            if ring_obj:
                detected_rings.append(ring_obj)

            connect_filter.DeleteSpecifiedRegion(region_index)
            connect_filter.Update()

        return detected_rings

    def _process_detected_ring_region(self, connect_filter, region_index, debug=False) -> Ring | None:
        """
        Processes a single connected component from the connectivity filter, cleaning and converting it to a Ring object.
        """
        # Tell the connect filter to focus only on the i-th region
        connect_filter.AddSpecifiedRegion(region_index)
        # Executes the filter for the specified region
        connect_filter.Update()
        # Gets the actual geometric data (points and lines) for this single ring
        region_ug = connect_filter.GetOutput()

        # Converts vtkUnstructuredGrid output from the connectivity filter into a vtkPolyData
        region_pd = apply_vtk_geom_filter(region_ug)
        region_pd = clean_polydata(region_pd)

        if debug:
            # Save a debug version of the ring for inspection
            debug_path = os.path.join(self.outdir, f'ring_{region_index}.vtk')
            write_vtk(debug_path, region_pd)

        ring_poly_copy = vtk.vtkPolyData()
        ring_poly_copy.DeepCopy(region_pd)

        # Compute the center of mass and distance from the apex for classification
        center = get_center_of_mass(region_pd, set_use_scalars_as_weights=False)
        distance = np.sqrt(np.sum((np.array(self.apex) - np.array(center)) ** 2, axis=0))
        num_pts = region_pd.GetNumberOfPoints()

        ring_obj = Ring(region_index, "", num_pts, center, distance, ring_poly_copy)

        return ring_obj

    # -------------------- Ring Classification and Marking --------------------
    def _cluster_rings(self,
                       rings: list[Ring],
                       n_clusters: int = 2,
                       debug: bool = False) -> tuple[list[int], list[int], list[int], int]:
        """
        Clusters non-MV rings using KMeans to separate anatomical groups.
        """
        # Calculating Pulmonary Vein indices
        pvs = [idx for idx, r in enumerate(rings) if r.name != "MV"]

        estimator = KMeans(n_clusters=n_clusters)
        non_mv_centers = [rings[idx].center for idx in pvs]
        labels = estimator.fit_predict(non_mv_centers)

        # Determine which cluster has the ring closest to the apex
        ap_dists = [rings[idx].ap_dist for idx in pvs]
        min_ap_dist = np.argmin(ap_dists)

        label_LPV = labels[min_ap_dist]

        # split into left vs right
        LPVs = [pvs[i] for i, lab in enumerate(labels) if lab == label_LPV]
        RPVs = [pvs[i] for i, lab in enumerate(labels) if lab != label_LPV]

        return pvs, LPVs, RPVs, min_ap_dist

    def _name_pv_subgroup(self,
                          group_indices: list[int],
                          group_name_prefix: str,
                          rings: list[Ring],
                          local_seed_idx: int) -> None:
        """
        Helper method to assign anatomical names (superior/inferior) to a subgroup of PV rings.
        The group_name_prefix is either "L" for left or "R" for right.
        """
        # Get centers of rings in the current group_indices
        estimator = KMeans(n_clusters=2)
        subgroup_centers = [rings[idx].center for idx in group_indices]
        labels = estimator.fit_predict(subgroup_centers)

        seed_label  = labels[local_seed_idx]
        SPV_idxs = [group_indices[i] for i, lab in enumerate(labels) if lab == seed_label]
        IPV_idxs = [group_indices[i] for i, lab in enumerate(labels) if lab != seed_label]

        for idx in SPV_idxs:
            rings[idx].name = f"{group_name_prefix}SPV"
        for idx in IPV_idxs:
            rings[idx].name = f"{group_name_prefix}IPV"


    def mark_la_rings(self,
                      LAA_id: int,
                      rings: list[Ring],
                      b_tag: np.ndarray,
                      centroids: dict,
                      LA_region: vtk.vtkPolyData,
                      debug: bool = False)\
            -> tuple[np.ndarray, dict]:
        """
        Marks the left atrial rings (MV and PVs) by assigning anatomical names, updating
        boundary tags in the b_tag array, and recording centroids.
        """

        # Identify the Mitral Valve ring by choosing the ring with the most points
        mv_idx = np.argmax([r.np for r in rings])
        rings[mv_idx].name = "MV"

        # Cluster non-MV rings into left and right groups.
        pvs, LPVs, RPVs, min_ap_dist = self._cluster_rings(rings, debug=debug)

        LSPV_id = LPVs.index(pvs[min_ap_dist])
        self._cutting_plane_to_identify_UAC(LPVs, RPVs, rings, LA_region, self.outdir)

        global_rspv = self._cutting_plane_to_identify_RSPV(LPVs, RPVs, rings)
        RSPV_id = RPVs.index(global_rspv)

        self._name_pv_subgroup(LPVs, "L", rings, local_seed_idx=LSPV_id)
        self._name_pv_subgroup(RPVs, "R", rings, local_seed_idx=RSPV_id)

        LPV_ids: list[int] = []
        RPV_ids: list[int] = []
        tag_map = {"MV": 1, "LIPV": 2, "LSPV": 3, "RIPV": 4, "RSPV": 5}

        for ring in rings:
            ring_name = ring.name
            id_vec = vtk_to_numpy(ring.vtk_polydata.GetPointData().GetArray("Ids"))

            tag = tag_map.get(ring_name)
            if tag is None:
                continue

            b_tag[id_vec] = tag

            if ring_name in ("LIPV", "LSPV"):
                LPV_ids.extend(id_vec.tolist())
            elif ring_name in ("RIPV", "RSPV"):
                RPV_ids.extend(id_vec.tolist())

            file_path = os.path.join(self.outdir, f"ids_{ring_name}.vtx")
            write_vtx_file(file_path, id_vec)
            centroids[ring_name] = ring.center

        write_vtx_file(os.path.join(self.outdir, "ids_LAA.vtx"), LAA_id)
        write_vtx_file(os.path.join(self.outdir, "ids_LPV.vtx"), LPV_ids)
        write_vtx_file(os.path.join(self.outdir, "ids_RPV.vtx"), RPV_ids)

        return b_tag, centroids

    def mark_ra_rings(self,
                      adjusted_RAA_id: int,
                      rings: list[Ring],
                      b_tag: np.ndarray,
                      centroids: dict,
                      debug: bool = False
                      ) -> tuple[np.ndarray, dict, list[Ring]]:
        """
        Marks the right atrial rings (TV, SVC, IVC, CS) by assigning anatomical names,
        updating the b_tag array with boundary tags, and recording centroids.
        """

        # Identify the TV ring by choosing the closer of the two largest rings.
        lengths = [r.np for r in rings]
        sorted_idx = np.argsort(lengths)
        largest_two = sorted_idx[-2:]
        i0, i1 = largest_two[0], largest_two[1]

        if rings[i0].ap_dist < rings[i1].ap_dist:
            tv_index = i0
        else:
            tv_index = i1

        rings[tv_index].name = "TV"

        # Other ring indices
        other_indices = [i for i in range(len(rings)) if i != tv_index]

        # Cluster other centers
        centers = [rings[i].center for i in other_indices]
        estimator = KMeans(n_clusters=2)
        labels = estimator.fit_predict(centers)

        # Seed SVC by apex distance
        ap_dists = [rings[i].ap_dist for i in other_indices]
        svc_seed = np.argmin(ap_dists)
        svc_label = labels[svc_seed]
        svc_cands = [other_indices[j] for j, lab in enumerate(labels) if lab == svc_label]
        svc_index = svc_cands[0]
        rings[svc_index].name = "SVC"

        # Identify IVC (IVC = largest of the other cluster)
        ivc_cs_indices = [other_indices[j] for j, lab in enumerate(labels) if lab != svc_label]
        ivc_counts = [rings[i].np for i in ivc_cs_indices]
        ivc_pos = int(np.argmax(ivc_counts))
        ivc_index = ivc_cs_indices[ivc_pos]
        rings[ivc_index].name = "IVC"

        # If third ring remains, name it CS
        if len(other_indices) > 2:
            remaining = [i for i in other_indices if i not in {svc_index, ivc_index}]
            cs_index = remaining[0]
            rings[cs_index].name = "CS"

        # Boundary tags, per‐ring .vtx, centroids
        tag_map = {"TV": 6, "SVC": 7, "IVC": 8, "CS": 9}
        for ring in rings:
            name = ring.name # for tiny performance gain
            if name in tag_map:
                ids = vtk_to_numpy(ring.vtk_polydata.GetPointData().GetArray("Ids"))
                b_tag[ids] = tag_map[name]
                write_vtx_file(os.path.join(self.outdir, f"ids_{name}.vtx"), ids)
                centroids[name] = ring.center

        write_vtx_file(os.path.join(self.outdir, "ids_RAA.vtx"), adjusted_RAA_id)

        return b_tag, centroids, rings


    def _cutting_plane_to_identify_UAC(self,
                                       LPV_indices: list[int],
                                       RPV_indices: list[int],
                                       rings: list[Ring],
                                       LA_region: vtk.vtkPolyData,
                                       debug: bool = False) -> None:

        # ring‐center means
        lpv_centers = [rings[i].center for i in LPV_indices]
        lpv_mean = np.mean(lpv_centers, axis=0)

        rpv_centers = [rings[i].center for i in RPV_indices]
        rpv_mean = np.mean(rpv_centers, axis=0)

        mv_index = np.argmax([r.np for r in rings])
        mv_mean = rings[mv_index].center

        plane = initialize_plane_with_points(mv_mean, rpv_mean, lpv_mean, mv_mean)

        # Extract surface above plane and get its boundary edges
        above = get_elements_above_plane(LA_region, plane)
        surface = apply_vtk_geom_filter(above)
        boundary = get_feature_edges(surface,
                                     boundary_edges_on=True,
                                     feature_edges_on=False,
                                     manifold_edges_on=False,
                                     non_manifold_edges_on=False)

        bnd_pts = vtk_to_numpy(boundary.GetPoints().GetData())
        bnd_ids = vtk_to_numpy(boundary.GetPointData().GetArray("Ids"))
        tree = cKDTree(bnd_pts)

        # Collect all ring point IDs
        all_ids = set()
        for ring in rings:
            ids = vtk_to_numpy(ring.vtk_polydata.GetPointData().GetArray("Ids"))
            all_ids.update(ids)

        # MV anterior/posterior
        mv_ring = next(r for r in rings if r.name == "MV")
        mv_ids = set(vtk_to_numpy(mv_ring.vtk_polydata.GetPointData().GetArray("Ids")))
        mv_ant = set(bnd_ids).intersection(mv_ids)
        mv_post = mv_ids - mv_ant

        write_vtx_file(os.path.join(self.outdir, "ids_MV_ant.vtx"), list(mv_ant))
        write_vtx_file(os.path.join(self.outdir, "ids_MV_post.vtx"), list(mv_post))

        # Find boundary‐graph vertices for MV and PV means
        lpv_bb = find_closest_point(boundary, lpv_mean)
        rpv_bb = find_closest_point(boundary, rpv_mean)

        mv_poly = mv_ring.vtk_polydata
        lpv_mv_idx = find_closest_point(mv_poly, lpv_mean)
        rpv_mv_idx = find_closest_point(mv_poly, rpv_mean)
        lpv_mv = find_closest_point(boundary, mv_poly.GetPoint(lpv_mv_idx))
        rpv_mv = find_closest_point(boundary, mv_poly.GetPoint(rpv_mv_idx))

        self._write_uac_path(boundary,
                             tree,
                             bnd_ids,
                             all_ids,
                             lpv_bb,
                             lpv_mv,
                             "ids_MV_LPV.vtx")

        self._write_uac_path(boundary,
                             tree,
                             bnd_ids,
                             all_ids,
                             rpv_bb,
                             rpv_mv,
                             "ids_MV_RPV.vtx")

        self._write_uac_path(boundary,
                             tree,
                             bnd_ids,
                             all_ids,
                             lpv_bb,
                             rpv_bb,
                             "ids_RPV_LPV.vtx")

    def _write_uac_path(self,
                        boundary: vtk.vtkPolyData,
                        tree: cKDTree,
                        bnd_ids: np.ndarray,
                        all_ids: set[int],
                        start_idx: int,
                        end_idx: int,
                        filename: str) -> None:
        """
        Run a Dijkstra geodesic path on `boundary` from start_idx to end_idx,
        map points back to boundary IDs, exclude any in all_ids, and write to disk.
        """
        dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
        dijkstra.SetInputData(boundary)
        dijkstra.SetStartVertex(start_idx)
        dijkstra.SetEndVertex(end_idx)
        dijkstra.Update()

        coords = vtk_to_numpy(dijkstra.GetOutput().GetPoints().GetData())
        _, idxs = tree.query(coords)
        path_ids = set(bnd_ids[idxs]) - all_ids

        write_vtx_file(os.path.join(self.outdir, filename), list(path_ids))


    @staticmethod
    def _cutting_plane_to_identify_RSPV(LPV_indices: list[int],
                                        RPV_indices: list[int],
                                        rings: list[Ring],
                                        debug: bool = False) -> int:
        """
        Uses a cutting plane on RPV candidate rings to identify the RSPV.

        Returns:
            The ring ID of the identified RSPV, or None on failure.
        """
        lpv_centers = [rings[i].center for i in LPV_indices]
        lpv_mean = np.mean(lpv_centers, axis=0)

        rpv_centers = [rings[i].center for i in RPV_indices]
        rpv_mean = np.mean(rpv_centers, axis=0)

        mv_index = int(np.argmax([r.np for r in rings]))
        mv_mean = rings[mv_index].center

        plane = initialize_plane_with_points(mv_mean, rpv_mean, lpv_mean, mv_mean)

        # append all RPV ring meshes, tagging each point with ring.id
        append_filter = vtk.vtkAppendPolyData()

        for idx in RPV_indices:
            ring = rings[idx]
            temp = vtk.vtkPolyData()
            temp.DeepCopy(ring.vtk_polydata)

            # Create a VTK integer array where each point in the ring gets the value `ring.id`
            ids_array = numpy_to_vtk(np.full(ring.np, ring.id, dtype=int), deep=True, array_type=vtk.VTK_INT)
            ids_array.SetName("id")
            temp.GetPointData().SetScalars(ids_array)

            append_filter.AddInputData(temp)

        append_filter.Update()

        # Extract points above the plane
        extracted = get_elements_above_plane(append_filter.GetOutput(), plane)

        # Grab the first “id” value as the RSPV ring id
        id_vtk = extracted.GetPointData().GetArray("id")
        ids = vtk_to_numpy(id_vtk)
        return int(ids[0])


    # -------------------- TV Splitting --------------------
    def _split_tv(self,
                  tv_polydata: vtk.vtkPolyData,
                  tv_center: tuple[float, float, float],
                  ivc_center: tuple[float, float, float],
                  svc_center: tuple[float, float, float],
                  debug: bool = False) -> None:
        """
        Splits the tricuspid valve (TV) ring into two regions: free wall (TV_F)
        and septal wall (TV_S). Writes separate VTX files for each.
        """
        norm1 = -get_normalized_cross_product(tv_center, svc_center, ivc_center)
        plane_f = initialize_plane(norm1, tv_center)
        tv_f = apply_vtk_geom_filter(get_elements_above_plane(tv_polydata, plane_f))
        ids_f = vtk_to_numpy(tv_f.GetPointData().GetArray("Ids"))
        write_vtx_file(os.path.join(self.outdir, "ids_TV_F.vtx"), ids_f)

        norm2 = -norm1
        plane_s = initialize_plane(norm2, tv_center)
        tv_s = apply_vtk_geom_filter(get_elements_above_plane(tv_polydata,plane_s,
                                                              extract_boundary_cells_on=True))
        ids_s = vtk_to_numpy(tv_s.GetPointData().GetArray("Ids"))
        write_vtx_file(os.path.join(self.outdir, "ids_TV_S.vtx"), ids_s)

        if debug:
            print(f"Split TV: {len(ids_f)} free‐wall pts, {len(ids_s)} septal‐wall pts.")

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

    @staticmethod
    def _get_region_not_including_ids(mesh, ids):
        connect = init_connectivity_filter(mesh, ExtractionModes.SPECIFIED_REGIONS)
        num_regions = connect.GetNumberOfExtractedRegions()
        for region_id in range(num_regions):
            connect.AddSpecifiedRegion(region_id)
            connect.Update()
            surface = connect.GetOutput()
            # Clean unused points
            surface = clean_polydata(surface)

            pts_surf = vtk_to_numpy(surface.GetPointData().GetArray("Ids"))

            if ids not in pts_surf:
                found_id = region_id
                # TODO: decided adding the following -> break

            # delete added region id
            connect.DeleteSpecifiedRegion(region_id)
            connect.Update()
        connect.AddSpecifiedRegion(found_id)
        connect.Update()
        return connect.GetOutput()

    # -------------------- TV-F / TV-S split + TOP_ENDO selection (bug-replicated) --
    def cutting_plane_to_identify_tv_f_tv_s(self,
                                            model: vtk.vtkPolyData,
                                            rings: list[Ring],
                                            debug: bool = True) -> None:
        """
        Standard single-surface workflow. Replicates the original bug by feeding
        SVC coordinates in place of IVC coordinates when selecting the top-cut.
        """
        tv_ring = next(r for r in rings if r.name == "TV")
        svc_ring = next(r for r in rings if r.name == "SVC")
        ivc_ring = next(r for r in rings if r.name == "IVC")

        tv_c = np.asarray(tv_ring.center)
        svc_c = np.asarray(svc_ring.center)
        ivc_c = np.asarray(ivc_ring.center)

        plane_f = initialize_plane_with_points(tv_c, svc_c, ivc_c, tv_c)

        surf_over = apply_vtk_geom_filter(get_elements_above_plane(model, plane_f))

        # 1. split the TV
        self._split_tv(tv_ring.vtk_polydata, tv_c, ivc_c, svc_c, debug=debug)

        # 2. BUG-replicating coordinate prep  -------------------------------
        svc_coords = vtk_to_numpy(svc_ring.vtk_polydata.GetPoints().GetData()).tolist()

        top_cut = self._extract_top_cut(surface_over_tv_f=surf_over,
                                        ivc_coords_for_comparison=svc_coords,  # <-- WRONG ON PURPOSE
                                        svc_coords_for_comparison=svc_coords,
                                        debug=debug)

        if debug:
            write_vtk(os.path.join(self.outdir, "top_endo_epi.vtk"), top_cut)

        # 3. Remove genuine SVC / IVC points -------------------------------
        svc_pts = set(vtk_to_numpy(svc_ring.vtk_polydata.GetPointData().GetArray("Ids")))
        ivc_pts = set(vtk_to_numpy(ivc_ring.vtk_polydata.GetPointData().GetArray("Ids")))

        top_ids = vtk_to_numpy(top_cut.GetPointData().GetArray("Ids"))
        to_delete = np.array([1 if (pid in svc_pts or pid in ivc_pts) else 0
                              for pid in top_ids], dtype=int)

        mesh_ds = dsa.WrapDataObject(top_cut)
        mesh_ds.PointData.append(to_delete, "delete")

        th = get_lower_threshold(mesh_ds.VTKObject, 0,
                                 "vtkDataObject::FIELD_ASSOCIATION_POINTS", "delete")
        th_cut = apply_vtk_geom_filter(th.GetOutputPort(), True)

        mv_id = top_ids[0]  # first point on MV (same trick as original code)
        top_endo_ids = self._get_top_endo_ids(mv_id, th_cut)

        write_vtx_file(os.path.join(self.outdir, "ids_TOP_ENDO.vtx"), top_endo_ids)

    # -------------------- Updated Helper for Endo/Epi Top-cut Test --------------------
    @staticmethod
    def _is_top_endo_epi_cut(ivc_ring_coords: List[List[float]],
                             svc_ring_coords: List[List[float]],
                             border_region_coords: List[List[float]]) -> bool:
        """
        Returns ``True`` iff *border_region_coords* contains at least one coordinate
        originating from **both** the IVC and the SVC rings.

        NOTE – this version works on coordinates (not point-IDs) and therefore
        enables the deliberate bug-replication downstream where the SVC
        coordinates are (wrongly) re-used for the IVC test.
        """
        set_ivc = {tuple(pt) for pt in ivc_ring_coords}
        set_svc = {tuple(pt) for pt in svc_ring_coords}

        has_ivc = False
        has_svc = False
        for coord in border_region_coords:
            tup = tuple(coord)

            if not has_ivc and tup in set_ivc:
                has_ivc = True
            if not has_svc and tup in set_svc:
                has_svc = True
            if has_ivc and has_svc:
                return True

        return False

    # -------------------- Extract the “top” loop (bug-replicating variant) ---------
    def _extract_top_cut(self,
                         surface_over_tv_f: vtk.vtkPolyData,
                         ivc_coords_for_comparison: List[List[float]],
                         svc_coords_for_comparison: List[List[float]],
                         debug: bool = False) -> vtk.vtkPolyData:
        """
        From *surface_over_tv_f* (geometry above the TV-F plane) extract the single
        boundary loop that simultaneously touches IVC **and** SVC coordinates
        (NB: the coordinate lists may be intentionally identical to replicate the
        historical bug).
        """
        gamma_top = get_feature_edges(surface_over_tv_f,
                                      boundary_edges_on=True,
                                      feature_edges_on=False,
                                      manifold_edges_on=False,
                                      non_manifold_edges_on=False)

        conn = init_connectivity_filter(gamma_top, ExtractionModes.SPECIFIED_REGIONS)
        num_regions = conn.GetNumberOfExtractedRegions()
        top_region = None

        for region_id in range(num_regions):
            conn.AddSpecifiedRegion(region_id)
            conn.Update()

            reg_pd = clean_polydata(conn.GetOutput())

            border_pts = vtk_to_numpy(reg_pd.GetPoints().GetData()).tolist()

            if self._is_top_endo_epi_cut(ivc_coords_for_comparison,
                                         svc_coords_for_comparison,
                                         border_pts):
                top_region = region_id
                break  # found – stop searching

            # not the one – discard and continue
            conn.DeleteSpecifiedRegion(region_id)
            conn.Update()

        # Fallback – return empty polydata
        if top_region is None:
            return vtk.vtkPolyData()

        # Re-select only the desired region
        #conn.InitializeSpecifiedRegionList()
        conn.AddSpecifiedRegion(top_region)
        conn.Update()
        top_cut = clean_polydata(conn.GetOutput())

        if debug:
            write_vtk_file_path = os.path.join(self.outdir,
                                               f"gamma_top_{top_region}.vtk")
            write_vtk(write_vtk_file_path, top_cut)

        return top_cut

    def _get_region_excluding_ids(self,
                                  mesh: vtk.vtkPolyData,
                                  ids_to_exclude: list[int],
                                  debug: bool = False) -> vtk.vtkPolyData:
        """
        Equivalent to get_region_not_including_ids(mesh, ids) in extract_rings.py.
        Splits mesh into connected regions and returns the first region that
        does NOT contain any of the specified point IDs.
        """
        # initialize connectivity filter for specified regions
        conn = init_connectivity_filter(mesh, ExtractionModes.SPECIFIED_REGIONS)
        num_regions = conn.GetNumberOfExtractedRegions()

        found_region = None
        for region_id in range(num_regions):
            conn.AddSpecifiedRegion(region_id)
            conn.Update()
            region = conn.GetOutput()
            region = clean_polydata(region)

            pts = vtk_to_numpy(region.GetPointData().GetArray("Ids"))

            # If none of the exclude IDs appear here, this is our region
            contains = False
            for ex in ids_to_exclude:
                if ex in pts:
                    contains = True
                    break

            if not contains:
                found_region = region_id
                break

            # Remove and continue
            conn.DeleteSpecifiedRegion(region_id)
            conn.Update()

        if found_region is None:
            # fallback: empty polydata
            return vtk.vtkPolyData()

        # Select the found region and return
        conn.InitializeSpecifiedRegionList()
        conn.AddSpecifiedRegion(found_region)
        conn.Update()
        return conn.GetOutput()


    def _get_top_endo_ids(self,
                          mv_id: int,
                          threshed_mesh: vtk.vtkPolyData,
                          debug: bool = False) -> np.ndarray:
        """
        Equivalent to get_top_endo_ids(...) from extract_rings.py.
        Starts from the thresholded mesh, excludes the single MV ID,
        cleans the result, and returns the remaining point IDs.
        """
        # Exclude MV point from the mesh
        region = self._get_region_excluding_ids(threshed_mesh, [mv_id], debug=debug)
        cleaned = clean_polydata(region)

        # Extract its 'Ids' array
        ids_array = cleaned.GetPointData().GetArray("Ids")
        top_endo_ids = vtk_to_numpy(ids_array)

        return top_endo_ids

    # -------------------- Specialized TOP_EPI/ENDO Workflow --------------------
    # -------------------- Re-usable helper for separate EPI / ENDO ---------
    def _process_top_surface(self,
                             surface_model: vtk.vtkPolyData,
                             plane: vtk.vtkPlane,
                             ivc_coords_for_buggy_comparison: List[List[float]],
                             svc_coords_for_buggy_comparison: List[List[float]],
                             ivc_actual_node_ids: np.ndarray,
                             svc_actual_node_ids: np.ndarray,
                             tv_ring_node_ids_for_mv_id: np.ndarray,
                             surface_name: str,
                             vtx_filename: str,
                             debug: bool = False) -> np.ndarray | None:
        """
        Shared helper for EPI and ENDO passes.  *ivc_coords_for_buggy_comparison*
        and *svc_coords_for_buggy_comparison* are the (possibly duplicated) lists
        of coordinates used to pick the top-cut region, while
        *ivc_actual_node_ids* and *svc_actual_node_ids* are the **true** point-ID
        sets used later for masking.
        """
        surf_over = apply_vtk_geom_filter(get_elements_above_plane(surface_model, plane))

        loop = self._extract_top_cut(surf_over,
                                     ivc_coords_for_buggy_comparison,
                                     svc_coords_for_buggy_comparison,
                                     debug=debug)

        if loop.GetNumberOfPoints() == 0:
            if debug:
                print(f"[{surface_name}] No loop found – skipping.")
            return None

        pts_ids = vtk_to_numpy(loop.GetPointData().GetArray("Ids"))
        excl_ids_set = set(ivc_actual_node_ids).union(set(svc_actual_node_ids))
        mask = np.array([1 if pid in excl_ids_set else 0 for pid in pts_ids],
                        dtype=int)

        ds = dsa.WrapDataObject(loop)
        ds.PointData.append(mask, "delete")

        th = get_lower_threshold(ds.VTKObject, 0,
                                 "vtkDataObject::FIELD_ASSOCIATION_POINTS", "delete")
        threshed = apply_vtk_geom_filter(th.GetOutputPort())

        mv_id = pts_ids[0]  # any MV point suffices (historical approach)
        final_ids = self._get_top_endo_ids(mv_id, threshed, debug=debug)

        write_vtx_file(os.path.join(self.outdir, vtx_filename), final_ids)
        return final_ids

    # -------------------- Public EPI / ENDO workflow (bug-replicating) -----
    def perform_tv_split_and_find_top_epi_endo(self,
                                               model_epi: vtk.vtkPolyData,
                                               endo_mesh_path: str,
                                               rings: list[Ring],
                                               debug: bool = True) -> dict:
        """
        Splits TV, then finds TOP_EPI and TOP_ENDO.  The “bug” is reproduced by
        feeding SVC-derived coordinates in place of IVC coordinates when choosing
        the top-cut on both surfaces.
        """
        tv_ring = next(r for r in rings if r.name == "TV")
        svc_ring = next(r for r in rings if r.name == "SVC")
        ivc_ring = next(r for r in rings if r.name == "IVC")

        # --- TV split -------------------------------------------------------
        self._split_tv(tv_ring.vtk_polydata,
                       tv_ring.center,
                       ivc_ring.center,
                       svc_ring.center,
                       debug=debug)

        norm_vec = -get_normalized_cross_product(tv_ring.center,
                                                 svc_ring.center,
                                                 ivc_ring.center)
        tv_plane = initialize_plane(norm_vec, tv_ring.center)

        # --- EPI ------------------------------------------------------------
        svc_coords_epi = vtk_to_numpy(svc_ring.vtk_polydata.GetPoints().GetData()).tolist()

        epi_ids = self._process_top_surface(
            surface_model=model_epi,
            plane=tv_plane,
            ivc_coords_for_buggy_comparison=svc_coords_epi,  # BUG: SVC passed twice
            svc_coords_for_buggy_comparison=svc_coords_epi,
            ivc_actual_node_ids=vtk_to_numpy(ivc_ring.vtk_polydata.
                                             GetPointData().GetArray("Ids")),
            svc_actual_node_ids=vtk_to_numpy(svc_ring.vtk_polydata.
                                             GetPointData().GetArray("Ids")),
            tv_ring_node_ids_for_mv_id=vtk_to_numpy(tv_ring.vtk_polydata.
                                                    GetPointData().GetArray("Ids")),
            surface_name="EPI",
            vtx_filename="ids_TOP_EPI.vtx",
            debug=debug)

        # --- ENDO mesh ------------------------------------------------------
        endo = MeshReader(endo_mesh_path).get_polydata()
        if not endo.GetPointData().GetArray("Ids"):
            endo = generate_ids(endo, "Ids", "Ids")

        endo_points = vtk_to_numpy(endo.GetPoints().GetData())
        endo_ids = vtk_to_numpy(endo.GetPointData().GetArray("Ids"))
        tree = cKDTree(endo_points)

        # Map epi-ring points to endo surface
        svc_epi_pts = vtk_to_numpy(svc_ring.vtk_polydata.GetPoints().GetData())
        ivc_epi_pts = vtk_to_numpy(ivc_ring.vtk_polydata.GetPoints().GetData())

        _, svc_nn = tree.query(svc_epi_pts)
        _, ivc_nn = tree.query(ivc_epi_pts)

        mapped_svc_ids = endo_ids[svc_nn]
        mapped_ivc_ids = endo_ids[ivc_nn]

        mapped_svc_coords = endo_points[svc_nn].tolist()
        # (mapped_ivc_coords are **not** used – bug replication)

        endo_ids_out = self._process_top_surface(
            surface_model=endo,
            plane=tv_plane,
            ivc_coords_for_buggy_comparison=mapped_svc_coords,
            svc_coords_for_buggy_comparison=mapped_svc_coords,
            ivc_actual_node_ids=mapped_ivc_ids,
            svc_actual_node_ids=mapped_svc_ids,
            tv_ring_node_ids_for_mv_id=vtk_to_numpy(tv_ring.vtk_polydata.GetPointData().GetArray("Ids")),
            surface_name="ENDO",
            vtx_filename="ids_TOP_ENDO.vtx",
            debug=debug)

        return {"TOP_EPI": epi_ids, "TOP_ENDO": endo_ids_out}
