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

# Import helper functions from the VTK helper module.
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

# For tag loading (from separate_epi_endo.py)
from Atrial_LDRBM.Generate_Boundaries.separate_epi_endo import load_element_tags
# Import split_tv from the procedural module (it exists in extract_rings.py)
from Atrial_LDRBM.Generate_Boundaries.extract_rings import split_tv


class Ring:
    def __init__(self, index, name, points_num, center_point, distance, polydata):
        self.id = index
        self.name = name
        self.np = points_num
        self.center = center_point
        self.ap_dist = distance
        self.vtk_polydata = polydata


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

    def _load_mesh(self) -> vtk.vtkPolyData:
        mesh = smart_reader(self.mesh_path)
        return apply_vtk_geom_filter(mesh)

    # --------------------------
    # 1. Mesh Generation
    # --------------------------
    def generate_mesh(self, la_mesh_scale: float = 1.0) -> None:
        """
        Generates a volumetric mesh (VTK format) using meshtool.
        The output is written to "<mesh_path>_vol".
        """
        subprocess.run([
            "meshtool",
            "generate",
            "mesh",
            f"-surf={self.mesh_path}.obj",
            "-ofmt=vtk",
            f"-outmsh={self.mesh_path}_vol"
        ], check=True)
        if self.debug:
            print("Mesh generated.")

    # --------------------------
    # 2. Standard Ring Extraction
    # --------------------------
    def extract_rings(self) -> None:
        """
        Extracts atrial rings from the mesh.
        Supports biatrial and single-chamber processing.
        Writes output VTK files and updates self.ring_info.
        """
        mesh = self._load_mesh()
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
            rings = self.detect_and_mark_rings(LA_region, LA_ap_point, outdir)
            b_tag, centroids = self.mark_LA_rings(adjusted_LAA, rings, b_tag, centroids, outdir, LA_region)
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
            rings_ra = self.detect_and_mark_rings(RA_region, RA_ap_point, outdir)
            b_tag_ra, centroids, rings_ra = self.mark_RA_rings(adjusted_RAA, rings_ra, b_tag_ra, centroids, outdir)
            # Invoke TV splitting for RA:
            self.cutting_plane_to_identify_tv_f_tv_s(RA_region, rings_ra, outdir, self.debug)
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
            rings = self.detect_and_mark_rings(LA_region, LA_ap_point, outdir)
            b_tag, centroids = self.mark_LA_rings(adjusted_LAA, rings, b_tag, centroids, outdir, LA_region)
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
            rings = self.detect_and_mark_rings(RA_region, RA_ap_point, outdir)
            b_tag, centroids, rings = self.mark_RA_rings(adjusted_RAA, rings, b_tag, centroids, outdir)
            # Invoke TV splitting for RA:
            self.cutting_plane_to_identify_tv_f_tv_s(RA_region, rings, outdir, self.debug)
            ds = dsa.WrapDataObject(RA_region)
            ds.PointData.append(b_tag, 'boundary_tag')
            vtk_polydata_writer(os.path.join(outdir, 'RA_boundaries_tagged.vtk'), ds.VTKObject)
        else:
            raise ValueError("At least one of LA or RA apex must be provided.")

        # Save centroids and update ring_info.
        df = pd.DataFrame(centroids)
        csv_path = os.path.join(outdir, "rings_centroids.csv")
        df.to_csv(csv_path, float_format="%.2f", index=False)
        self.ring_info = centroids
        if self.debug:
            print("Ring extraction complete. Centroids saved.")

    def detect_and_mark_rings(self, surf: vtk.vtkPolyData, ap_point, outdir: str) -> list:
        """
        Detects rings via connectivity filtering and returns a list of Ring objects.
        """
        boundary_edges = get_feature_edges(
            surf,
            boundary_edges_on=True,
            feature_edges_on=False,
            manifold_edges_on=False,
            non_manifold_edges_on=False
        )
        connect = init_connectivity_filter(boundary_edges, ExtractionModes.ALL_REGIONS)
        num_regions = connect.GetNumberOfExtractedRegions()
        connect.SetExtractionModeToSpecifiedRegions()
        rings = []
        for i in range(num_regions):
            connect.AddSpecifiedRegion(i)
            connect.Update()
            region = connect.GetOutput()
            region = apply_vtk_geom_filter(region)
            region = clean_polydata(region)
            if self.debug:
                vtk_polydata_writer(os.path.join(outdir, f'ring_{i}.vtk'), region)
            ring_surf = vtk.vtkPolyData()
            ring_surf.DeepCopy(region)
            center = get_center_of_mass(region, False)
            dist = np.sqrt(np.sum((np.array(ap_point) - np.array(center)) ** 2))
            ring = Ring(i, "", region.GetNumberOfPoints(), center, dist, ring_surf)
            rings.append(ring)
            connect.DeleteSpecifiedRegion(i)
            connect.Update()
        return rings

    def mark_LA_rings(self, LAA_id, rings: list, b_tag, centroids: dict, outdir: str, LA_region: vtk.vtkPolyData):
        """
        Marks the left atrial rings using clustering and updates boundary tags and centroids.
        """
        idx_mv = np.argmax([r.np for r in rings])
        rings[idx_mv].name = "MV"
        pvs = [i for i in range(len(rings)) if rings[i].name != "MV"]
        estimator = KMeans(n_clusters=2)
        centers = [rings[i].center for i in range(len(rings)) if rings[i].name != "MV"]
        estimator.fit(centers)
        label_pred = estimator.labels_
        pvs_rings = [rings[i] for i in pvs]
        ap_dists = [r.ap_dist for r in pvs_rings]
        min_idx = np.argmin(ap_dists)
        label_LPV = label_pred[min_idx]
        LPVs = [pvs[i] for i in np.where(label_pred == label_LPV)[0]]
        RPVs = [pvs[i] for i in np.where(label_pred != label_LPV)[0]]

        self.cutting_plane_to_identify_UAC(LPVs, RPVs, rings, LA_region, outdir)
        RSPV_id = self.cutting_plane_to_identify_RSPV(LPVs, RPVs, rings)
        RSPV_id = RPVs.index(RSPV_id)

        LPV_ids = []
        RPV_ids = []
        for r in rings:
            id_vec = vtk_to_numpy(r.vtk_polydata.GetPointData().GetArray("Ids"))
            if r.name == "MV":
                b_tag[id_vec] = 1
            elif r.name == "LIPV":
                b_tag[id_vec] = 2
                LPV_ids.extend(list(id_vec))
            elif r.name == "LSPV":
                b_tag[id_vec] = 3
                LPV_ids.extend(list(id_vec))
            elif r.name == "RIPV":
                b_tag[id_vec] = 4
                RPV_ids.extend(list(id_vec))
            elif r.name == "RSPV":
                b_tag[id_vec] = 5
                RPV_ids.extend(list(id_vec))
            write_to_vtx(os.path.join(outdir, f'ids_{r.name}.vtx'), id_vec, True)
            centroids[r.name] = r.center

        write_to_vtx(os.path.join(outdir, 'ids_LAA.vtx'), LAA_id)
        write_to_vtx(os.path.join(outdir, 'ids_LPV.vtx'), LPV_ids)
        write_to_vtx(os.path.join(outdir, 'ids_RPV.vtx'), RPV_ids)
        return b_tag, centroids

    def mark_RA_rings(self, RAA_id, rings: list, b_tag, centroids: dict, outdir: str) -> Tuple[
        Any, Dict[str, Any], list]:
        """
        Marks the right atrial rings using clustering (assigning "TV", "SVC", "IVC", "CS")
        and updates boundary tags and centroids.
        Mirrors the procedural mark_RA_rings logic, with TV selection based on the two largest rings.
        """
        # Select the two largest rings and choose the one with the lower apical distance as "TV"
        largest = sorted(rings, key=lambda r: r.np, reverse=True)[:2]
        if largest[0].ap_dist < largest[1].ap_dist:
            tv_ring = largest[0]
        else:
            tv_ring = largest[1]
        tv_ring.name = "TV"

        other = [i for i in range(len(rings)) if rings[i].name != "TV"]
        estimator = KMeans(n_clusters=2)
        centers = [rings[i].center for i in range(len(rings)) if rings[i].name != "TV"]
        estimator.fit(centers)
        label_pred = estimator.labels_
        other_rings = [rings[i] for i in other]
        ap_dists = [r.ap_dist for r in other_rings]
        min_idx = np.argmin(ap_dists)
        label_SVC = label_pred[min_idx]
        SVC = other[np.where(label_pred == label_SVC)[0][0]]
        IVC_CS = [other[i] for i in np.where(label_pred != label_SVC)[0]]
        IVC_CS_r = [rings[r] for r in IVC_CS]
        IVC = IVC_CS[np.argmax([r.np for r in IVC_CS_r])]
        rings[SVC].name = "SVC"
        rings[IVC].name = "IVC"
        if len(other) > 2:
            remaining = list(set(other) - set([IVC, SVC]))
            if remaining:
                rings[remaining[0]].name = "CS"

        for r in rings:
            id_vec = vtk_to_numpy(r.vtk_polydata.GetPointData().GetArray("Ids"))
            if r.name == "TV":
                b_tag[id_vec] = 6
            elif r.name == "SVC":
                b_tag[id_vec] = 7
            elif r.name == "IVC":
                b_tag[id_vec] = 8
            elif r.name == "CS":
                b_tag[id_vec] = 9
            write_to_vtx(os.path.join(outdir, f'ids_{r.name}.vtx'), id_vec)
            centroids[r.name] = r.center

        write_to_vtx(os.path.join(outdir, 'ids_RAA.vtx'), RAA_id)
        return b_tag, centroids, rings

    def cutting_plane_to_identify_UAC(self, LPVs, RPVs, rings, LA_region, outdir: str):
        """
        Implements UAC identification using a cutting plane.
        """
        LPVs_c = np.array([rings[i].center for i in LPVs])
        lpv_mean = np.mean(LPVs_c, axis=0)
        RPVs_c = np.array([rings[i].center for i in RPVs])
        rpv_mean = np.mean(RPVs_c, axis=0)
        mv_mean = rings[np.argmax([r.np for r in rings])].center

        plane = initialize_plane_with_points(mv_mean, rpv_mean, lpv_mean, mv_mean)
        surface = apply_vtk_geom_filter(get_elements_above_plane(LA_region, plane))
        boundary_edges = get_feature_edges(surface,
                                           boundary_edges_on=True,
                                           feature_edges_on=False,
                                           manifold_edges_on=False,
                                           non_manifold_edges_on=False)
        tree = cKDTree(vtk_to_numpy(boundary_edges.GetPoints().GetData()))
        ids = vtk_to_numpy(boundary_edges.GetPointData().GetArray('Ids'))
        MV_ring = [r for r in rings if r.name == "MV"]
        MV_ids = set(vtk_to_numpy(MV_ring[0].vtk_polydata.GetPointData().GetArray("Ids")))
        MV_ant = set(ids).intersection(MV_ids)
        MV_post = MV_ids - MV_ant
        write_to_vtx(os.path.join(outdir, 'ids_MV_ant.vtx'), MV_ant)
        write_to_vtx(os.path.join(outdir, 'ids_MV_post.vtx'), MV_post)

        lpv_mv = find_closest_point(MV_ring[0].vtk_polydata, lpv_mean)
        rpv_mv = find_closest_point(MV_ring[0].vtk_polydata, rpv_mean)
        lpv_bb = find_closest_point(boundary_edges, lpv_mean)
        rpv_bb = find_closest_point(boundary_edges, rpv_mean)
        lpv_mv = find_closest_point(boundary_edges, MV_ring[0].vtk_polydata.GetPoint(lpv_mv))
        rpv_mv = find_closest_point(boundary_edges, MV_ring[0].vtk_polydata.GetPoint(rpv_mv))

        path = vtk.vtkDijkstraGraphGeodesicPath()
        path.SetInputData(boundary_edges)
        path.SetStartVertex(lpv_bb)
        path.SetEndVertex(lpv_mv)
        path.Update()
        p = vtk_to_numpy(path.GetOutput().GetPoints().GetData())
        _, ii = tree.query(p)
        mv_lpv = set(ids[ii])
        for r in rings:
            mv_lpv = mv_lpv - set(vtk_to_numpy(r.vtk_polydata.GetPointData().GetArray("Ids")))
        write_to_vtx(os.path.join(outdir, 'ids_MV_LPV.vtx'), mv_lpv)
        # Additional similar steps for other regions would be implemented here.

    def cutting_plane_to_identify_RSPV(self, LPVs, RPVs, rings):
        """
        Implements identification of RSPV using a cutting plane.
        """
        LPVs_c = np.array([rings[i].center for i in LPVs])
        lpv_mean = np.mean(LPVs_c, axis=0)
        RPVs_c = np.array([rings[i].center for i in RPVs])
        rpv_mean = np.mean(RPVs_c, axis=0)
        mv_mean = rings[np.argmax([r.np for r in rings])].center
        plane = initialize_plane_with_points(mv_mean, rpv_mean, lpv_mean, mv_mean)
        appendFilter = vtk.vtkAppendPolyData()
        for i in RPVs:
            r = rings[i]
            tag_data = numpy_to_vtk(np.ones((r.np,)) * r.id, deep=True, array_type=vtk.VTK_INT)
            tag_data.SetNumberOfComponents(1)
            tag_data.SetName("id")
            temp = vtk.vtkPolyData()
            temp.DeepCopy(r.vtk_polydata)
            temp.GetPointData().SetScalars(tag_data)
            appendFilter.AddInputData(temp)
        appendFilter.Update()
        extracted_mesh = get_elements_above_plane(appendFilter.GetOutput(), plane)
        RSPV_id = int(vtk_to_numpy(extracted_mesh.GetPointData().GetArray('id'))[0])
        return RSPV_id

    # --------------------------
    # 3. Epi/Endo Separation
    # --------------------------
    def separate_epi_endo(self, atrium: str) -> None:
        """
        Fully separates the epicardial and endocardial surfaces.
        Implements separate thresholding for epi and endo as in the procedural code.
        Writes both VTK and OBJ files.
        """
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
        # Combined threshold output
        combined_thresh = get_threshold_between(model, endo_tag, epi_tag,
                                                "vtkDataObject::FIELD_ASSOCIATION_CELLS", "tag")
        filtered_combined = apply_vtk_geom_filter(combined_thresh.GetOutput())
        outdir = self._prepare_output_directory("_vol_surf")
        vtk_polydata_writer(os.path.join(outdir, f"{atrium}.vtk"), filtered_combined)
        vtk_obj_writer(os.path.join(outdir, f"{atrium}.obj"), filtered_combined)
        # Separate epicardial surface:
        epi_thresh = get_threshold_between(model, epi_tag, epi_tag,
                                           "vtkDataObject::FIELD_ASSOCIATION_CELLS", "tag")
        filtered_epi = apply_vtk_geom_filter(epi_thresh.GetOutput())
        vtk_polydata_writer(os.path.join(outdir, f"{atrium}_epi.vtk"), filtered_epi)
        vtk_obj_writer(os.path.join(outdir, f"{atrium}_epi.obj"), filtered_epi)
        # Separate endocardial surface:
        endo_thresh = get_threshold_between(model, endo_tag, endo_tag,
                                            "vtkDataObject::FIELD_ASSOCIATION_CELLS", "tag")
        filtered_endo = apply_vtk_geom_filter(endo_thresh.GetOutput())
        vtk_polydata_writer(os.path.join(outdir, f"{atrium}_endo.vtk"), filtered_endo)
        vtk_obj_writer(os.path.join(outdir, f"{atrium}_endo.obj"), filtered_endo)
        if self.debug:
            print(f"Epi/endo separation completed for {atrium}.")

    # --------------------------
    # 4. Surface ID Generation
    # --------------------------
    def generate_surf_id(self, atrium: str, resampled: bool = False) -> None:
        """
        Generates surface ID files mapping mesh points to anatomical boundaries.
        Processes both epicardial and endocardial surfaces.
        Uses write_to_vtx to output ".vtx" files.
        """
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
        # Process endocardial surface:
        endo_obj = smart_reader(f"{base}_{atrium}_endo.obj")
        endo_pts = vtk_to_numpy(endo_obj.GetPoints().GetData())
        _, endo_indices = tree.query(endo_pts)
        endo_indices = np.setdiff1d(endo_indices, epi_ids)
        write_to_vtx(os.path.join(outdir, "ENDO.vtx"), endo_indices)
        if self.debug:
            print(f"Surface ID generation completed for {atrium}.")

    # --------------------------
    # 5. Tag Loading
    # --------------------------
    def load_element_tags(self, csv_filepath: str) -> None:
        """
        Loads element tags from a CSV file.
        """
        self.element_tags = load_element_tags(csv_filepath)
        if self.debug:
            print("Element tags loaded.")

    # --------------------------
    # 6. Top Epi/Endo Extraction
    # --------------------------
    def extract_rings_top_epi_endo(self) -> None:
        """
        Implements top epi/endo ring extraction.
        Delegates to the procedural function.
        """
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
    # New: TV Splitting for RA
    # --------------------------
    def cutting_plane_to_identify_tv_f_tv_s(self, model: vtk.vtkPolyData, rings: list, outdir: str,
                                            debug: bool) -> None:
        """
        Implements the TV splitting logic as in the procedural code.
        Now replicates the detailed "split_tv" functionality.
        """
        # Find the rings for TV, SVC, and IVC
        tv_ring = None
        svc_ring = None
        ivc_ring = None
        for r in rings:
            if r.name == "TV":
                tv_ring = r
            elif r.name == "SVC":
                svc_ring = r
            elif r.name == "IVC":
                ivc_ring = r
        if tv_ring is None or svc_ring is None or ivc_ring is None:
            if debug:
                print("TV splitting: Missing one of TV, SVC, or IVC rings.")
            return
        # Use the split_tv procedural function to perform detailed splitting
        # (This function generates two files: ids_TV_F.vtx and ids_TV_S.vtx)
        split_tv(outdir, tv_ring.vtk_polydata, tv_ring.center, ivc_ring.center, svc_ring.center)
        if debug:
            print("TV splitting for RA completed.")

    def cutting_plane_to_identify_RSPV(self, LPVs, RPVs, rings):
        """
        Implements identification of RSPV using a cutting plane.
        """
        LPVs_c = np.array([rings[i].center for i in LPVs])
        lpv_mean = np.mean(LPVs_c, axis=0)
        RPVs_c = np.array([rings[i].center for i in RPVs])
        rpv_mean = np.mean(RPVs_c, axis=0)
        mv_mean = rings[np.argmax([r.np for r in rings])].center
        plane = initialize_plane_with_points(mv_mean, rpv_mean, lpv_mean, mv_mean)
        appendFilter = vtk.vtkAppendPolyData()
        for i in RPVs:
            r = rings[i]
            tag_data = numpy_to_vtk(np.ones((r.np,)) * r.id, deep=True, array_type=vtk.VTK_INT)
            tag_data.SetNumberOfComponents(1)
            tag_data.SetName("id")
            temp = vtk.vtkPolyData()
            temp.DeepCopy(r.vtk_polydata)
            temp.GetPointData().SetScalars(tag_data)
            appendFilter.AddInputData(temp)
        appendFilter.Update()
        extracted_mesh = get_elements_above_plane(appendFilter.GetOutput(), plane)
        RSPV_id = int(vtk_to_numpy(extracted_mesh.GetPointData().GetArray('id'))[0])
        return RSPV_id

