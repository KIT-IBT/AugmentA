import os
import numpy as np
from glob import glob
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

from vtk_opencarp_helper_methods.vtk_methods.filters import (
    apply_vtk_geom_filter, clean_polydata, generate_ids, get_center_of_mass,
    get_feature_edges, get_elements_above_plane
)
from vtk_opencarp_helper_methods.vtk_methods.exporting import write_to_vtx
from vtk_opencarp_helper_methods.vtk_methods.finder import find_closest_point
from vtk_opencarp_helper_methods.vtk_methods.init_objects import (
    init_connectivity_filter, ExtractionModes, initialize_plane_with_points, initialize_plane
)
from vtk_opencarp_helper_methods.vtk_methods.converters import vtk_to_numpy, numpy_to_vtk
from vtk_opencarp_helper_methods.mathematical_operations.vector_operations import get_normalized_cross_product
from vtk_opencarp_helper_methods.vtk_methods.thresholding import get_lower_threshold, get_threshold_between

from file_manager import write_vtk, write_vtx_file

from Atrial_LDRBM.Generate_Boundaries.extract_rings import split_tv


class Ring:
    """
    Represents a detected anatomical ring.
    """

    def __init__(self, index, name, points_num, center_point, distance, polydata):
        self.id = index  # Unique ring identifier.
        self.name = name  # Anatomical name (e.g., "MV", "LIPV", etc.) – to be set later.
        self.np = points_num  # Number of points in the ring.
        self.center = center_point  # Center of mass of the ring.
        self.ap_dist = distance  # Distance from the given apex.
        self.vtk_polydata = polydata  # The VTK polydata representing the ring.


class RingDetector:
    """
    Encapsulates the complete ring detection and marking workflow.

    Attributes:
        surface (vtk.vtkPolyData): The input surface.
        apex (tuple): The apex point (coordinate) used for distance calculations.
        outdir (str): Output directory for intermediate file writing.
    """

    def __init__(self, surface: vtk.vtkPolyData, apex, outdir: str):
        self.surface = surface
        self.apex = apex
        self.outdir = outdir

    def detect_rings(self) -> list:
        """
        Detects rings using connectivity filtering.

        Returns:
            list: List of Ring objects.
        """
        boundary_edges = get_feature_edges(
            self.surface,
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
            write_vtk(os.path.join(self.outdir, f'ring_{i}.vtk'), region)
            ring_surf = vtk.vtkPolyData()
            ring_surf.DeepCopy(region)
            center = get_center_of_mass(region, False)
            dist = np.sqrt(np.sum((np.array(self.apex) - np.array(center)) ** 2))
            ring = Ring(i, "", region.GetNumberOfPoints(), center, dist, ring_surf)
            rings.append(ring)
            connect.DeleteSpecifiedRegion(i)
            connect.Update()
        return rings

    def _cluster_rings(self, rings: list) -> (list, list, np.ndarray):
        """
        Clusters rings into two groups using KMeans on the ring centers.

        Returns:
            Tuple: (LPVs indices, RPVs indices, predicted labels)
        """
        indices = [i for i, r in enumerate(rings) if r.name != "MV"]
        centers = [rings[i].center for i in indices]
        estimator = KMeans(n_clusters=2)
        estimator.fit(centers)
        label_pred = estimator.labels_
        LPVs = [indices[i] for i in
                np.where(label_pred == label_pred[np.argmin([rings[i].ap_dist for i in indices])])[0]]
        RPVs = [i for i in indices if i not in LPVs]
        return LPVs, RPVs, label_pred

    def mark_la_rings(self, adjusted_LAA, rings: list, b_tag, centroids: dict, LA_region: vtk.vtkPolyData):
        """
        Marks left atrial rings by assigning names and updating boundary tags and centroids.

        Returns:
            Tuple: (updated b_tag, updated centroids)
        """
        idx_mv = np.argmax([r.np for r in rings])
        rings[idx_mv].name = "MV"
        LPVs, RPVs, _ = self._cluster_rings(rings)
        self._cutting_plane_to_identify_UAC(LPVs, RPVs, rings, LA_region)
        RSPV_id = self._cutting_plane_to_identify_RSPV(LPVs, RPVs, rings)
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
            write_vtx_file(os.path.join(self.outdir, f'ids_{r.name}.vtx'), id_vec)
            centroids[r.name] = r.center
        write_vtx_file(os.path.join(self.outdir, 'ids_LAA.vtx'), adjusted_LAA)
        write_vtx_file(os.path.join(self.outdir, 'ids_LPV.vtx'), LPV_ids)
        write_vtx_file(os.path.join(self.outdir, 'ids_RPV.vtx'), RPV_ids)
        return b_tag, centroids

    def mark_ra_rings(self, adjusted_RAA, rings: list, b_tag, centroids: dict) -> (np.ndarray, dict, list):
        """
        Marks right atrial rings by assigning anatomical names and updating boundary tags.

        Returns:
            Tuple: (updated b_tag, updated centroids, modified rings list)
        """
        largest = sorted(rings, key=lambda r: r.np, reverse=True)[:2]
        tv_ring = largest[0] if largest[0].ap_dist < largest[1].ap_dist else largest[1]
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
            write_vtx_file(os.path.join(self.outdir, f'ids_{r.name}.vtx'), id_vec)
            centroids[r.name] = r.center
        write_vtx_file(os.path.join(self.outdir, 'ids_RAA.vtx'), adjusted_RAA)
        return b_tag, centroids, rings

    def _cutting_plane_to_identify_UAC(self, LPVs, RPVs, rings, LA_region):
        """
        Uses a cutting plane to assist in identifying the UAC region.
        Writes intermediate ID files for MV anterior and posterior segments.
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
        write_vtx_file(os.path.join(self.outdir, 'ids_MV_ant.vtx'), MV_ant)
        write_vtx_file(os.path.join(self.outdir, 'ids_MV_post.vtx'), MV_post)
        return

    def _cutting_plane_to_identify_RSPV(self, LPVs, RPVs, rings):
        """
        Uses a cutting plane to identify the RSPV ring.

        Returns:
            The index of the ring identified as RSPV.
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

    def cutting_plane_to_identify_tv_f_tv_s(self, model: vtk.vtkPolyData, rings: list, debug: bool = True) -> None:
        """
        Identifies TV, SVC, and IVC rings from the RA region and performs TV splitting.

        This method calls the procedural split_tv function.
        """
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
        split_tv(self.outdir, tv_ring.vtk_polydata, tv_ring.center, ivc_ring.center, svc_ring.center)
        if debug:
            print("TV splitting for RA completed.")
