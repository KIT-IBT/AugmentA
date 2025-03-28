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
    def __init__(self, index, name, points_num, center_point, distance, polydata):
        self.id = index
        self.name = name
        self.np = points_num
        self.center = center_point
        self.ap_dist = distance
        self.vtk_polydata = polydata

def detect_and_mark_rings(surf: vtk.vtkPolyData, ap_point, outdir: str) -> list:
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
        write_vtk(os.path.join(outdir, f'ring_{i}.vtk'), region)
        ring_surf = vtk.vtkPolyData()
        ring_surf.DeepCopy(region)
        center = get_center_of_mass(region, False)
        dist = np.sqrt(np.sum((np.array(ap_point) - np.array(center)) ** 2))
        ring = Ring(i, "", region.GetNumberOfPoints(), center, dist, ring_surf)
        rings.append(ring)
        connect.DeleteSpecifiedRegion(i)
        connect.Update()
    return rings

def mark_LA_rings(LAA_id, rings: list, b_tag, centroids: dict, outdir: str, LA_region: vtk.vtkPolyData):
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

    cutting_plane_to_identify_UAC(LPVs, RPVs, rings, LA_region, outdir)
    RSPV_id = cutting_plane_to_identify_RSPV(LPVs, RPVs, rings)
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
        write_vtx_file(os.path.join(outdir, f'ids_{r.name}.vtx'), id_vec)
        centroids[r.name] = r.center

    write_vtx_file(os.path.join(outdir, 'ids_LAA.vtx'), LAA_id)
    write_vtx_file(os.path.join(outdir, 'ids_LPV.vtx'), LPV_ids)
    write_vtx_file(os.path.join(outdir, 'ids_RPV.vtx'), RPV_ids)
    return b_tag, centroids

def mark_RA_rings(RAA_id, rings: list, b_tag, centroids: dict, outdir: str) -> Tuple[Any, Dict[str, Any], list]:
    """
    Marks the right atrial rings using clustering (assigning "TV", "SVC", "IVC", "CS")
    and updates boundary tags and centroids.
    """
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
        write_vtx_file(os.path.join(outdir, f'ids_{r.name}.vtx'), id_vec)
        centroids[r.name] = r.center

    write_vtx_file(os.path.join(outdir, 'ids_RAA.vtx'), RAA_id)
    return b_tag, centroids, rings

def cutting_plane_to_identify_UAC(LPVs, RPVs, rings, LA_region, outdir: str):
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
    write_vtx_file(os.path.join(outdir, 'ids_MV_ant.vtx'), MV_ant)
    write_vtx_file(os.path.join(outdir, 'ids_MV_post.vtx'), MV_post)

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
    write_vtx_file(os.path.join(outdir, 'ids_MV_LPV.vtx'), mv_lpv)
    # Additional steps can be added here if needed.

def cutting_plane_to_identify_RSPV(LPVs, RPVs, rings):
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

def cutting_plane_to_identify_tv_f_tv_s(model: vtk.vtkPolyData, rings: list, outdir: str, debug: bool) -> None:
    # Find TV, SVC, and IVC rings from RA
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
    split_tv(outdir, tv_ring.vtk_polydata, tv_ring.center, ivc_ring.center, svc_ring.center)
    if debug:
        print("TV splitting for RA completed.")
