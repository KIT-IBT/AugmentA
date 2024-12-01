#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:55:02 2021

@author: Luca Azzolin

Copyright 2021 Luca Azzolin

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.  
"""
import argparse
import os
from glob import glob
from logging import warning

import numpy as np
import pandas as pd
import vtk
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from vtk.numpy_interface import dataset_adapter as dsa

from vtk_opencarp_helper_methods.mathematical_operations.vector_operations import get_normalized_cross_product
from vtk_opencarp_helper_methods.openCARP.exporting import write_to_pts
from vtk_opencarp_helper_methods.vtk_methods.converters import vtk_to_numpy, numpy_to_vtk
from vtk_opencarp_helper_methods.vtk_methods.exporting import vtk_polydata_writer, write_to_vtx
from vtk_opencarp_helper_methods.vtk_methods.filters import apply_vtk_geom_filter, get_vtk_geom_filter_port, \
    clean_polydata, generate_ids, get_center_of_mass, get_feature_edges, get_elements_above_plane
from vtk_opencarp_helper_methods.vtk_methods.finder import find_closest_point
from vtk_opencarp_helper_methods.vtk_methods.init_objects import initialize_plane_with_points, initialize_plane, \
    init_connectivity_filter, ExtractionModes
from vtk_opencarp_helper_methods.vtk_methods.reader import smart_reader
from vtk_opencarp_helper_methods.vtk_methods.thresholding import get_lower_threshold, get_threshold_between

vtk_version = vtk.vtkVersion.GetVTKSourceVersion().split()[-1].split('.')[0]


class Ring:
    def __init__(self, index, name, points_num, center_point, distance, polydata):
        self.id = index
        self.name = name
        self.np = points_num
        self.center = center_point
        self.ap_dist = distance
        self.vtk_polydata = polydata


def parser():
    parser = argparse.ArgumentParser(description='Generate boundaries.')
    parser.add_argument('--mesh',
                        type=str,
                        default="",
                        help='path to meshname')
    parser.add_argument('--LAA',
                        type=str,
                        default="",
                        help='LAA apex point index, leave empty if no LA')
    parser.add_argument('--RAA',
                        type=str,
                        default="",
                        help='RAA apex point index, leave empty if no RA')
    parser.add_argument('--LAA_base',
                        type=str,
                        default="",
                        help='LAA basis point index, leave empty if no LA')
    parser.add_argument('--RAA_base',
                        type=str,
                        default="",
                        help='RAA basis point index, leave empty if no RA')
    parser.add_argument('--debug',
                        type=int,
                        default=1,
                        help='Set to 1 for debbuging the code')
    return parser


def label_atrial_orifices(mesh, LAA_id="", RAA_id="", LAA_base_id="", RAA_base_id="", debug=1):
    """Extrating Rings"""
    print('Extracting rings...')

    mesh_surf = smart_reader(mesh)
    mesh_surf = apply_vtk_geom_filter(mesh_surf)

    centroids = dict()

    extension = mesh.split('.')[-1]
    mesh = mesh[:-(len(extension) + 1)]

    outdir = f"{mesh}_surf"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fname = glob(outdir + '/ids_*')
    for r in fname:
        os.remove(r)
    # Biatrial geometry
    if LAA_id != "" and RAA_id != "":
        LA_ap_point = mesh_surf.GetPoint(int(LAA_id))
        RA_ap_point = mesh_surf.GetPoint(int(RAA_id))

        centroids["LAA"] = LA_ap_point
        centroids["RAA"] = RA_ap_point

        if (LAA_base_id != "" and RAA_base_id != ""):
            LA_bs_point = mesh_surf.GetPoint(int(LAA_base_id))
            RA_bs_point = mesh_surf.GetPoint(int(RAA_base_id))

            centroids["LAA_base"] = LA_bs_point
            centroids["RAA_base"] = RA_bs_point

        mesh_conn = init_connectivity_filter(mesh_surf, ExtractionModes.ALL_REGIONS, True).GetOutput()
        mesh_conn.GetPointData().GetArray("RegionId").SetName("RegionID")
        id_vec = vtk_to_numpy(mesh_conn.GetPointData().GetArray("RegionID"))

        # It can happen that the connectivity filter changes the ids
        LAA_id = find_closest_point(mesh_conn, LA_ap_point)
        RAA_id = find_closest_point(mesh_conn, RA_ap_point)

        LA_tag = id_vec[int(LAA_id)]
        RA_tag = id_vec[int(RAA_id)]

        warning("WARNING: Should be checkt for functionality extract_rings l151")
        thr = get_threshold_between(mesh_conn, LA_tag, LA_tag, "vtkDataObject::FIELD_ASSOCIATION_POINTS", "RegionID")

        mesh_poly = apply_vtk_geom_filter(thr.GetOutputPort(), True)

        LA = generate_ids(mesh_poly, "Ids", "Ids")

        vtk_write(LA, outdir + '/LA.vtk')

        LAA_id = find_closest_point(LA, LA_ap_point)

        if LAA_base_id != "":
            LAA_base_id = find_closest_point(LA, LA_bs_point)

        b_tag = np.zeros((LA.GetNumberOfPoints(),))

        LA_rings = detect_and_mark_rings(LA, LA_ap_point, outdir, debug)
        b_tag, centroids = mark_LA_rings(LAA_id, LA_rings, b_tag, centroids, outdir, LA)
        dataSet = dsa.WrapDataObject(LA)
        dataSet.PointData.append(b_tag, 'boundary_tag')

        vtk_write(dataSet.VTKObject, outdir + '/LA_boundaries_tagged.vtk'.format(mesh))

        thr.ThresholdBetween(RA_tag, RA_tag)
        thr.Update()
        RA_poly = apply_vtk_geom_filter(thr.GetOutputPort(), True)

        RA = generate_ids(RA_poly, "Ids", "Ids")

        RAA_id = find_closest_point(RA, RA_ap_point)

        if LAA_base_id != "":
            RAA_base_id = find_closest_point(RA, RA_bs_point)

        vtk_write(RA, outdir + '/RA.vtk')
        b_tag = np.zeros((RA.GetNumberOfPoints(),))
        RA_rings = detect_and_mark_rings(RA, RA_ap_point, outdir, debug)
        b_tag, centroids, RA_rings = mark_RA_rings(RAA_id, RA_rings, b_tag, centroids, outdir)
        cutting_plane_to_identify_tv_f_tv_s(RA, RA_rings, outdir, debug)

        dataSet = dsa.WrapDataObject(RA)
        dataSet.PointData.append(b_tag, 'boundary_tag')

        vtk_write(dataSet.VTKObject, outdir + '/RA_boundaries_tagged.vtk'.format(mesh))

    elif RAA_id == "":
        vtk_write(mesh_surf, outdir + '/LA.vtk'.format(mesh))
        LA_ap_point = mesh_surf.GetPoint(int(LAA_id))
        centroids["LAA"] = LA_ap_point
        array_name = "Ids"
        if mesh_surf.GetPointData().GetArray(array_name) is not None:
            # Remove previouse id so they match with indices
            mesh_surf.GetPointData().RemoveArray(array_name)

        LA = generate_ids(mesh_surf, "Ids", "Ids")

        LA_rings = detect_and_mark_rings(LA, LA_ap_point, outdir, debug)
        b_tag = np.zeros((LA.GetNumberOfPoints(),))
        b_tag, centroids = mark_LA_rings(LAA_id, LA_rings, b_tag, centroids, outdir, LA)

        dataSet = dsa.WrapDataObject(LA)
        dataSet.PointData.append(b_tag, 'boundary_tag')

        vtk_write(dataSet.VTKObject, outdir + '/LA_boundaries_tagged.vtk'.format(mesh))

    elif LAA_id == "":
        vtk_write(mesh_surf, outdir + '/RA.vtk'.format(mesh))
        RA_ap_point = mesh_surf.GetPoint(int(RAA_id))

        centroids["RAA"] = RA_ap_point
        RA = generate_ids(mesh_surf, "Ids", "Ids")
        RA_rings = detect_and_mark_rings(RA, RA_ap_point, outdir, debug)
        b_tag = np.zeros((RA.GetNumberOfPoints(),))
        b_tag, centroids, RA_rings = mark_RA_rings(RAA_id, RA_rings, b_tag, centroids, outdir)
        cutting_plane_to_identify_tv_f_tv_s(RA, RA_rings, outdir, debug)

        dataSet = dsa.WrapDataObject(RA)
        dataSet.PointData.append(b_tag, 'boundary_tag')

        vtk_write(dataSet.VTKObject, outdir + '/RA_boundaries_tagged.vtk'.format(mesh))

    df = pd.DataFrame(centroids)
    df.to_csv(outdir + "/rings_centroids.csv", float_format="%.2f", index=False)


def run(args=None):
    if args is None:
        args = parser().parse_args()
    else:
        args = parser().parse_args(args)

    label_atrial_orifices(args.mesh, args.LAA, args.RAA, args.LAA_base, args.RAA_base, args.debug)


def detect_and_mark_rings(surf, ap_point, outdir, debug):
    boundary_edges = get_feature_edges(surf, boundary_edges_on=True, feature_edges_on=False, manifold_edges_on=False,
                                       non_manifold_edges_on=False)
    "Splitting rings"
    connect = init_connectivity_filter(boundary_edges, ExtractionModes.ALL_REGIONS)
    num = connect.GetNumberOfExtractedRegions()

    connect.SetExtractionModeToSpecifiedRegions()

    rings = []

    for i in range(num):
        connect.AddSpecifiedRegion(i)
        connect.Update()
        surface = connect.GetOutput()

        # Clean unused points
        surface = apply_vtk_geom_filter(surface)
        surface = clean_polydata(surface)

        # be careful overwrite previous rings
        if debug:
            vtk_write(surface, outdir + '/ring_' + str(i) + '.vtk')

        ring_surf = vtk.vtkPolyData()
        ring_surf.DeepCopy(surface)

        c_mass = get_center_of_mass(surface, False)

        ring = Ring(i, "", surface.GetNumberOfPoints(), c_mass, np.sqrt(np.sum((np.array(ap_point) - \
                                                                                np.array(c_mass)) ** 2, axis=0)),
                    ring_surf)

        rings.append(ring)

        connect.DeleteSpecifiedRegion(i)
        connect.Update()

    return rings


def mark_LA_rings(LAA_id, rings, b_tag, centroids, outdir, LA):
    rings[np.argmax([r.np for r in rings])].name = "MV"
    pvs = [i for i in range(len(rings)) if rings[i].name != "MV"]

    estimator = KMeans(n_clusters=2)
    estimator.fit([r.center for r in rings if r.name != "MV"])
    label_pred = estimator.labels_

    min_ap_dist = np.argmin([r.ap_dist for r in [rings[i] for i in pvs]])
    label_LPV = label_pred[min_ap_dist]

    LPVs = [pvs[i] for i in np.where(label_pred == label_LPV)[0]]
    LSPV_id = LPVs.index(pvs[min_ap_dist])
    RPVs = [pvs[i] for i in np.where(label_pred != label_LPV)[0]]

    cutting_plane_to_identify_UAC(LPVs, RPVs, rings, LA, outdir)

    RSPV_id = cutting_plane_to_identify_RSPV(LPVs, RPVs, rings)
    RSPV_id = RPVs.index(RSPV_id)

    estimator = KMeans(n_clusters=2)
    estimator.fit([r.center for r in [rings[i] for i in LPVs]])
    LPV_lab = estimator.labels_
    LSPVs = [LPVs[i] for i in np.where(LPV_lab == LPV_lab[LSPV_id])[0]]
    LIPVs = [LPVs[i] for i in np.where(LPV_lab != LPV_lab[LSPV_id])[0]]

    estimator = KMeans(n_clusters=2)
    estimator.fit([r.center for r in [rings[i] for i in RPVs]])
    RPV_lab = estimator.labels_
    RSPVs = [RPVs[i] for i in np.where(RPV_lab == RPV_lab[RSPV_id])[0]]
    RIPVs = [RPVs[i] for i in np.where(RPV_lab != RPV_lab[RSPV_id])[0]]

    LPV = []
    RPV = []

    for i in range(len(pvs)):
        if pvs[i] in LSPVs:
            rings[pvs[i]].name = "LSPV"
        elif pvs[i] in LIPVs:
            rings[pvs[i]].name = "LIPV"
        elif pvs[i] in RIPVs:
            rings[pvs[i]].name = "RIPV"
        else:
            rings[pvs[i]].name = "RSPV"

    for r in rings:
        id_vec = vtk_to_numpy(r.vtk_polydata.GetPointData().GetArray("Ids"))
        if r.name == "MV":
            b_tag[id_vec] = 1
        elif r.name == "LIPV":
            b_tag[id_vec] = 2
            LPV = LPV + list(id_vec)
        elif r.name == "LSPV":
            b_tag[id_vec] = 3
            LPV = LPV + list(id_vec)
        elif r.name == "RIPV":
            b_tag[id_vec] = 4
            RPV = RPV + list(id_vec)
        elif r.name == "RSPV":
            b_tag[id_vec] = 5
            RPV = RPV + list(id_vec)

        write_to_vtx(outdir + f'/ids_{r.name}.vtx', id_vec, True)

        centroids[r.name] = r.center

    write_to_vtx(outdir + '/ids_LAA.vtx', LAA_id)
    write_to_vtx(outdir + '/ids_LPV.vtx', LPV)
    write_to_vtx(outdir + '/ids_RPV.vtx', RPV)

    return b_tag, centroids


def mark_RA_rings(RAA_id, rings, b_tag, centroids, outdir):
    """
    Identifies rings of the right atrium and assigns labels corresponding to their position
    Assumes that tricuspid valve is the closest ring to the right atrial appendage of the two larges orifices
    @param RAA_id:
    @param rings: Orifices of the right atria
    @param b_tag:
    @param centroids: points for each orifice of the left atrium. This function addes the points of the right atrium
    @param outdir:
    @return:
    """
    ring_lengths = [r.np for r in rings]
    sorted_indices = np.argsort(ring_lengths)[-2:]
    tv_index = sorted_indices[0] if rings[sorted_indices[0]].ap_dist < rings[sorted_indices[1]].ap_dist else \
        sorted_indices[1]
    rings[tv_index].name = "TV"

    # It can happen that the TV is not the biggest ring!
    # rings[np.argmax([r.np for r in rings])].name = "TV"
    other = [i for i in range(len(rings)) if rings[i].name != "TV"]

    estimator = KMeans(n_clusters=2)
    estimator.fit([r.center for r in rings if r.name != "TV"])
    label_pred = estimator.labels_

    min_ap_dist = np.argmin([r.ap_dist for r in [rings[i] for i in other]])
    label_SVC = label_pred[min_ap_dist]

    SVC = other[np.where(label_pred == label_SVC)[0][0]]
    IVC_CS = [other[i] for i in np.where(label_pred != label_SVC)[0]]
    IVC_CS_r = [rings[r] for r in IVC_CS]
    IVC = IVC_CS[np.argmax([r.np for r in IVC_CS_r])]

    rings[SVC].name = "SVC"
    rings[IVC].name = "IVC"
    if (len(other) > 2):
        rings[list(set(other) - set([IVC, SVC]))[0]].name = "CS"

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

        write_to_vtx(outdir + f'/ids_{r.name}.vtx', id_vec)

        centroids[r.name] = r.center

    write_to_vtx(outdir + '/ids_RAA.vtx', RAA_id)

    return b_tag, centroids, rings


def cutting_plane_to_identify_RSPV(LPVs, RPVs, rings):
    LPVs_c = np.array([r.center for r in [rings[i] for i in LPVs]])
    lpv_mean = np.mean(LPVs_c, axis=0)
    RPVs_c = np.array([r.center for r in [rings[i] for i in RPVs]])
    rpv_mean = np.mean(RPVs_c, axis=0)
    mv_mean = rings[np.argmax([r.np for r in rings])].center

    plane = initialize_plane_with_points(mv_mean, rpv_mean, lpv_mean, mv_mean)

    appendFilter = vtk.vtkAppendPolyData()
    for r in [rings[i] for i in RPVs]:
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


def cutting_plane_to_identify_UAC(LPVs, RPVs, rings, LA, outdir):
    LPVs_c = np.array([r.center for r in [rings[i] for i in LPVs]])
    lpv_mean = np.mean(LPVs_c, axis=0)
    RPVs_c = np.array([r.center for r in [rings[i] for i in RPVs]])
    rpv_mean = np.mean(RPVs_c, axis=0)
    mv_mean = rings[np.argmax([r.np for r in rings])].center

    plane = initialize_plane_with_points(mv_mean, rpv_mean, lpv_mean, mv_mean)

    surface = apply_vtk_geom_filter(get_elements_above_plane(LA, plane))

    """
    here we will extract the feature edge 
    """

    boundary_edges = get_feature_edges(surface, boundary_edges_on=True, feature_edges_on=False, manifold_edges_on=False,
                                       non_manifold_edges_on=False)

    tree = cKDTree(vtk_to_numpy(boundary_edges.GetPoints().GetData()))
    ids = vtk_to_numpy(boundary_edges.GetPointData().GetArray('Ids'))
    MV_ring = [r for r in rings if r.name == "MV"]

    MV_ids = set(vtk_to_numpy(MV_ring[0].vtk_polydata.GetPointData().GetArray("Ids")))

    MV_ant = set(ids).intersection(MV_ids)
    MV_post = MV_ids - MV_ant

    write_to_vtx(outdir + '/ids_MV_ant.vtx', MV_ant)
    write_to_vtx(outdir + '/ids_MV_post.vtx', MV_post)

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
    dd, ii = tree.query(p)
    mv_lpv = set(ids[ii])
    for r in rings:
        mv_lpv = mv_lpv - set(vtk_to_numpy(r.vtk_polydata.GetPointData().GetArray("Ids")))

    write_to_vtx(outdir + '/ids_MV_LPV.vtx', mv_lpv)

    path = vtk.vtkDijkstraGraphGeodesicPath()
    path.SetInputData(boundary_edges)
    path.SetStartVertex(rpv_bb)
    path.SetEndVertex(rpv_mv)
    path.Update()

    p = vtk_to_numpy(path.GetOutput().GetPoints().GetData())
    dd, ii = tree.query(p)
    mv_rpv = set(ids[ii])
    for r in rings:
        mv_rpv = mv_rpv - set(vtk_to_numpy(r.vtk_polydata.GetPointData().GetArray("Ids")))

    write_to_vtx(outdir + '/ids_MV_RPV.vtx', mv_rpv)

    path = vtk.vtkDijkstraGraphGeodesicPath()
    path.SetInputData(boundary_edges)
    path.SetStartVertex(lpv_bb)
    path.SetEndVertex(rpv_bb)
    path.Update()

    p = vtk_to_numpy(path.GetOutput().GetPoints().GetData())
    dd, ii = tree.query(p)
    rpv_lpv = set(ids[ii])
    for r in rings:
        rpv_lpv = rpv_lpv - set(vtk_to_numpy(r.vtk_polydata.GetPointData().GetArray("Ids")))

    write_to_vtx(outdir + '/ids_RPV_LPV.vtx', rpv_lpv)


def cutting_plane_to_identify_tv_f_tv_s(model, rings, outdir, debug):
    for r in rings:
        if r.name == "TV":
            tv_center = np.array(r.center)
            tv = r.vtk_polydata
        elif r.name == "SVC":
            svc_center = np.array(r.center)
            svc = r.vtk_polydata
        elif r.name == "IVC":
            ivc_center = np.array(r.center)
            ivc = r.vtk_polydata

    tv_f_plane = initialize_plane_with_points(tv_center, svc_center, ivc_center, tv_center)

    model_surface = apply_vtk_geom_filter(model)

    surface_over_tv_f = apply_vtk_geom_filter(get_elements_above_plane(model_surface, tv_f_plane))

    if debug:
        vtk_write(surface_over_tv_f, outdir + '/cutted_RA.vtk')

    """
    separate the tv into tv tv-f and tv-f
    """
    split_tv(outdir, tv, tv_center, ivc_center, svc_center)

    svc_points = svc.GetPoints().GetData()
    svc_points = vtk_to_numpy(svc_points)

    ivc_points = svc.GetPoints().GetData()  # Changed
    ivc_points = vtk_to_numpy(ivc_points)

    """
    here we will extract the feature edge 
    """

    top_cut = extract_top_cut(outdir, surface_over_tv_f, ivc_points, svc_points, debug)

    if debug:
        vtk_write(top_cut, outdir + '/top_endo_epi.vtk')  # If this is the CS, then change top_endo_id in 877

    pts_in_top = vtk_to_numpy(top_cut.GetPointData().GetArray("Ids"))
    pts_in_svc = vtk_to_numpy(svc.GetPointData().GetArray("Ids"))
    pts_in_ivc = vtk_to_numpy(ivc.GetPointData().GetArray("Ids"))

    to_delete = np.zeros((len(pts_in_top),), dtype=int)

    for region_id in range(len(pts_in_top)):
        if pts_in_top[region_id] in pts_in_svc or pts_in_top[region_id] in pts_in_ivc:
            to_delete[region_id] = 1

    meshNew = dsa.WrapDataObject(top_cut)
    meshNew.PointData.append(to_delete, "delete")

    thresh = get_lower_threshold(meshNew.VTKObject, 0, "vtkDataObject::FIELD_ASSOCIATION_POINTS", "delete")

    geo_port, _geo_filter = get_vtk_geom_filter_port(thresh.GetOutputPort(), True)

    mv_id = vtk_to_numpy(top_cut.GetPointData().GetArray("Ids"))[0]

    connect = init_connectivity_filter(geo_port, ExtractionModes.SPECIFIED_REGIONS)
    num_regions = connect.GetNumberOfExtractedRegions()

    for region_id in range(num_regions):
        connect.AddSpecifiedRegion(region_id)
        connect.Update()
        surface_over_tv_f = connect.GetOutput()
        # Clean unused points
        surface_over_tv_f = clean_polydata(surface_over_tv_f)

        pts_surf = vtk_to_numpy(surface_over_tv_f.GetPointData().GetArray("Ids"))

        if mv_id not in pts_surf:
            found_id = region_id
            break

        # delete added region id
        connect.DeleteSpecifiedRegion(region_id)
        connect.Update()

    connect.AddSpecifiedRegion(found_id)
    connect.Update()
    surface_over_tv_f = clean_polydata(connect.GetOutput())

    top_endo = vtk_to_numpy(surface_over_tv_f.GetPointData().GetArray("Ids"))

    write_to_vtx(outdir + '/ids_TOP_ENDO.vtx', top_endo)


def extract_top_cut(outdir, surface_over_tv_f, ivc_points, svc_points, debug):
    gamma_top = get_feature_edges(surface_over_tv_f, boundary_edges_on=True, feature_edges_on=False, manifold_edges_on=False,
                                  non_manifold_edges_on=False)
    if debug:
        surface_over_tv_f = apply_vtk_geom_filter(gamma_top)

        vtk_write(surface_over_tv_f, outdir + '/gamma_top.vtk')
    connect = init_connectivity_filter(gamma_top, ExtractionModes.SPECIFIED_REGIONS)
    num_regions = connect.GetNumberOfExtractedRegions()
    for region_id in range(num_regions):
        connect.AddSpecifiedRegion(region_id)
        connect.Update()
        surface_over_tv_f = clean_polydata(connect.GetOutput())

        if debug:
            vtk_write(surface_over_tv_f, outdir + f'/gamma_top_{str(region_id)}.vtk')

        boarder_points = surface_over_tv_f.GetPoints().GetData()
        boarder_points = vtk_to_numpy(boarder_points).tolist()

        if debug:
            create_pts(boarder_points, f'/border_points_{str(region_id)}', outdir)

        if is_top_endo_epi_cut(ivc_points, svc_points, boarder_points):
            top_endo_id = region_id
        else:
            print(f"Region {region_id} is not the top cut for endo and epi")

        # delete added region id
        connect.DeleteSpecifiedRegion(region_id)
        connect.Update()
    # It can happen that the first i=region(0) is the CS. Remove the -1 if that is the case
    connect.AddSpecifiedRegion(top_endo_id - 1)  # Find the id in the points dividing the RA, avoid CS
    connect.Update()
    surface_over_tv_f = connect.GetOutput()
    # Clean unused points
    top_cut = clean_polydata(surface_over_tv_f)
    return top_cut


def split_tv(out_dir, tv, tv_center, ivc_center, svc_center):
    """
    Splits the tricuspid valve along the svc to ivc axis
    :param out_dir: Output directory where the two parts are stored
    :param tv: The tricuspid valve geometry
    :param tv_center: Center of the tricuspid valve
    :param ivc_center: Center of the ivc
    :param svc_center: Center of the svc
    :return:
    """
    norm_1 = get_normalized_cross_product(tv_center, svc_center, ivc_center)
    tv_f_plane = initialize_plane(norm_1[0], tv_center)
    tv_f = apply_vtk_geom_filter(get_elements_above_plane(tv, tv_f_plane))
    tv_f_ids = vtk_to_numpy(tv_f.GetPointData().GetArray("Ids"))
    write_to_vtx(out_dir + '/ids_TV_F.vtx', tv_f_ids)

    norm_2 = - norm_1
    tv_s_plane = initialize_plane(norm_2[0], tv_center)
    tv_s = apply_vtk_geom_filter(get_elements_above_plane(tv, tv_s_plane, extract_boundary_cells_on=True))
    tv_s_ids = vtk_to_numpy(tv_s.GetPointData().GetArray("Ids"))
    write_to_vtx(out_dir + '/ids_TV_S.vtx', tv_s_ids)


def is_top_endo_epi_cut(ivc_points, svc_points, points):
    """
    Returns if the given region cuts the mesh into top and lower cut for endo and epi
    :param ivc_points: Points labeled as IVC (Inferior vena cava)
    :param svc_points: Points labeled as SVC (Superior vena cava)
    :param points: all points associated with this region
    :return:
    """
    in_ivc = False
    in_svc = False
    # if there is point of region_id in both svc and ivc then it is the "top_endo+epi" we need
    for var in points:
        if var in ivc_points:
            in_ivc = True
        if var in svc_points:
            in_svc = True
        if in_ivc and in_svc:
            return True
    return False


def create_pts(array_points, array_name, mesh_dir):
    write_to_pts(f"{mesh_dir}{array_name}.pts", array_points)


def to_polydata(mesh):
    return apply_vtk_geom_filter(mesh)


def vtk_write(input_data, name):
    vtk_polydata_writer(name, input_data)


if __name__ == '__main__':
    run()
