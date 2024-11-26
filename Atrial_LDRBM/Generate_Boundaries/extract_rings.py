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

from vtk_opencarp_helper_methods.AugmentA_methods.vtk_operations import get_normalized_cross_product
from vtk_opencarp_helper_methods.vtk_methods.converters import vtk_to_numpy, numpy_to_vtk
from vtk_opencarp_helper_methods.vtk_methods.exporting import vtk_polydata_writer
from vtk_opencarp_helper_methods.vtk_methods.filters import apply_vtk_geom_filter, get_vtk_geom_filter_port
from vtk_opencarp_helper_methods.vtk_methods.init_objects import initialize_plane_with_points, initialize_plane
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

    meshname = mesh.split("/")[-1]
    outdir = f"{mesh}_surf"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fname = glob(outdir + '/ids_*')
    for r in fname:
        os.remove(r)
    # Biatrial geometry
    if (LAA_id != "" and RAA_id != ""):
        LA_ap_point = mesh_surf.GetPoint(int(LAA_id))
        RA_ap_point = mesh_surf.GetPoint(int(RAA_id))

        centroids["LAA"] = LA_ap_point
        centroids["RAA"] = RA_ap_point

        if (LAA_base_id != "" and RAA_base_id != ""):
            LA_bs_point = mesh_surf.GetPoint(int(LAA_base_id))
            RA_bs_point = mesh_surf.GetPoint(int(RAA_base_id))

            centroids["LAA_base"] = LA_bs_point
            centroids["RAA_base"] = RA_bs_point

        connect = vtk.vtkConnectivityFilter()
        connect.SetInputData(mesh_surf)
        connect.SetExtractionModeToAllRegions()
        connect.ColorRegionsOn()
        connect.Update()
        mesh_conn = connect.GetOutput()
        mesh_conn.GetPointData().GetArray("RegionId").SetName("RegionID")
        id_vec = vtk_to_numpy(mesh_conn.GetPointData().GetArray("RegionID"))

        # It can happen that the connectivity filter changes the ids
        loc = vtk.vtkPointLocator()
        loc.SetDataSet(mesh_conn)
        loc.BuildLocator()
        LAA_id = loc.FindClosestPoint(LA_ap_point)
        RAA_id = loc.FindClosestPoint(RA_ap_point)

        LA_tag = id_vec[int(LAA_id)]
        RA_tag = id_vec[int(RAA_id)]

        warning("WARNING: Should be checkt for functionality extract_rings l151")
        thr = get_threshold_between(mesh_conn, LA_tag, LA_tag, "vtkDataObject::FIELD_ASSOCIATION_POINTS", "RegionID")

        geo_port, _geo_filter = get_vtk_geom_filter_port(thr.GetOutputPort(), True)

        idFilter = vtk.vtkIdFilter()
        idFilter.SetInputConnection(geo_port)
        if int(vtk_version) >= 9:
            idFilter.SetPointIdsArrayName('Ids')
            idFilter.SetCellIdsArrayName('Ids')
        else:
            idFilter.SetIdsArrayName('Ids')
        idFilter.Update()

        LA = idFilter.GetOutput()

        vtkWrite(LA, outdir + '/LA.vtk')

        loc = vtk.vtkPointLocator()
        loc.SetDataSet(LA)
        loc.BuildLocator()
        LAA_id = loc.FindClosestPoint(LA_ap_point)

        if LAA_base_id != "":
            loc = vtk.vtkPointLocator()
            loc.SetDataSet(LA)
            loc.BuildLocator()
            LAA_base_id = loc.FindClosestPoint(LA_bs_point)

        b_tag = np.zeros((LA.GetNumberOfPoints(),))

        LA_rings = detect_and_mark_rings(LA, LA_ap_point, outdir, debug)
        b_tag, centroids = mark_LA_rings(LAA_id, LA_rings, b_tag, centroids, outdir, LA)
        dataSet = dsa.WrapDataObject(LA)
        dataSet.PointData.append(b_tag, 'boundary_tag')

        vtkWrite(dataSet.VTKObject, outdir + '/LA_boundaries_tagged.vtk'.format(mesh))

        thr.ThresholdBetween(RA_tag, RA_tag)
        thr.Update()
        geo_port, _geo_filter = get_vtk_geom_filter_port(thr.GetOutputPort(), True)



        idFilter = vtk.vtkIdFilter()
        idFilter.SetInputConnection(geo_port)
        if int(vtk_version) >= 9:
            idFilter.SetPointIdsArrayName('Ids')
            idFilter.SetCellIdsArrayName('Ids')
        else:
            idFilter.SetIdsArrayName('Ids')
        idFilter.Update()

        RA = idFilter.GetOutput()

        loc = vtk.vtkPointLocator()
        loc.SetDataSet(RA)
        loc.BuildLocator()
        RAA_id = loc.FindClosestPoint(RA_ap_point)

        if LAA_base_id != "":
            loc = vtk.vtkPointLocator()
            loc.SetDataSet(RA)
            loc.BuildLocator()
            RAA_base_id = loc.FindClosestPoint(RA_bs_point)

        vtkWrite(RA, outdir + '/RA.vtk')
        b_tag = np.zeros((RA.GetNumberOfPoints(),))
        RA_rings = detect_and_mark_rings(RA, RA_ap_point, outdir, debug)
        b_tag, centroids, RA_rings = mark_RA_rings(RAA_id, RA_rings, b_tag, centroids, outdir)
        cutting_plane_to_identify_tv_f_tv_s(RA, RA_rings, outdir, debug)

        dataSet = dsa.WrapDataObject(RA)
        dataSet.PointData.append(b_tag, 'boundary_tag')

        vtkWrite(dataSet.VTKObject, outdir + '/RA_boundaries_tagged.vtk'.format(mesh))

    elif RAA_id == "":
        vtkWrite(mesh_surf, outdir + '/LA.vtk'.format(mesh))
        LA_ap_point = mesh_surf.GetPoint(int(LAA_id))
        centroids["LAA"] = LA_ap_point
        array_name = "Ids"
        if mesh_surf.GetPointData().GetArray(array_name) is not None:
            # Remove previouse id so they match with indices
            mesh_surf.GetPointData().RemoveArray(array_name)
        idFilter = vtk.vtkIdFilter()
        idFilter.SetInputData(mesh_surf)
        if int(vtk_version) >= 9:
            idFilter.SetPointIdsArrayName('Ids')
            idFilter.SetCellIdsArrayName('Ids')
        else:
            idFilter.SetIdsArrayName('Ids')
        idFilter.Update()
        LA = idFilter.GetOutput()

        LA_rings = detect_and_mark_rings(LA, LA_ap_point, outdir, debug)
        b_tag = np.zeros((LA.GetNumberOfPoints(),))
        b_tag, centroids = mark_LA_rings(LAA_id, LA_rings, b_tag, centroids, outdir, LA)

        dataSet = dsa.WrapDataObject(LA)
        dataSet.PointData.append(b_tag, 'boundary_tag')

        vtkWrite(dataSet.VTKObject, outdir + '/LA_boundaries_tagged.vtk'.format(mesh))

    elif LAA_id == "":
        vtkWrite(mesh_surf, outdir + '/RA.vtk'.format(mesh))
        RA_ap_point = mesh_surf.GetPoint(int(RAA_id))
        idFilter = vtk.vtkIdFilter()
        idFilter.SetInputData(mesh_surf)
        if int(vtk_version) >= 9:
            idFilter.SetPointIdsArrayName('Ids')
            idFilter.SetCellIdsArrayName('Ids')
        else:
            idFilter.SetIdsArrayName('Ids')
        idFilter.Update()
        centroids["RAA"] = RA_ap_point
        RA = idFilter.GetOutput()
        RA_rings = detect_and_mark_rings(RA, RA_ap_point, outdir, debug)
        b_tag = np.zeros((RA.GetNumberOfPoints(),))
        b_tag, centroids, RA_rings = mark_RA_rings(RAA_id, RA_rings, b_tag, centroids, outdir)
        cutting_plane_to_identify_tv_f_tv_s(RA, RA_rings, outdir, debug)

        dataSet = dsa.WrapDataObject(RA)
        dataSet.PointData.append(b_tag, 'boundary_tag')

        vtkWrite(dataSet.VTKObject, outdir + '/RA_boundaries_tagged.vtk'.format(mesh))

    df = pd.DataFrame(centroids)
    df.to_csv(outdir + "/rings_centroids.csv", float_format="%.2f", index=False)


def run(args=None):
    if args is None:
        args = parser().parse_args()
    else:
        args = parser().parse_args(args)

    label_atrial_orifices(args.mesh, args.LAA, args.RAA, args.LAA_base, args.RAA_base, args.debug)


def detect_and_mark_rings(surf, ap_point, outdir, debug):
    boundaryEdges = vtk.vtkFeatureEdges()
    boundaryEdges.SetInputData(surf)
    boundaryEdges.BoundaryEdgesOn()
    boundaryEdges.FeatureEdgesOff()
    boundaryEdges.ManifoldEdgesOff()
    boundaryEdges.NonManifoldEdgesOff()
    boundaryEdges.Update()

    "Splitting rings"

    connect = vtk.vtkConnectivityFilter()
    connect.SetInputData(boundaryEdges.GetOutput())
    connect.SetExtractionModeToAllRegions()
    connect.Update()
    num = connect.GetNumberOfExtractedRegions()

    connect.SetExtractionModeToSpecifiedRegions()

    rings = []

    for i in range(num):
        connect.AddSpecifiedRegion(i)
        connect.Update()
        surface = connect.GetOutput()

        # Clean unused points
        surface = apply_vtk_geom_filter(surface)


        cln = vtk.vtkCleanPolyData()
        cln.SetInputData(surface)
        cln.Update()
        surface = cln.GetOutput()

        # be careful overwrite previous rings
        if debug:
            vtkWrite(surface, outdir + '/ring_' + str(i) + '.vtk')

        ring_surf = vtk.vtkPolyData()
        ring_surf.DeepCopy(surface)

        centerOfMassFilter = vtk.vtkCenterOfMass()
        centerOfMassFilter.SetInputData(surface)
        centerOfMassFilter.SetUseScalarsAsWeights(False)
        centerOfMassFilter.Update()

        c_mass = centerOfMassFilter.GetCenter()

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
        fname = outdir + f'/ids_{r.name}.vtx'
        if os.path.exists(fname):
            id_vec = id_vec[0:len(id_vec) - 1]
            f = open(fname, 'a')
        else:
            f = open(fname, 'w')
            f.write(f'{len(id_vec)}\n')
            f.write('extra\n')

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

        for i in id_vec:
            f.write(f'{i}\n')
        f.close()

        centroids[r.name] = r.center

    fname = outdir + '/ids_LAA.vtx'
    f = open(fname, 'w')
    f.write(f'{1}\n')
    f.write('extra\n')
    f.write(f'{LAA_id}\n')
    f.close()

    fname = outdir + '/ids_LPV.vtx'
    f = open(fname, 'w')
    f.write(f'{len(LPV)}\n')
    f.write('extra\n')
    for i in LPV:
        f.write(f'{i}\n')
    f.close()

    fname = outdir + '/ids_RPV.vtx'
    f = open(fname, 'w')
    f.write(f'{len(RPV)}\n')
    f.write('extra\n')
    for i in RPV:
        f.write(f'{i}\n')
    f.close()

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
        fname = outdir + f'/ids_{r.name}.vtx'

        f = open(fname, 'w')
        f.write(f'{len(id_vec)}\n')
        f.write('extra\n')

        if r.name == "TV":
            b_tag[id_vec] = 6
        elif r.name == "SVC":
            b_tag[id_vec] = 7
        elif r.name == "IVC":
            b_tag[id_vec] = 8
        elif r.name == "CS":
            b_tag[id_vec] = 9

        for i in id_vec:
            f.write(f'{i}\n')

        f.close()

        centroids[r.name] = r.center

    fname = outdir + '/ids_RAA.vtx'
    f = open(fname, 'w')
    f.write(f'{1}\n')
    f.write('extra\n')
    f.write(f'{RAA_id}\n')
    f.close()

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

    meshExtractFilter = vtk.vtkExtractGeometry()
    meshExtractFilter.SetInputData(appendFilter.GetOutput())
    meshExtractFilter.SetImplicitFunction(plane)
    meshExtractFilter.Update()

    RSPV_id = int(vtk_to_numpy(meshExtractFilter.GetOutput().GetPointData().GetArray('id'))[0])

    return RSPV_id


def cutting_plane_to_identify_UAC(LPVs, RPVs, rings, LA, outdir):
    LPVs_c = np.array([r.center for r in [rings[i] for i in LPVs]])
    lpv_mean = np.mean(LPVs_c, axis=0)
    RPVs_c = np.array([r.center for r in [rings[i] for i in RPVs]])
    rpv_mean = np.mean(RPVs_c, axis=0)
    mv_mean = rings[np.argmax([r.np for r in rings])].center

    plane = initialize_plane_with_points(mv_mean, rpv_mean, lpv_mean, mv_mean)

    meshExtractFilter = vtk.vtkExtractGeometry()
    meshExtractFilter.SetInputData(LA)
    meshExtractFilter.SetImplicitFunction(plane)
    meshExtractFilter.Update()

    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(meshExtractFilter.GetOutput())
    geo_filter.Update()
    surface = geo_filter.GetOutput()

    """
    here we will extract the feature edge 
    """
    boundaryEdges = vtk.vtkFeatureEdges()
    boundaryEdges.SetInputData(surface)
    boundaryEdges.BoundaryEdgesOn()
    boundaryEdges.FeatureEdgesOff()
    boundaryEdges.ManifoldEdgesOff()
    boundaryEdges.NonManifoldEdgesOff()
    boundaryEdges.Update()

    tree = cKDTree(vtk_to_numpy(boundaryEdges.GetOutput().GetPoints().GetData()))
    ids = vtk_to_numpy(boundaryEdges.GetOutput().GetPointData().GetArray('Ids'))
    MV_ring = [r for r in rings if r.name == "MV"]

    MV_ids = set(vtk_to_numpy(MV_ring[0].vtk_polydata.GetPointData().GetArray("Ids")))

    MV_ant = set(ids).intersection(MV_ids)
    MV_post = MV_ids - MV_ant

    fname = outdir + '/ids_MV_ant.vtx'
    f = open(fname, 'w')
    f.write(f'{len(MV_ant)}\n')
    f.write('extra\n')
    for i in MV_ant:
        f.write(f'{i}\n')
    f.close()

    fname = outdir + '/ids_MV_post.vtx'
    f = open(fname, 'w')
    f.write(f'{len(MV_post)}\n')
    f.write('extra\n')
    for i in MV_post:
        f.write(f'{i}\n')
    f.close()

    loc = vtk.vtkPointLocator()
    loc.SetDataSet(MV_ring[0].vtk_polydata)
    loc.BuildLocator()

    lpv_mv = loc.FindClosestPoint(lpv_mean)
    rpv_mv = loc.FindClosestPoint(rpv_mean)

    loc = vtk.vtkPointLocator()
    loc.SetDataSet(boundaryEdges.GetOutput())
    loc.BuildLocator()
    lpv_bb = loc.FindClosestPoint(lpv_mean)
    rpv_bb = loc.FindClosestPoint(rpv_mean)
    lpv_mv = loc.FindClosestPoint(MV_ring[0].vtk_polydata.GetPoint(lpv_mv))
    rpv_mv = loc.FindClosestPoint(MV_ring[0].vtk_polydata.GetPoint(rpv_mv))

    path = vtk.vtkDijkstraGraphGeodesicPath()
    path.SetInputData(boundaryEdges.GetOutput())
    path.SetStartVertex(lpv_bb)
    path.SetEndVertex(lpv_mv)
    path.Update()

    p = vtk_to_numpy(path.GetOutput().GetPoints().GetData())
    dd, ii = tree.query(p)
    mv_lpv = set(ids[ii])
    for r in rings:
        mv_lpv = mv_lpv - set(vtk_to_numpy(r.vtk_polydata.GetPointData().GetArray("Ids")))

    fname = outdir + '/ids_MV_LPV.vtx'
    f = open(fname, 'w')
    f.write(f'{len(mv_lpv)}\n')
    f.write('extra\n')
    for i in mv_lpv:
        f.write(f'{i}\n')
    f.close()

    path = vtk.vtkDijkstraGraphGeodesicPath()
    path.SetInputData(boundaryEdges.GetOutput())
    path.SetStartVertex(rpv_bb)
    path.SetEndVertex(rpv_mv)
    path.Update()

    p = vtk_to_numpy(path.GetOutput().GetPoints().GetData())
    dd, ii = tree.query(p)
    mv_rpv = set(ids[ii])
    for r in rings:
        mv_rpv = mv_rpv - set(vtk_to_numpy(r.vtk_polydata.GetPointData().GetArray("Ids")))

    fname = outdir + '/ids_MV_RPV.vtx'
    f = open(fname, 'w')
    f.write(f'{len(mv_rpv)}\n')
    f.write('extra\n')
    for i in mv_rpv:
        f.write(f'{i}\n')
    f.close()

    path = vtk.vtkDijkstraGraphGeodesicPath()
    path.SetInputData(boundaryEdges.GetOutput())
    path.SetStartVertex(lpv_bb)
    path.SetEndVertex(rpv_bb)
    path.Update()

    p = vtk_to_numpy(path.GetOutput().GetPoints().GetData())
    dd, ii = tree.query(p)
    rpv_lpv = set(ids[ii])
    for r in rings:
        rpv_lpv = rpv_lpv - set(vtk_to_numpy(r.vtk_polydata.GetPointData().GetArray("Ids")))

    fname = outdir + '/ids_RPV_LPV.vtx'
    f = open(fname, 'w')
    f.write(f'{len(rpv_lpv)}\n')
    f.write('extra\n')
    for i in rpv_lpv:
        f.write(f'{i}\n')
    f.close()


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

    plane = initialize_plane_with_points(tv_center, svc_center, ivc_center, tv_center)

    surface = apply_vtk_geom_filter(model)


    meshExtractFilter = vtk.vtkExtractGeometry()
    meshExtractFilter.SetInputData(surface)
    meshExtractFilter.SetImplicitFunction(plane)
    meshExtractFilter.Update()

    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(meshExtractFilter.GetOutput())
    geo_filter.Update()
    surface = geo_filter.GetOutput()

    if debug:
        vtkWrite(surface, outdir + '/cutted_RA.vtk')

    """
    here we will extract the feature edge 
    """
    boundaryEdges = vtk.vtkFeatureEdges()
    boundaryEdges.SetInputData(surface)
    boundaryEdges.BoundaryEdgesOn()
    boundaryEdges.FeatureEdgesOff()
    boundaryEdges.ManifoldEdgesOff()
    boundaryEdges.NonManifoldEdgesOff()
    boundaryEdges.Update()

    gamma_top = boundaryEdges.GetOutput()

    if debug:
        surface = apply_vtk_geom_filter(gamma_top)

        vtkWrite(surface, outdir + '/gamma_top.vtk')

    """
    separate the tv into tv tv-f and tv-f
    """
    norm_1 = get_normalized_cross_product(tv_center, svc_center, ivc_center)
    norm_2 = - norm_1

    plane = initialize_plane(norm_1[0], tv_center)

    plane2=initialize_plane(norm_2[0], tv_center)

    meshExtractFilter = vtk.vtkExtractGeometry()
    meshExtractFilter.SetInputData(tv)
    meshExtractFilter.SetImplicitFunction(plane)
    meshExtractFilter.Update()

    meshExtractFilter2 = vtk.vtkExtractGeometry()
    meshExtractFilter2.SetInputData(tv)
    meshExtractFilter2.ExtractBoundaryCellsOn()
    meshExtractFilter2.SetImplicitFunction(plane2)
    meshExtractFilter2.Update()

    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(meshExtractFilter.GetOutput())
    geo_filter.Update()
    tv_f = geo_filter.GetOutput()

    tv_f_ids = vtk_to_numpy(tv_f.GetPointData().GetArray("Ids"))
    fname = outdir + '/ids_TV_F.vtx'
    f = open(fname, 'w')
    f.write(f'{len(tv_f_ids)}\n')
    f.write('extra\n')
    for i in tv_f_ids:
        f.write(f'{i}\n')
    f.close()

    geo_filter2 = vtk.vtkGeometryFilter()
    geo_filter2.SetInputData(meshExtractFilter2.GetOutput())
    geo_filter2.Update()
    tv_s = geo_filter2.GetOutput()

    tv_s_ids = vtk_to_numpy(tv_s.GetPointData().GetArray("Ids"))
    fname = outdir + '/ids_TV_S.vtx'
    f = open(fname, 'w')
    f.write(f'{len(tv_s_ids)}\n')
    f.write('extra\n')
    for i in tv_s_ids:
        f.write(f'{i}\n')
    f.close()

    svc_points = svc.GetPoints().GetData()
    svc_points = vtk_to_numpy(svc_points)

    ivc_points = svc.GetPoints().GetData()  # Changed
    ivc_points = vtk_to_numpy(ivc_points)

    connect = vtk.vtkConnectivityFilter()
    connect.SetInputData(gamma_top)
    connect.SetExtractionModeToSpecifiedRegions()
    connect.Update()
    num = connect.GetNumberOfExtractedRegions()

    for i in range(num):
        connect.AddSpecifiedRegion(i)
        connect.Update()
        surface = connect.GetOutput()

        if debug:
            vtkWrite(surface, outdir + f'/gamma_top_{str(i)}.vtk')

        # Clean unused points
        cln = vtk.vtkCleanPolyData()
        cln.SetInputData(surface)
        cln.Update()
        surface = cln.GetOutput()
        points = surface.GetPoints().GetData()
        points = vtk_to_numpy(points)
        points = points.tolist()

        if debug:
            create_pts(points, f'/border_points_{str(i)}', outdir)

        in_ivc = False
        in_svc = False
        # if there is point of group i in both svc and ivc then it is the "top_endo+epi" we need
        while in_ivc == False and in_svc == False:
            for var in points:
                if var in ivc_points:
                    in_ivc = True
                if var in svc_points:
                    in_svc = True
            if in_ivc and in_svc:
                top_endo_id = i
                break
            else:
                top_endo_id = i  # comment this line if errors
                break

        # delete added region id
        connect.DeleteSpecifiedRegion(i)
        connect.Update()
    # It can happen that the first i=region(0) is the CS. Remove the -1 if that is the case
    connect.AddSpecifiedRegion(top_endo_id - 1)  # Find the id in the points dividing the RA, avoid CS
    connect.Update()
    surface = connect.GetOutput()

    # Clean unused points
    cln = vtk.vtkCleanPolyData()
    cln.SetInputData(surface)
    cln.Update()

    top_cut = cln.GetOutput()

    if debug:
        vtkWrite(top_cut, outdir + '/top_endo_epi.vtk')  # If this is the CS, then change top_endo_id in 877

    pts_in_top = vtk_to_numpy(top_cut.GetPointData().GetArray("Ids"))
    pts_in_svc = vtk_to_numpy(svc.GetPointData().GetArray("Ids"))
    pts_in_ivc = vtk_to_numpy(ivc.GetPointData().GetArray("Ids"))

    to_delete = np.zeros((len(pts_in_top),), dtype=int)

    for i in range(len(pts_in_top)):
        if pts_in_top[i] in pts_in_svc or pts_in_top[i] in pts_in_ivc:
            to_delete[i] = 1

    meshNew = dsa.WrapDataObject(top_cut)
    meshNew.PointData.append(to_delete, "delete")

    thresh = get_lower_threshold(meshNew.VTKObject, 0, "vtkDataObject::FIELD_ASSOCIATION_POINTS", "delete")

    geo_port, _geo_filter = get_vtk_geom_filter_port(thresh.GetOutputPort(), True)



    mv_id = vtk_to_numpy(top_cut.GetPointData().GetArray("Ids"))[0]

    connect = vtk.vtkConnectivityFilter()
    connect.SetInputData(geo_port)
    connect.SetExtractionModeToSpecifiedRegions()
    connect.Update()

    num = connect.GetNumberOfExtractedRegions()

    for i in range(num):
        connect.AddSpecifiedRegion(i)
        connect.Update()
        surface = connect.GetOutput()
        # Clean unused points
        cln = vtk.vtkCleanPolyData()
        cln.SetInputData(surface)
        cln.Update()
        surface = cln.GetOutput()

        pts_surf = vtk_to_numpy(surface.GetPointData().GetArray("Ids"))

        if mv_id not in pts_surf:
            found_id = i
            break

        # delete added region id
        connect.DeleteSpecifiedRegion(i)
        connect.Update()

    connect.AddSpecifiedRegion(found_id)
    connect.Update()
    surface = connect.GetOutput()

    # Clean unused points
    cln = vtk.vtkCleanPolyData()
    cln.SetInputData(surface)
    cln.Update()

    top_endo = vtk_to_numpy(cln.GetOutput().GetPointData().GetArray("Ids"))
    fname = outdir + '/ids_TOP_ENDO.vtx'
    f = open(fname, 'w')
    f.write(f'{len(top_endo)}\n')
    f.write('extra\n')
    for i in top_endo:
        f.write(f'{i}\n')
    f.close()


def create_pts(array_points, array_name, mesh_dir):
    f = open(f"{mesh_dir}{array_name}.pts", "w")
    f.write("0 0 0\n")
    for i in range(len(array_points)):
        f.write(f"{array_points[i][0]} {array_points[i][1]} {array_points[i][2]}\n")
    f.close()


def to_polydata(mesh):
    polydata = apply_vtk_geom_filter(mesh)

    return polydata


def vtkWrite(input_data, name):
    vtk_polydata_writer(name, input_data)


if __name__ == '__main__':
    run()
