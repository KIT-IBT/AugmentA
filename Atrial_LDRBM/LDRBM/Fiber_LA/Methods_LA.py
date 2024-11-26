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
import collections

import numpy as np
import vtk
from scipy.spatial import cKDTree
from scipy.spatial.distance import cosine
from vtk.numpy_interface import dataset_adapter as dsa

from vtk_opencarp_helper_methods.AugmentA_methods.vtk_operations import vtk_thr, get_normalized_cross_product
from vtk_opencarp_helper_methods.openCARP.exporting import write_to_pts, write_to_elem, write_to_lon
from vtk_opencarp_helper_methods.vtk_methods.converters import vtk_to_numpy, numpy_to_vtk
from vtk_opencarp_helper_methods.vtk_methods.exporting import vtk_polydata_writer, vtk_unstructured_grid_writer, \
    vtk_xml_unstructured_grid_writer
from vtk_opencarp_helper_methods.vtk_methods.filters import apply_vtk_geom_filter, get_vtk_geom_filter_port
from vtk_opencarp_helper_methods.vtk_methods.init_objects import initialize_plane
from vtk_opencarp_helper_methods.vtk_methods.reader import smart_reader
from vtk_opencarp_helper_methods.vtk_methods.thresholding import get_lower_threshold, get_upper_threshold, \
    get_threshold_between

vtk_version = vtk.vtkVersion.GetVTKSourceVersion().split()[-1].split('.')[0]


def mark_LA_endo_elemTag(model, tag, tao_mv, tao_lpv, tao_rpv, max_phie_ab_tau_lpv, max_phie_r2_tau_lpv):
    thresh = get_upper_threshold(model, tao_mv, "vtkDataObject::FIELD_ASSOCIATION_CELLS", "phie_r")

    MV_ids = vtk_to_numpy(thresh.GetOutput().GetCellData().GetArray('Global_ids'))

    thresh2 = get_upper_threshold(model, max_phie_r2_tau_lpv + 0.01, "vtkDataObject::FIELD_ASSOCIATION_CELLS",
                                  "phie_r2")

    thresh = get_lower_threshold(thresh2.GetOutputPort(), max_phie_ab_tau_lpv + 0.01,
                                 "vtkDataObject::FIELD_ASSOCIATION_CELLS", "phie_ab", source_is_input_connection=True)

    LAA_ids = vtk_to_numpy(thresh.GetOutput().GetCellData().GetArray('Global_ids'))

    thresh = get_lower_threshold(model, tao_lpv, "vtkDataObject::FIELD_ASSOCIATION_CELLS", "phie_v")

    LPV_ids = vtk_to_numpy(thresh.GetOutput().GetCellData().GetArray('Global_ids'))

    thresh = get_upper_threshold(model, tao_rpv, "vtkDataObject::FIELD_ASSOCIATION_CELLS", "phie_v")

    RPV_ids = vtk_to_numpy(thresh.GetOutput().GetCellData().GetArray('Global_ids'))

    meshNew = dsa.WrapDataObject(model)
    meshNew.CellData.append(tag, "elemTag")
    endo = meshNew.VTKObject


def move_surf_along_normals(mesh, eps, direction):
    polydata = apply_vtk_geom_filter(mesh)

    normalGenerator = vtk.vtkPolyDataNormals()
    normalGenerator.SetInputData(polydata)
    normalGenerator.ComputeCellNormalsOff()
    normalGenerator.ComputePointNormalsOn()
    normalGenerator.ConsistencyOn()
    normalGenerator.AutoOrientNormalsOff()
    normalGenerator.SplittingOff()
    normalGenerator.Update()

    PointNormalArray = vtk_to_numpy(normalGenerator.GetOutput().GetPointData().GetNormals())
    atrial_points = vtk_to_numpy(polydata.GetPoints().GetData())

    atrial_points = atrial_points + eps * direction * PointNormalArray

    vtkPts = vtk.vtkPoints()
    vtkPts.SetData(numpy_to_vtk(atrial_points))
    polydata.SetPoints(vtkPts)

    mesh = vtk.vtkUnstructuredGrid()
    mesh.DeepCopy(polydata)

    return mesh


def generate_bilayer(endo, epi):
    geo_port, _geo_filter = get_vtk_geom_filter_port(epi)
    reverse = vtk.vtkReverseSense()
    reverse.ReverseCellsOn()
    reverse.ReverseNormalsOn()
    reverse.SetInputConnection(geo_port)
    reverse.Update()

    epi = vtk.vtkUnstructuredGrid()
    epi.DeepCopy(reverse.GetOutput())

    endo_pts = vtk_to_numpy(endo.GetPoints().GetData())
    epi_pts = vtk_to_numpy(epi.GetPoints().GetData())

    tree = cKDTree(endo_pts)
    dd, ii = tree.query(epi_pts)

    lines = vtk.vtkCellArray()

    for i in range(len(endo_pts)):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i);
        line.GetPointIds().SetId(1, len(endo_pts) + ii[i]);
        lines.InsertNextCell(line)

    points = np.concatenate((endo_pts, epi_pts[ii, :]), axis=0)
    polydata = vtk.vtkUnstructuredGrid()
    vtkPts = vtk.vtkPoints()
    vtkPts.SetData(numpy_to_vtk(points))
    polydata.SetPoints(vtkPts)
    polydata.SetCells(3, lines)

    fibers = np.zeros((len(endo_pts), 3), dtype="float32")
    fibers[:, 0] = 1

    tag = np.ones((len(endo_pts), 1), dtype=int)
    tag[:, ] = 100

    meshNew = dsa.WrapDataObject(polydata)
    meshNew.CellData.append(tag, "elemTag")
    meshNew.CellData.append(fibers, "fiber")
    fibers = np.zeros((len(endo_pts), 3), dtype="float32")
    fibers[:, 1] = 1
    meshNew.CellData.append(fibers, "sheet")

    appendFilter = vtk.vtkAppendFilter()
    appendFilter.AddInputData(endo)
    appendFilter.AddInputData(epi)
    appendFilter.AddInputData(meshNew.VTKObject)
    appendFilter.MergePointsOn()
    appendFilter.Update()

    bilayer = appendFilter.GetOutput()

    return bilayer


def write_bilayer(bilayer, args, job):
    file_name = job.ID + "/result_LA/LA_bilayer_with_fiber"
    if args.ofmt == 'vtk':
        vtk_unstructured_grid_writer(f"{file_name}.vtk", bilayer, store_binary=True)
    else:
        vtk_xml_unstructured_grid_writer(f"{file_name}.vtu", bilayer)

    pts = vtk_to_numpy(bilayer.GetPoints().GetData())
    tag_epi = vtk_to_numpy(bilayer.GetCellData().GetArray('elemTag'))
    el_epi = vtk_to_numpy(bilayer.GetCellData().GetArray('fiber'))
    sheet_epi = vtk_to_numpy(bilayer.GetCellData().GetArray('sheet'))

    write_to_pts(f'{file_name}.pts', pts)
    write_to_elem(f'{file_name}.elem', bilayer, tag_epi)
    write_to_lon(f'{file_name}.lon', el_epi, sheet_epi)
    print('Done..')


def creat_tube_around_spline(points_data, radius):
    # Creat a points set
    spline_points = vtk.vtkPoints()
    for i in range(len(points_data)):
        spline_points.InsertPoint(i, points_data[i][0], points_data[i][1], points_data[i][2])

    # Fit a spline to the points
    spline = vtk.vtkParametricSpline()
    spline.SetPoints(spline_points)

    functionSource = vtk.vtkParametricFunctionSource()
    functionSource.SetParametricFunction(spline)
    functionSource.SetUResolution(10 * spline_points.GetNumberOfPoints())
    functionSource.Update()

    # Interpolate the scalars
    interpolatedRadius = vtk.vtkTupleInterpolator()
    interpolatedRadius.SetInterpolationTypeToLinear()
    interpolatedRadius.SetNumberOfComponents(1)

    # Generate the radius scalars
    tubeRadius = vtk.vtkDoubleArray()
    n = functionSource.GetOutput().GetNumberOfPoints()
    tubeRadius.SetNumberOfTuples(n)
    tubeRadius.SetName("TubeRadius")

    # TODO make the radius variable???
    tMin = interpolatedRadius.GetMinimumT()
    tMax = interpolatedRadius.GetMaximumT()
    for i in range(n):
        t = (tMax - tMin) / (n - 1) * i + tMin
        r = radius
        # interpolatedRadius.InterpolateTuple(t, r)
        tubeRadius.SetTuple1(i, r)

    # Add the scalars to the polydata
    tubePolyData = functionSource.GetOutput()
    tubePolyData.GetPointData().AddArray(tubeRadius)
    tubePolyData.GetPointData().SetActiveScalars("TubeRadius")

    # Create the tubes TODO: SidesShareVerticesOn()???
    tuber = vtk.vtkTubeFilter()
    tuber.SetInputData(tubePolyData)
    tuber.SetNumberOfSides(20)
    tuber.SidesShareVerticesOn()
    tuber.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
    tuber.SetCapping(0)
    tuber.Update()

    triangle = vtk.vtkTriangleFilter()
    triangle.SetInputData(tuber.GetOutput())
    triangle.Update()

    tuber = triangle
    return tuber


def dijkstra_path(polydata, StartVertex, EndVertex):
    path = vtk.vtkDijkstraGraphGeodesicPath()
    path.SetInputData(polydata)
    # attention the return value will be reversed
    path.SetStartVertex(EndVertex)
    path.SetEndVertex(StartVertex)
    path.Update()
    points_data = path.GetOutput().GetPoints().GetData()
    points_data = vtk_to_numpy(points_data)
    return points_data


def dijkstra_path_on_a_plane(polydata, StartVertex, EndVertex, plane_point):
    point_start = np.asarray(polydata.GetPoint(StartVertex))
    point_end = np.asarray(polydata.GetPoint(EndVertex))
    point_third = plane_point

    norm_1 = get_normalized_cross_product(point_start, point_end, point_third)

    plane = initialize_plane(norm_1[0], point_start)

    meshExtractFilter1 = vtk.vtkExtractGeometry()
    meshExtractFilter1.SetInputData(polydata)
    meshExtractFilter1.SetImplicitFunction(plane)
    meshExtractFilter1.Update()

    point_moved = point_start - 1.5 * norm_1

    plane2 = initialize_plane(-norm_1[0], point_moved[0])

    meshExtractFilter2 = vtk.vtkExtractGeometry()
    meshExtractFilter2.SetInputData(meshExtractFilter1.GetOutput())
    meshExtractFilter2.SetImplicitFunction(plane2)
    meshExtractFilter2.Update()
    band = meshExtractFilter2.GetOutput()
    band = apply_vtk_geom_filter(band)

    # print(band)
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(band)
    loc.BuildLocator()
    StartVertex = loc.FindClosestPoint(point_start)
    EndVertex = loc.FindClosestPoint(point_end)

    points_data = dijkstra_path(band, StartVertex, EndVertex)
    return points_data


def creat_sphere(center, radius):
    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(center[0], center[1], center[2])
    sphere.SetThetaResolution(40)
    sphere.SetPhiResolution(40)
    sphere.SetRadius(radius)
    sphere.Update()
    return sphere


def creat_tube(center1, center2, radius):
    line = vtk.vtkLineSource()
    line.SetPoint1(center1[0], center1[1], center1[2])
    line.SetPoint2(center2[0], center2[1], center2[2])
    line.Update()

    tube = vtk.vtkTubeFilter()
    tube.SetInputData(line.GetOutput())
    tube.SetRadius(radius)
    tube.SetNumberOfSides(20)
    tube.Update()
    return tube


def get_element_ids_around_path_within_radius(mesh, points_data, radius):
    gl_ids = vtk_to_numpy(mesh.GetCellData().GetArray('Global_ids'))

    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()

    mesh_id_list = vtk.vtkIdList()
    for i in range(len(points_data)):
        temp_result = vtk.vtkIdList()
        locator.FindPointsWithinRadius(radius, points_data[i], temp_result)
        for j in range(temp_result.GetNumberOfIds()):
            mesh_id_list.InsertNextId(temp_result.GetId(j))

    mesh_cell_id_list = vtk.vtkIdList()
    mesh_cell_temp_id_list = vtk.vtkIdList()
    for i in range(mesh_id_list.GetNumberOfIds()):
        mesh.GetPointCells(mesh_id_list.GetId(i), mesh_cell_temp_id_list)
        for j in range(mesh_cell_temp_id_list.GetNumberOfIds()):
            mesh_cell_id_list.InsertNextId(mesh_cell_temp_id_list.GetId(j))

    ids = []

    for i in range(mesh_cell_id_list.GetNumberOfIds()):
        index = mesh_cell_id_list.GetId(i)
        ids.append(gl_ids[index])

    return ids


def assign_element_tag_around_path_within_radius(mesh, points_data, radius, tag, element_tag):
    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()

    mesh_id_list = vtk.vtkIdList()
    for i in range(len(points_data)):
        temp_result = vtk.vtkIdList()
        locator.FindPointsWithinRadius(radius, points_data[i], temp_result)
        for j in range(temp_result.GetNumberOfIds()):
            mesh_id_list.InsertNextId(temp_result.GetId(j))

    mesh_cell_id_list = vtk.vtkIdList()
    mesh_cell_temp_id_list = vtk.vtkIdList()
    for i in range(mesh_id_list.GetNumberOfIds()):
        mesh.GetPointCells(mesh_id_list.GetId(i), mesh_cell_temp_id_list)
        for j in range(mesh_cell_temp_id_list.GetNumberOfIds()):
            mesh_cell_id_list.InsertNextId(mesh_cell_temp_id_list.GetId(j))

    for i in range(mesh_cell_id_list.GetNumberOfIds()):
        index = mesh_cell_id_list.GetId(i)
        tag[index] = element_tag

    return tag


def normalize_vector(vector):
    abs = np.linalg.norm(vector)
    if abs != 0:
        vector_norm = vector / abs
    else:
        vector_norm = vector

    return vector_norm


def assign_element_fiber_around_path_within_radius(mesh, points_data, radius, fiber, smooth=True):
    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()
    if smooth:
        for i in range(len(points_data)):
            if i % 5 == 0 and i < 5:
                vector = points_data[5] - points_data[0]
            else:
                vector = points_data[i] - points_data[i - 5]
            vector = normalize_vector(vector)
            mesh_point_temp_id_list = vtk.vtkIdList()
            locator.FindPointsWithinRadius(radius, points_data[i], mesh_point_temp_id_list)
            mesh_cell_temp_id_list = vtk.vtkIdList()
            mesh_cell_id_list = vtk.vtkIdList()
            for j in range(mesh_point_temp_id_list.GetNumberOfIds()):
                mesh.GetPointCells(mesh_point_temp_id_list.GetId(j), mesh_cell_temp_id_list)
                for h in range(mesh_cell_temp_id_list.GetNumberOfIds()):
                    mesh_cell_id_list.InsertNextId(mesh_cell_temp_id_list.GetId(h))

            for k in range(mesh_cell_id_list.GetNumberOfIds()):
                index = mesh_cell_id_list.GetId(k)
                fiber[index] = vector
    else:
        for i in range(len(points_data)):
            if i < 1:
                vector = points_data[1] - points_data[0]
            else:
                vector = points_data[i] - points_data[i - 1]
            vector = normalize_vector(vector)
            mesh_point_temp_id_list = vtk.vtkIdList()
            locator.FindPointsWithinRadius(radius, points_data[i], mesh_point_temp_id_list)
            mesh_cell_temp_id_list = vtk.vtkIdList()
            mesh_cell_id_list = vtk.vtkIdList()
            for j in range(mesh_point_temp_id_list.GetNumberOfIds()):
                mesh.GetPointCells(mesh_point_temp_id_list.GetId(j), mesh_cell_temp_id_list)
                for h in range(mesh_cell_temp_id_list.GetNumberOfIds()):
                    mesh_cell_id_list.InsertNextId(mesh_cell_temp_id_list.GetId(h))

            for k in range(mesh_cell_id_list.GetNumberOfIds()):
                index = mesh_cell_id_list.GetId(k)
                fiber[index] = vector
    return fiber


def get_mean_point(data):
    ring_points = data.GetPoints().GetData()
    ring_points = vtk_to_numpy(ring_points)
    center_point = [np.mean(ring_points[:, 0]), np.mean(ring_points[:, 1]), np.mean(ring_points[:, 2])]
    center_point = np.array(center_point)
    return center_point


def multidim_intersect(arr1, arr2):
    arr1_view = arr1.view([('', arr1.dtype)] * arr1.shape[1])
    arr2_view = arr2.view([('', arr2.dtype)] * arr2.shape[1])
    intersected = np.intersect1d(arr1_view, arr2_view)
    return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])


def multidim_intersect_bool(arr1, arr2):
    arr1_view = arr1.view([('', arr1.dtype)] * arr1.shape[1])
    arr2_view = arr2.view([('', arr2.dtype)] * arr2.shape[1])
    intersected = np.intersect1d(arr1_view, arr2_view)
    if len(intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])) == 0:
        res = 0
    else:
        res = 1
    return res


def get_ct_end_points_id(endo, ct, scv, icv):
    # endo
    points_data = endo.GetPoints().GetData()
    endo_points = vtk_to_numpy(points_data)

    # ct
    points_data = ct.GetPoints().GetData()
    ct_points = vtk_to_numpy(points_data)

    # scv
    points_data = scv.GetPoints().GetData()
    scv_points = vtk_to_numpy(points_data)

    # icv
    points_data = icv.GetPoints().GetData()
    icv_points = vtk_to_numpy(points_data)

    # intersection
    # inter_ct_endo = multidim_intersect(endo_points, ct_points)
    # inter_icv = multidim_intersect(inter_ct_endo, icv_points)
    # inter_scv = multidim_intersect(inter_ct_endo, scv_points)`
    inter_icv = multidim_intersect(ct_points, icv_points)
    inter_scv = multidim_intersect(ct_points, scv_points)

    # calculating mean point
    path_icv = np.asarray([np.mean(inter_icv[:, 0]), np.mean(inter_icv[:, 1]), np.mean(inter_icv[:, 2])])
    path_scv = np.asarray([np.mean(inter_scv[:, 0]), np.mean(inter_scv[:, 1]), np.mean(inter_scv[:, 2])])

    loc = vtk.vtkPointLocator()
    loc.SetDataSet(endo)
    loc.BuildLocator()

    path_ct_id_icv = loc.FindClosestPoint(path_icv)
    path_ct_id_scv = loc.FindClosestPoint(path_scv)

    return path_ct_id_icv, path_ct_id_scv


def get_tv_end_points_id(endo, ra_tv_s_surface, ra_ivc_surface, ra_svc_surface, ra_tv_surface):
    tv_center = get_mean_point(ra_tv_surface)
    tv_ivc_center = get_mean_point(ra_ivc_surface)
    tv_svc_center = get_mean_point(ra_svc_surface)

    norm_1 = get_normalized_cross_product(tv_center, tv_ivc_center, tv_svc_center)
    moved_center = tv_center - norm_1 * 5

    plane = initialize_plane(-norm_1[0], moved_center[0])

    meshExtractFilter = vtk.vtkExtractGeometry()
    meshExtractFilter.SetInputData(ra_tv_s_surface)
    meshExtractFilter.SetImplicitFunction(plane)
    meshExtractFilter.Update()

    connect = vtk.vtkConnectivityFilter()
    connect.SetInputData(meshExtractFilter.GetOutput())
    connect.SetExtractionModeToAllRegions()
    connect.Update()
    connect.SetExtractionModeToSpecifiedRegions()
    connect.AddSpecifiedRegion(1)
    connect.Update()

    # Clean unused points
    surface = apply_vtk_geom_filter(connect.GetOutput())

    cln = vtk.vtkCleanPolyData()
    cln.SetInputData(surface)
    cln.Update()
    points_data = cln.GetOutput().GetPoints().GetData()
    ring = vtk_to_numpy(points_data)
    center_point_1 = np.asarray([np.mean(ring[:, 0]), np.mean(ring[:, 1]), np.mean(ring[:, 2])])

    connect.DeleteSpecifiedRegion(1)
    connect.AddSpecifiedRegion(0)
    connect.Update()

    # Clean unused points
    surface = apply_vtk_geom_filter(connect.GetOutput())

    cln = vtk.vtkCleanPolyData()
    cln.SetInputData(surface)
    cln.Update()
    points_data = cln.GetOutput().GetPoints().GetData()
    ring = vtk_to_numpy(points_data)
    center_point_2 = np.asarray([np.mean(ring[:, 0]), np.mean(ring[:, 1]), np.mean(ring[:, 2])])
    dis_1 = np.linalg.norm(center_point_1 - tv_ivc_center)
    dis_2 = np.linalg.norm(center_point_1 - tv_svc_center)
    # print(dis_1)
    # print(dis_2)
    if dis_1 < dis_2:
        center_point_icv = center_point_1
        center_point_scv = center_point_2
    else:
        center_point_icv = center_point_2
        center_point_scv = center_point_1
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(endo)
    loc.BuildLocator()

    path_tv_id_icv = loc.FindClosestPoint(center_point_icv)
    path_tv_id_scv = loc.FindClosestPoint(center_point_scv)

    return path_tv_id_icv, path_tv_id_scv


def extract_largest_region(mesh):
    connect = vtk.vtkConnectivityFilter()
    connect.SetInputData(mesh)
    connect.SetExtractionModeToLargestRegion()
    connect.Update()
    surface = connect.GetOutput()

    surface = apply_vtk_geom_filter(surface)

    cln = vtk.vtkCleanPolyData()
    cln.SetInputData(surface)
    cln.Update()
    res = cln.GetOutput()

    return res


def assign_ra_appendage(model, SCV, appex_point, tag):
    appex_point = np.asarray(appex_point)
    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(model)
    locator.BuildLocator()

    locator2 = vtk.vtkStaticPointLocator()
    locator2.SetDataSet(SCV)
    locator2.BuildLocator()
    SCV_id = locator2.FindClosestPoint(appex_point)
    SCV_closed_point = SCV.GetPoint(SCV_id)
    radius = np.linalg.norm(appex_point - SCV_closed_point)
    print(radius)

    mesh_point_temp_id_list = vtk.vtkIdList()
    locator.FindPointsWithinRadius(radius, appex_point, mesh_point_temp_id_list)
    print(mesh_point_temp_id_list.GetNumberOfIds())
    mesh_cell_id_list = vtk.vtkIdList()
    mesh_cell_temp_id_list = vtk.vtkIdList()
    for i in range(mesh_point_temp_id_list.GetNumberOfIds()):
        model.GetPointCells(mesh_point_temp_id_list.GetId(i), mesh_cell_temp_id_list)
        for j in range(mesh_cell_temp_id_list.GetNumberOfIds()):
            mesh_cell_id_list.InsertNextId(mesh_cell_temp_id_list.GetId(j))

    for i in range(mesh_cell_id_list.GetNumberOfIds()):
        index = mesh_cell_id_list.GetId(i)
        tag[index] = 59

    return tag


def get_endo_ct_intersection_cells(endo, ct):
    points_data = ct.GetPoints().GetData()
    ct_points = vtk_to_numpy(points_data)

    points_data = endo.GetPoints().GetData()
    endo_points = vtk_to_numpy(points_data)

    intersection = multidim_intersect(ct_points, endo_points)

    loc = vtk.vtkPointLocator()
    loc.SetDataSet(endo)
    loc.BuildLocator()

    endo_id_list = []
    for i in range(len(intersection)):
        endo_id_list.append(loc.FindClosestPoint(intersection[i]))
    endo_cell_id_list = vtk.vtkIdList()
    endo_cell_temp_id_list = vtk.vtkIdList()
    for i in range(len(endo_id_list)):
        endo.GetPointCells(endo_id_list[i], endo_cell_temp_id_list)
        for j in range(endo_cell_temp_id_list.GetNumberOfIds()):
            endo_cell_id_list.InsertNextId(endo_cell_temp_id_list.GetId(j))
    print(endo_cell_id_list.GetNumberOfIds())
    extract = vtk.vtkExtractCells()
    extract.SetInputData(endo)
    extract.SetCellList(endo_cell_id_list)
    extract.Update()
    endo_ct = extract.GetOutput()

    return endo_ct


def get_connection_point_la_and_ra(appen_point):
    la_mv_surface = smart_reader('../../Generate_Boundaries/LA/result/la_mv_surface.vtk')
    la_rpv_inf_surface = smart_reader('../../Generate_Boundaries/LA/result/la_rpv_inf_surface.vtk')
    la_lpv_inf_surface = smart_reader('../../Generate_Boundaries/LA/result/la_lpv_inf_surface.vtk')
    endo = smart_reader('../../Generate_Boundaries/LA/result/la_endo_surface.vtk')
    la_epi_surface = smart_reader('../../Generate_Boundaries/LA/result/la_epi_surface.vtk')
    ra_epi_surface = smart_reader('../../Generate_Boundaries/RA/result/ra_epi_surface.vtk')

    loc_mv = vtk.vtkPointLocator()
    loc_mv.SetDataSet(la_mv_surface)
    loc_mv.BuildLocator()

    point_1_id = loc_mv.FindClosestPoint(appen_point)
    point_1 = la_mv_surface.GetPoint(point_1_id)

    loc_rpv_inf = vtk.vtkPointLocator()
    loc_rpv_inf.SetDataSet(la_rpv_inf_surface)
    loc_rpv_inf.BuildLocator()
    point_2_id = loc_rpv_inf.FindClosestPoint(appen_point)
    point_2 = la_rpv_inf_surface.GetPoint(point_2_id)

    loc_endo = vtk.vtkPointLocator()
    loc_endo.SetDataSet(endo)
    loc_endo.BuildLocator()

    point_1_id_endo = loc_endo.FindClosestPoint(point_1)
    point_2_id_endo = loc_endo.FindClosestPoint(point_2)

    bb_aux_l_points = dijkstra_path(apply_vtk_geom_filter(endo), point_1_id_endo, point_2_id_endo)
    length = len(bb_aux_l_points)
    la_connect_point = bb_aux_l_points[int(length * 0.5)]

    # ra
    la_epi_surface = apply_vtk_geom_filter(la_epi_surface)

    ra_epi_surface = apply_vtk_geom_filter(ra_epi_surface)

    loc_la_epi = vtk.vtkPointLocator()
    loc_la_epi.SetDataSet(la_epi_surface)
    loc_la_epi.BuildLocator()

    loc_ra_epi = vtk.vtkPointLocator()
    loc_ra_epi.SetDataSet(ra_epi_surface)
    loc_ra_epi.BuildLocator()

    la_connect_point_id = loc_la_epi.FindClosestPoint(la_connect_point)
    la_connect_point = la_epi_surface.GetPoint(la_connect_point_id)

    ra_connect_point_id = loc_ra_epi.FindClosestPoint(la_connect_point)
    ra_connect_point = ra_epi_surface.GetPoint(ra_connect_point_id)

    return la_connect_point, ra_connect_point


def point_array_mapper(mesh1, mesh2, mesh2_name, idat):
    pts1 = vtk_to_numpy(mesh1.GetPoints().GetData())
    pts2 = vtk_to_numpy(mesh2.GetPoints().GetData())

    tree = cKDTree(pts1)

    dd, ii = tree.query(pts2)  # n_jobs=-1)

    meshNew = dsa.WrapDataObject(mesh2)
    if idat == "all":
        for i in range(mesh1.GetPointData().GetNumberOfArrays()):
            data = vtk_to_numpy(
                mesh1.GetPointData().GetArray(mesh1.GetPointData().GetArrayName(i)))
            if isinstance(data[0], collections.abc.Sized):
                data2 = np.zeros((len(pts2), len(data[0])), dtype=data.dtype)
            else:
                data2 = np.zeros((len(pts2),), dtype=data.dtype)

            data2 = data[ii]
            data2 = np.where(np.isnan(data2), 10000, data2)
            # ghosts = np.zeros(meshNew.GetNumberOfPoints(), dtype=np.uint8)
            # ghosts[1] = vtk.vtkDataSetAttributes.DUPLICATEPOINT
            # meshNew.PointData.append(ghosts, vtk.vtkDataSetAttributes.GhostArrayName())
            # assert algs.make_point_mask_from_NaNs(meshNew, data2)[1] == vtk.vtkDataSetAttributes.DUPLICATEPOINT | vtk.vtkDataSetAttributes.HIDDENPOINT
            meshNew.PointData.append(data2, mesh1.GetPointData().GetArrayName(i))
    else:
        data = vtk_to_numpy(mesh1.GetPointData().GetArray(idat))
        if isinstance(data[0], collections.abc.Sized):
            data2 = np.zeros((len(pts2), len(data[0])), dtype=data.dtype)
        else:
            data2 = np.zeros((len(pts2),), dtype=data.dtype)

        data2 = data[ii]
        meshNew.PointData.append(data2, idat)

    vtk_unstructured_grid_writer(f"{mesh2_name.split('.vtk')[0]}_with_data.vtk", meshNew.VTKObject, store_binary=True)
    return meshNew.VTKObject


def cell_array_mapper(mesh1, mesh2, mesh2_name, idat):
    filter_cell_centers = vtk.vtkCellCenters()
    filter_cell_centers.SetInputData(mesh1)
    filter_cell_centers.Update()
    centroids1 = filter_cell_centers.GetOutput().GetPoints()
    centroids1_array = vtk_to_numpy(centroids1.GetData())

    filter_cell_centers = vtk.vtkCellCenters()
    filter_cell_centers.SetInputData(mesh2)
    filter_cell_centers.Update()
    centroids2 = filter_cell_centers.GetOutput().GetPoints()
    pts2 = vtk_to_numpy(centroids2.GetData())

    tree = cKDTree(centroids1_array)

    dd, ii = tree.query(pts2)  # , n_jobs=-1)

    meshNew = dsa.WrapDataObject(mesh2)
    if idat == "all":
        for i in range(mesh1.GetCellData().GetNumberOfArrays()):
            data = vtk_to_numpy(
                mesh1.GetCellData().GetArray(mesh1.GetCellData().GetArrayName(i)))
            if isinstance(data[0], collections.abc.Sized):
                data2 = np.zeros((len(pts2), len(data[0])), dtype=data.dtype)
            else:
                data2 = np.zeros((len(pts2),), dtype=data.dtype)

            data2 = data[ii]
            meshNew.PointData.append(data2, mesh1.GetCellData().GetArrayName(i))
    else:
        data = vtk_to_numpy(mesh1.GetCellData().GetArray(idat))
        if isinstance(data[0], collections.abc.Sized):
            data2 = np.zeros((len(pts2), len(data[0])), dtype=data.dtype)
        else:
            data2 = np.zeros((len(pts2),), dtype=data.dtype)

        data2 = data[ii]
        meshNew.CellData.append(data2, idat)

    return meshNew.VTKObject


def get_bachmann_path_left(appendage_basis, lpv_sup_basis):
    la_mv_surface = smart_reader('../../Generate_Boundaries/LA/result/la_mv_surface.vtk')
    la_lpv_inf_surface = smart_reader('../../Generate_Boundaries/LA/result/la_lpv_inf_surface.vtk')
    endo = smart_reader('../../Generate_Boundaries/LA/result/la_endo_surface.vtk')
    epi = smart_reader('../../Generate_Boundaries/LA/result/la_epi_surface.vtk')

    loc_mv = vtk.vtkPointLocator()
    loc_mv.SetDataSet(la_mv_surface)
    loc_mv.BuildLocator()

    loc_epi = vtk.vtkPointLocator()
    loc_epi.SetDataSet(epi)
    loc_epi.BuildLocator()

    appendage_basis_id = loc_epi.FindClosestPoint(appendage_basis)
    lpv_sup_basis_id = loc_epi.FindClosestPoint(lpv_sup_basis)

    left_inf_pv_center = get_mean_point(la_lpv_inf_surface)
    point_l1_id = loc_mv.FindClosestPoint(left_inf_pv_center)
    point_l1 = la_mv_surface.GetPoint(point_l1_id)
    bb_mv_id = loc_epi.FindClosestPoint(point_l1)

    bb_1_points = dijkstra_path(apply_vtk_geom_filter(epi), lpv_sup_basis_id, appendage_basis_id)
    bb_2_points = dijkstra_path(apply_vtk_geom_filter(epi), appendage_basis_id, bb_mv_id)
    np.delete(bb_1_points, -1)
    bb_left = np.concatenate((bb_1_points, bb_2_points), axis=0)

    return bb_left, appendage_basis


def compute_wide_BB_path_left(epi, df, left_atrial_appendage_epi, mitral_valve_epi):
    # Extract the LAA
    thresh = get_threshold_between(epi, left_atrial_appendage_epi, left_atrial_appendage_epi,
                                   "vtkDataObject::FIELD_ASSOCIATION_CELLS", "elemTag")

    LAA = thresh.GetOutput()

    min_r2_cell_LAA = np.argmin(vtk_to_numpy(LAA.GetCellData().GetArray('phie_r2')))

    ptIds = vtk.vtkIdList()

    LAA.GetCellPoints(min_r2_cell_LAA, ptIds)
    # sup_appendage_basis_id = int(LAA.GetPointData().GetArray('Global_ids').GetTuple(ptIds.GetId(0))[0])
    sup_appendage_basis = LAA.GetPoint(ptIds.GetId(0))

    max_r2_cell_LAA = np.argmax(vtk_to_numpy(LAA.GetCellData().GetArray('phie_r2')))

    ptIds = vtk.vtkIdList()

    LAA.GetCellPoints(max_r2_cell_LAA, ptIds)
    # bb_mv_id = int(LAA.GetPointData().GetArray('Global_ids').GetTuple(ptIds.GetId(0))[0])
    bb_mv_laa = LAA.GetPoint(ptIds.GetId(0))

    max_v_cell_LAA = np.argmax(vtk_to_numpy(LAA.GetCellData().GetArray('phie_v')))

    ptIds = vtk.vtkIdList()

    LAA.GetCellPoints(max_v_cell_LAA, ptIds)

    # inf_appendage_basis_id = int(LAA.GetPointData().GetArray('Global_ids').GetTuple(ptIds.GetId(0))[0])

    inf_appendage_basis = LAA.GetPoint(ptIds.GetId(0))

    # Extract the border of the LAA

    boundaryEdges = vtk.vtkFeatureEdges()
    boundaryEdges.SetInputData(apply_vtk_geom_filter(thresh.GetOutputPort(), True))
    boundaryEdges.BoundaryEdgesOn()
    boundaryEdges.FeatureEdgesOff()
    boundaryEdges.ManifoldEdgesOff()
    boundaryEdges.NonManifoldEdgesOff()
    boundaryEdges.Update()

    LAA_border = boundaryEdges.GetOutput()
    LAA_pts_border = vtk_to_numpy(LAA_border.GetPoints().GetData())
    max_dist = 0
    for i in range(len(LAA_pts_border)):
        if np.sqrt(np.sum((LAA_pts_border[i] - df["LIPV"].to_numpy()) ** 2, axis=0)) > max_dist:
            max_dist = np.sqrt(np.sum((LAA_pts_border[i] - df["LIPV"].to_numpy()) ** 2, axis=0))
            LAA_pt_far_from_LIPV = LAA_pts_border[i]

    # Extract the MV
    thresh = get_threshold_between(epi, mitral_valve_epi, mitral_valve_epi, "vtkDataObject::FIELD_ASSOCIATION_CELLS",
                                   "elemTag")

    MV = thresh.GetOutput()

    # Get the closest point to the inferior appendage base in the MV
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(MV)
    loc.BuildLocator()

    bb_mv = MV.GetPoint(loc.FindClosestPoint(bb_mv_laa))
    thresh = get_threshold_between(epi, left_atrial_appendage_epi, left_atrial_appendage_epi,
                                   "vtkDataObject::FIELD_ASSOCIATION_CELLS", "elemTag")

    loc = vtk.vtkPointLocator()
    loc.SetDataSet(thresh.GetOutput())
    # loc.SetDataSet(epi)
    loc.BuildLocator()
    LAA_pt_far_from_LIPV_id = loc.FindClosestPoint(LAA_pt_far_from_LIPV)
    inf_appendage_basis_id = loc.FindClosestPoint(inf_appendage_basis)
    sup_appendage_basis_id = loc.FindClosestPoint(sup_appendage_basis)
    bb_mv_id = loc.FindClosestPoint(bb_mv)

    bb_left = get_wide_bachmann_path_left(thresh.GetOutput(), inf_appendage_basis_id, sup_appendage_basis_id, bb_mv_id,
                                          LAA_pt_far_from_LIPV_id)

    return bb_left, thresh.GetOutput().GetPoint(inf_appendage_basis_id), thresh.GetOutput().GetPoint(
        sup_appendage_basis_id), thresh.GetOutput().GetPoint(LAA_pt_far_from_LIPV_id)


def get_in_surf1_closest_point_in_surf2(surf1, surf2, pt_id_in_surf2):
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(surf1)
    loc.BuildLocator()
    return loc.FindClosestPoint(surf2.GetPoint(pt_id_in_surf2))


def get_wide_bachmann_path_left(epi, inf_appendage_basis_id, sup_appendage_basis_id, bb_mv_id, LAA_pt_far_from_LIPV_id):
    poly_mesh_epi = apply_vtk_geom_filter(epi)

    bb_1_points = dijkstra_path(poly_mesh_epi, sup_appendage_basis_id, LAA_pt_far_from_LIPV_id)
    bb_2_points = dijkstra_path(poly_mesh_epi, LAA_pt_far_from_LIPV_id, inf_appendage_basis_id)
    bb_3_points = dijkstra_path(poly_mesh_epi, inf_appendage_basis_id, bb_mv_id)

    np.delete(bb_1_points, -1)
    bb_left = np.concatenate((bb_1_points, bb_2_points, bb_3_points), axis=0)
    # bb_left = dijkstra_path(geo_filter.GetOutput(), sup_appendage_basis_id, bb_mv_id)

    return bb_left


def creat_center_line(start_end_point):
    spline_points = vtk.vtkPoints()
    for i in range(len(start_end_point)):
        spline_points.InsertPoint(i, start_end_point[i][0], start_end_point[i][1], start_end_point[i][2])

    # Fit a spline to the points
    spline = vtk.vtkParametricSpline()
    spline.SetPoints(spline_points)

    functionSource = vtk.vtkParametricFunctionSource()
    functionSource.SetParametricFunction(spline)
    functionSource.SetUResolution(30 * spline_points.GetNumberOfPoints())
    functionSource.Update()
    tubePolyData = functionSource.GetOutput()
    points = tubePolyData.GetPoints().GetData()
    points = vtk_to_numpy(points)

    return points


def smart_bridge_writer(tube, sphere_1, sphere_2, name, job):
    meshNew = dsa.WrapDataObject(tube.GetOutput())
    vtk_polydata_writer(job.ID + "/bridges/" + str(name) + "_tube.vtk", meshNew.VTKObject)
    meshNew = dsa.WrapDataObject(sphere_1.GetOutput())
    vtk_polydata_writer(job.ID + "/bridges/" + str(name) + "_sphere_1.vtk", meshNew.VTKObject)
    meshNew = dsa.WrapDataObject(sphere_2.GetOutput())
    vtk_polydata_writer(job.ID + "/bridges/" + str(name) + "_sphere_2.vtk", meshNew.VTKObject)


def find_tau(model, ub, lb, low_up, scalar):
    k = 1
    while ub - lb > 0.01:
        if low_up == "low":
            thresh = get_lower_threshold(model, (ub + lb) / 2, "vtkDataObject::FIELD_ASSOCIATION_CELLS", scalar)
        else:
            thresh = get_upper_threshold(model, (ub + lb) / 2, "vtkDataObject::FIELD_ASSOCIATION_CELLS", scalar)

        connect = vtk.vtkConnectivityFilter()
        connect.SetInputData(thresh.GetOutput())
        connect.SetExtractionModeToAllRegions()
        connect.Update()
        num = connect.GetNumberOfExtractedRegions()

        print("Iteration: ", k)
        print("Value of tao: ", (ub + lb) / 2)
        print("Number of regions: ", num, "\n")

        if low_up == "low":
            if num == 1:
                ub = (ub + lb) / 2
            elif num > 1:
                lb = (ub + lb) / 2
        else:
            if num == 1:
                lb = (ub + lb) / 2
            elif num > 1:
                ub = (ub + lb) / 2

        k += 1

    if low_up == "low":
        return lb
    else:
        return ub


def distinguish_PVs(connect, PVs, df, name1, name2):
    num = connect.GetNumberOfExtractedRegions()
    connect.SetExtractionModeToSpecifiedRegions()

    centroid1 = df[name1].to_numpy()
    centroid2 = df[name2].to_numpy()

    for i in range(num):
        connect.AddSpecifiedRegion(i)
        connect.Update()
        single_PV = connect.GetOutput()

        # Clean unused points
        surface = apply_vtk_geom_filter(single_PV)

        cln = vtk.vtkCleanPolyData()
        cln.SetInputData(surface)
        cln.Update()
        surface = cln.GetOutput()

        if name1.startswith("L"):
            phie_v = np.max(vtk_to_numpy(surface.GetCellData().GetArray('phie_v')))
        elif name1.startswith("R"):
            phie_v = np.min(vtk_to_numpy(surface.GetCellData().GetArray('phie_v')))

        if name1.startswith("L") and phie_v > 0.04:  # 0.025
            found, val = optimize_shape_PV(surface, 10, 0)
            if found:
                single_PV = vtk_thr(single_PV, 1, "CELLS", "phie_v", val)
                surface = apply_vtk_geom_filter(single_PV)

        elif name1.startswith("R") and phie_v < 0.9:  # 0.975
            found, val = optimize_shape_PV(surface, 10, 1)
            if found:
                single_PV = vtk_thr(single_PV, 0, "CELLS", "phie_v", val)
                surface = apply_vtk_geom_filter(single_PV)

        centerOfMassFilter = vtk.vtkCenterOfMass()
        centerOfMassFilter.SetInputData(surface)
        centerOfMassFilter.SetUseScalarsAsWeights(False)
        centerOfMassFilter.Update()

        c_mass = centerOfMassFilter.GetCenter()

        centroid1_d = np.sqrt(np.sum((np.array(centroid1) - np.array(c_mass)) ** 2, axis=0))
        centroid2_d = np.sqrt(np.sum((np.array(centroid2) - np.array(c_mass)) ** 2, axis=0))

        if centroid1_d < centroid2_d:
            PVs[name1] = vtk_to_numpy(single_PV.GetCellData().GetArray('Global_ids'))
        else:
            PVs[name2] = vtk_to_numpy(single_PV.GetCellData().GetArray('Global_ids'))

        connect.DeleteSpecifiedRegion(i)
        connect.Update()

    return PVs


def optimize_shape_PV(surface, num, bound):
    if bound == 0:
        phie_v = np.max(vtk_to_numpy(surface.GetCellData().GetArray('phie_v')))
    else:
        phie_v = np.min(vtk_to_numpy(surface.GetCellData().GetArray('phie_v')))

    arr = np.linspace(bound, phie_v, num)

    c_mass_l = []
    found = 0
    for l in range(num - 1):
        if bound == 0:
            out = vtk_thr(surface, 2, "CELLS", "phie_v", arr[l], arr[l + 1])
        else:
            out = vtk_thr(surface, 2, "CELLS", "phie_v", arr[l + 1], arr[l])

        centerOfMassFilter = vtk.vtkCenterOfMass()
        centerOfMassFilter.SetInputData(apply_vtk_geom_filter(out))
        centerOfMassFilter.SetUseScalarsAsWeights(False)
        centerOfMassFilter.Update()

        c_mass_l.append(centerOfMassFilter.GetCenter())

    v1 = np.array(c_mass_l[0]) - np.array(c_mass_l[1])
    for l in range(1, num - 2):
        v2 = np.array(c_mass_l[l]) - np.array(c_mass_l[l + 1])
        if 1 - cosine(v1, v2) < 0:
            found = 1
            break

    return found, arr[l - 1]
