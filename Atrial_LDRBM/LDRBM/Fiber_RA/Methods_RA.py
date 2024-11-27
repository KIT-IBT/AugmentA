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
import numpy as np
import vtk
from scipy.spatial import cKDTree
from vtk.numpy_interface import dataset_adapter as dsa

import vtk_opencarp_helper_methods.AugmentA_methods.vtk_operations
from Atrial_LDRBM.LDRBM.Fiber_LA import Methods_LA
from Atrial_LDRBM.LDRBM.Fiber_LA.Methods_LA import generate_spline_points
from vtk_opencarp_helper_methods.AugmentA_methods.vtk_operations import get_normalized_cross_product
from vtk_opencarp_helper_methods.openCARP.exporting import write_to_elem, write_to_pts, write_to_lon
from vtk_opencarp_helper_methods.vtk_methods.converters import vtk_to_numpy, numpy_to_vtk
from vtk_opencarp_helper_methods.vtk_methods.exporting import vtk_unstructured_grid_writer, vtk_polydata_writer, \
    vtk_xml_unstructured_grid_writer, vtk_obj_writer
from vtk_opencarp_helper_methods.vtk_methods.filters import apply_vtk_geom_filter, get_vtk_geom_filter_port, \
    clean_polydata, vtk_append, apply_extract_cell_filter, get_elements_above_plane
from vtk_opencarp_helper_methods.vtk_methods.finder import find_closest_point
from vtk_opencarp_helper_methods.vtk_methods.init_objects import initialize_plane
from vtk_opencarp_helper_methods.vtk_methods.reader import smart_reader

vtk_version = vtk.vtkVersion.GetVTKSourceVersion().split()[-1].split('.')[0]


def downsample_path(points_data, step):
    # down sampling
    path_all = np.asarray(
        [points_data[i] for i in range(len(points_data)) if i % step == 0 or i == len(points_data) - 1])

    # fit a spline
    return vtk_to_numpy(generate_spline_points(path_all).GetPoints().GetData())


def move_surf_along_normals(mesh, eps, direction):
    polydata = apply_vtk_geom_filter(mesh)

    normalGenerator = vtk.vtkPolyDataNormals()
    normalGenerator.SetInputData(polydata)
    normalGenerator.ComputeCellNormalsOff()
    normalGenerator.ComputePointNormalsOn()
    normalGenerator.ConsistencyOn()
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


def generate_bilayer(args, job, endo, epi, max_dist=np.inf):
    geo_port, _geo_filter = get_vtk_geom_filter_port(endo)

    reverse = vtk.vtkReverseSense()
    reverse.ReverseCellsOn()
    reverse.ReverseNormalsOn()
    reverse.SetInputConnection(geo_port)
    reverse.Update()

    endo = vtk.vtkUnstructuredGrid()
    endo.DeepCopy(reverse.GetOutput())
    # endo.DeepCopy(extract_surf.GetOutputPort())

    endo_pts = vtk_to_numpy(endo.GetPoints().GetData())
    epi_pts = vtk_to_numpy(epi.GetPoints().GetData())

    tree = cKDTree(epi_pts)
    dd, ii = tree.query(endo_pts, distance_upper_bound=max_dist)  # , n_jobs=-1)

    endo_ids = np.where(dd != np.inf)[0]
    epi_ids = ii[endo_ids]
    lines = vtk.vtkCellArray()

    for i in range(len(endo_ids)):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i);
        line.GetPointIds().SetId(1, len(endo_ids) + i);
        lines.InsertNextCell(line)

    points = np.vstack((endo_pts[endo_ids], epi_pts[epi_ids]))
    polydata = vtk.vtkUnstructuredGrid()
    vtkPts = vtk.vtkPoints()
    vtkPts.SetData(numpy_to_vtk(points))
    polydata.SetPoints(vtkPts)
    polydata.SetCells(3, lines)

    fibers = np.zeros((len(endo_ids), 3), dtype="float32")
    fibers[:, 0] = 1

    tag = np.ones((len(endo_ids),), dtype=int)
    tag[:] = 100

    meshNew = dsa.WrapDataObject(polydata)
    meshNew.CellData.append(tag, "elemTag")
    meshNew.CellData.append(fibers, "fiber")
    fibers = np.zeros((len(endo_ids), 3), dtype="float32")
    fibers[:, 1] = 1
    meshNew.CellData.append(fibers, "sheet")
    bilayer_1 = meshNew.VTKObject

    if args.debug:
        vtk_xml_unstructured_grid_writer(job.ID + "/result_RA/test_LA_RA_bilayer.vtu", bilayer_1)

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(job.ID + "/result_RA/test_LA_RA_bilayer.vtu")  #
    reader.Update()
    test = reader.GetOutput()

    bilayer = vtk_append([endo, epi, test], True)

    if args.ofmt == 'vtk':
        vtk_unstructured_grid_writer(job.ID + "/result_RA/LA_RA_bilayer_with_fiber.vtk", bilayer, store_binary=True)
    else:
        vtk_xml_unstructured_grid_writer(job.ID + "/result_RA/LA_RA_bilayer_with_fiber.vtu",
                                         bilayer)  # Has elemTag!

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(job.ID + "/result_RA/LA_RA_bilayer_with_fiber.vtu")  # Has elemTag! :)
    reader.Update()
    bilayer = reader.GetOutput()

    return bilayer


def write_bilayer(bilayer, args, job):
    file_name = job.ID + "/result_RA/LA_RA_bilayer_with_fiber"
    if args.ofmt == 'vtk':
        vtk_unstructured_grid_writer(f"{file_name}.vtk", bilayer, True)
    else:
        vtk_xml_unstructured_grid_writer(f"{file_name}.vtu", bilayer)

    pts = vtk_to_numpy(bilayer.GetPoints().GetData())

    write_to_pts(f'{file_name}.pts', pts)

    tag_epi = vtk.util.vtk_to_numpy(bilayer.GetCellData().GetArray('elemTag'))

    write_to_elem(f'{file_name}.elem', bilayer, tag_epi)

    el_epi = vtk.util.vtk_to_numpy(bilayer.GetCellData().GetArray('fiber'))
    sheet_epi = vtk.util.vtk_to_numpy(bilayer.GetCellData().GetArray('sheet'))

    write_to_lon(f'{file_name}.lon', el_epi, sheet_epi)


def generate_sheet_dir(args, model, job):
    fiber = model.GetCellData().GetArray('fiber')
    fiber = vtk_to_numpy(fiber)

    fiber = np.where(fiber == [0, 0, 0], [1, 0, 0], fiber).astype("float32")

    print('reading done!')

    '''
    extract the surface
    '''
    surface = apply_vtk_geom_filter(model)
    cln_surface = clean_polydata(surface)

    '''
    calculate normals of surface cells
    '''
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(surface)
    normals.ComputeCellNormalsOn()
    normals.ComputePointNormalsOff()
    normals.Update()
    normal_vectors = normals.GetOutput().GetCellData().GetArray('Normals')
    normal_vectors = vtk_to_numpy(normal_vectors)

    # print('Normal vectors: \n', normal_vectors, '\n')
    print('Number of normals: ', len(normal_vectors))
    print('2nd norm of vectors: ', np.linalg.norm(normal_vectors[0], ord=2), '\n')

    if args.mesh_type == "vol":

        '''
        calculate the centers of surface mesh cells
        '''
        filter_cell_centers = vtk.vtkCellCenters()
        filter_cell_centers.SetInputData(cln_surface)
        filter_cell_centers.Update()
        center_surface = filter_cell_centers.GetOutput().GetPoints()
        center_surface_array = vtk_to_numpy(center_surface.GetData())
        print('Number of center_surface: ', len(center_surface_array), '\n')

        '''
        calculate the centers of volume mesh cells
        '''
        filter_cell_centers = vtk.vtkCellCenters()
        filter_cell_centers.SetInputData(model)
        filter_cell_centers.Update()
        center_volume = filter_cell_centers.GetOutput().GetPoints().GetData()
        center_volume = vtk_to_numpy(center_volume)
        print('Number of center_volume: ', len(center_volume), '\n')

        '''
        Mapping
        '''
        normals = np.ones([len(center_volume), 3], dtype=float)
        kDTree = vtk.vtkKdTree()
        kDTree.BuildLocatorFromPoints(center_surface)

        for i in range(len(center_volume)):
            id_list = vtk.vtkIdList()
            kDTree.FindClosestNPoints(1, center_volume[i], id_list)
            index = id_list.GetId(0)
            normals[i] = normal_vectors[index]

        print('Mapping done!')
        '''
        calculate the sheet
        '''

    elif args.mesh_type == "bilayer":
        normals = normal_vectors

    sheet = np.cross(normals, fiber)
    sheet = np.where(sheet == [0, 0, 0], [1, 0, 0], sheet).astype("float32")
    # normalize
    abs_sheet = np.linalg.norm(sheet, axis=1, keepdims=True)
    sheet_norm = sheet / abs_sheet

    '''
    writing
    '''
    print('writing...')

    meshNew = dsa.WrapDataObject(model)
    meshNew.CellData.append(fiber, "fiber")
    meshNew.CellData.append(sheet_norm, "sheet")
    if args.debug:
        vtk_unstructured_grid_writer(job.ID + "/result_RA/model_with_sheet.vtk", meshNew.VTKObject)
        print('writing... done!')

    return meshNew.VTKObject


def vtk_thr(model, mode, points_cells, array, thr1, thr2="None"):
    return vtk_opencarp_helper_methods.AugmentA_methods.vtk_operations.vtk_thr(model, mode, points_cells, array, thr1,
                                                                               thr2)


def creat_tube_around_spline(points_data, radius):
    # Interpolate the scalars
    interpolatedRadius = vtk.vtkTupleInterpolator()
    interpolatedRadius.SetInterpolationTypeToLinear()
    interpolatedRadius.SetNumberOfComponents(1)

    # Generate the radius scalars
    tubeRadius = vtk.vtkDoubleArray()
    n = generate_spline_points(points_data).GetNumberOfPoints()
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
    tuber.SetNumberOfSides(40)
    tuber.SidesShareVerticesOn()
    tuber.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
    tuber.SetCapping(1)
    tuber.Update()

    triangle = vtk.vtkTriangleFilter()
    triangle.SetInputData(tuber.GetOutput())
    triangle.Update()

    tuber = triangle
    return tuber


def dijkstra_path(polydata, StartVertex, EndVertex):
    path = vtk.vtkDijkstraGraphGeodesicPath()
    path.SetInputData(polydata)

    path.SetStartVertex(EndVertex)
    path.SetEndVertex(StartVertex)
    path.Update()
    points_data = path.GetOutput().GetPoints().GetData()
    points_data = vtk_to_numpy(points_data)
    return points_data


def dijkstra_path_coord(polydata, StartVertex, EndVertex):
    StartVertex = find_closest_point(polydata, StartVertex)
    EndVertex = find_closest_point(polydata, EndVertex)

    path = vtk.vtkDijkstraGraphGeodesicPath()
    path.SetInputData(polydata)

    path.SetStartVertex(EndVertex)
    path.SetEndVertex(StartVertex)
    path.Update()
    points_data = path.GetOutput().GetPoints().GetData()
    points_data = vtk_to_numpy(points_data)
    return points_data


def dijkstra_path_on_a_plane(polydata, args, StartVertex, EndVertex, plane_point):
    point_start = np.asarray(polydata.GetPoint(StartVertex))
    point_end = np.asarray(polydata.GetPoint(EndVertex))
    point_third = plane_point

    norm_1 = get_normalized_cross_product(point_start, point_end, point_third)

    plane = initialize_plane(norm_1, point_start)

    extracted_mesh_1 = get_elements_above_plane(polydata, plane)

    point_moved = point_start - 2 * args.scale * norm_1

    plane2 = initialize_plane(-norm_1, point_moved)

    band = clean_polydata(apply_vtk_geom_filter(get_elements_above_plane(extracted_mesh_1, plane2)))

    if args.debug:
        writer_vtk(band, f'{args.mesh}_surf/' + "band_" + str(StartVertex) + "_" + str(EndVertex) + ".vtk")

    StartVertex = find_closest_point(band, point_start)
    EndVertex = find_closest_point(band, point_end)

    points_data = dijkstra_path(band, StartVertex, EndVertex)
    return points_data


def creat_sphere(center, radius):
    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(center[0], center[1], center[2])
    sphere.SetRadius(radius)
    sphere.SetThetaResolution(40)
    sphere.SetPhiResolution(40)
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
    tube.SetNumberOfSides(40)
    tube.Update()
    return tube


def find_elements_around_path_within_radius(mesh, points_data, radius):
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

    id_set = set()
    for i in range(mesh_cell_id_list.GetNumberOfIds()):
        id_set.add(mesh_cell_id_list.GetId(i))

    return id_set


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

    path_ct_id_icv = find_closest_point(endo, path_icv)
    path_ct_id_scv = find_closest_point(endo, path_scv)

    return path_ct_id_icv, path_ct_id_scv


def get_tv_end_points_id(endo, ra_tv_s_surface, ra_ivc_surface, ra_svc_surface, ra_tv_surface):
    tv_center = get_mean_point(ra_tv_surface)
    tv_ivc_center = get_mean_point(ra_ivc_surface)
    tv_svc_center = get_mean_point(ra_svc_surface)

    norm_1 = get_normalized_cross_product(tv_center, tv_ivc_center, tv_svc_center)
    moved_center = tv_center - norm_1 * 5

    plane = initialize_plane(-norm_1[0], moved_center[0])

    extracted_mesh = get_elements_above_plane(ra_tv_s_surface, plane)

    connect = vtk.vtkConnectivityFilter()
    connect.SetInputData(extracted_mesh)
    connect.SetExtractionModeToAllRegions()
    connect.Update()
    connect.SetExtractionModeToSpecifiedRegions()
    connect.AddSpecifiedRegion(1)
    connect.Update()

    # Clean unused points
    surface = apply_vtk_geom_filter(connect.GetOutput())
    points_data = clean_polydata(surface).GetPoints().GetData()
    ring = vtk_to_numpy(points_data)
    center_point_1 = np.asarray([np.mean(ring[:, 0]), np.mean(ring[:, 1]), np.mean(ring[:, 2])])

    connect.DeleteSpecifiedRegion(1)
    connect.AddSpecifiedRegion(0)
    connect.Update()

    # Clean unused points
    surface = apply_vtk_geom_filter(connect.GetOutput())

    points_data = clean_polydata(surface).GetPoints().GetData()
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

    path_tv_id_icv = find_closest_point(endo, center_point_icv)
    path_tv_id_scv = find_closest_point(endo, center_point_scv)

    return path_tv_id_icv, path_tv_id_scv


def extract_largest_region(mesh):
    connect = vtk.vtkConnectivityFilter()
    connect.SetInputData(mesh)
    connect.SetExtractionModeToLargestRegion()
    connect.Update()
    surface = connect.GetOutput()

    surface = apply_vtk_geom_filter(surface)

    return clean_polydata(surface)


def assign_ra_appendage(model, SCV, appex_point, tag, elemTag):
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
        tag[index] = elemTag

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

    return apply_extract_cell_filter(endo, endo_cell_id_list)


def get_connection_point_la_and_ra_surface(appen_point, la_mv_surface, la_rpv_inf_surface, la_epi_surface,
                                           ra_epi_surface):
    point_1_id = find_closest_point(la_mv_surface, appen_point)
    point_1 = la_mv_surface.GetPoint(point_1_id)

    point_2_id = find_closest_point(la_rpv_inf_surface, appen_point)
    point_2 = la_rpv_inf_surface.GetPoint(point_2_id)

    point_1_id = find_closest_point(la_epi_surface, point_1)
    point_2_id = find_closest_point(la_epi_surface, point_2)

    bb_aux_l_points = dijkstra_path(apply_vtk_geom_filter(la_epi_surface), point_1_id, point_2_id)
    length = len(bb_aux_l_points)
    la_connect_point = bb_aux_l_points[int(length * 0.5)]

    # ra
    la_epi_surface = apply_vtk_geom_filter(la_epi_surface)
    ra_epi_surface = apply_vtk_geom_filter(ra_epi_surface)

    la_connect_point_id = find_closest_point(la_epi_surface, la_connect_point)
    la_connect_point = la_epi_surface.GetPoint(la_connect_point_id)

    ra_connect_point_id = find_closest_point(ra_epi_surface, la_connect_point)
    ra_connect_point = ra_epi_surface.GetPoint(ra_connect_point_id)

    return la_connect_point, ra_connect_point


def get_connection_point_la_and_ra(appen_point):
    la_mv_surface = smart_reader('../../Generate_Boundaries/LA/result/la_mv_surface.vtk')
    la_rpv_inf_surface = smart_reader('../../Generate_Boundaries/LA/result/la_rpv_inf_surface.vtk')
    endo = smart_reader('../../Generate_Boundaries/LA/result/la_endo_surface.vtk')
    la_epi_surface = smart_reader('../../Generate_Boundaries/LA/result/la_epi_surface.vtk')
    ra_epi_surface = smart_reader('../../Generate_Boundaries/RA/result/ra_epi_surface.vtk')

    point_1_id = find_closest_point(la_mv_surface, appen_point)
    point_1 = la_mv_surface.GetPoint(point_1_id)

    point_2_id = find_closest_point(la_rpv_inf_surface, appen_point)
    point_2 = la_rpv_inf_surface.GetPoint(point_2_id)

    point_1_id_endo = find_closest_point(endo, point_1)
    point_2_id_endo = find_closest_point(endo, point_2)

    bb_aux_l_points = dijkstra_path(apply_vtk_geom_filter(endo), point_1_id_endo, point_2_id_endo)
    length = len(bb_aux_l_points)
    la_connect_point = bb_aux_l_points[int(length * 0.5)]

    # ra
    la_epi_surface = apply_vtk_geom_filter(la_epi_surface)

    ra_epi_surface = apply_vtk_geom_filter(ra_epi_surface)

    la_connect_point_id = find_closest_point(la_epi_surface, la_connect_point)
    la_connect_point = la_epi_surface.GetPoint(la_connect_point_id)

    ra_connect_point_id = find_closest_point(ra_epi_surface, la_connect_point)
    ra_connect_point = ra_epi_surface.GetPoint(ra_connect_point_id)

    return la_connect_point, ra_connect_point


def get_bachmann_path_left(appendage_basis, lpv_sup_basis):
    la_mv_surface = smart_reader('../../Generate_Boundaries/LA/result/la_mv_surface.vtk')
    la_lpv_inf_surface = smart_reader('../../Generate_Boundaries/LA/result/la_lpv_inf_surface.vtk')
    endo = smart_reader('../../Generate_Boundaries/LA/result/la_endo_surface.vtk')
    epi = smart_reader('../../Generate_Boundaries/LA/result/la_epi_surface.vtk')

    appendage_basis_id = find_closest_point(epi, appendage_basis)
    lpv_sup_basis_id = find_closest_point(epi, lpv_sup_basis)

    left_inf_pv_center = get_mean_point(la_lpv_inf_surface)
    point_l1_id = find_closest_point(la_mv_surface, left_inf_pv_center)
    point_l1 = la_mv_surface.GetPoint(point_l1_id)
    bb_mv_id = find_closest_point(epi, point_l1)

    epi_polydata = apply_vtk_geom_filter(epi)

    bb_1_points = dijkstra_path(epi_polydata, lpv_sup_basis_id, appendage_basis_id)
    bb_2_points = dijkstra_path(epi_polydata, appendage_basis_id, bb_mv_id)
    np.delete(bb_1_points, -1)
    bb_left = np.concatenate((bb_1_points, bb_2_points), axis=0)

    return bb_left, appendage_basis


def create_free_bridge_semi_auto(la_epi, ra_epi, ra_point, radius):
    point_end_id = find_closest_point(la_epi, ra_point)
    point_end = la_epi.GetPoint(point_end_id)
    start = np.asarray(ra_point)
    end = np.asarray(point_end)
    point_data = np.vstack((start, end))
    tube = creat_tube_around_spline(point_data, radius)

    sphere_a = creat_sphere(start, radius * 1.02)
    sphere_b = creat_sphere(end, radius * 1.02)

    fiber_direction = end - start
    fiber_direction_norm = normalize_vector(fiber_direction)
    return tube, sphere_a, sphere_b, fiber_direction_norm


def creat_center_line(start_end_point):
    return Methods_LA.creat_center_line(start_end_point)


def smart_bridge_writer(tube, sphere_1, sphere_2, name, job):
    meshNew = dsa.WrapDataObject(tube.GetOutput())
    vtk_obj_writer(job.ID + "/bridges/" + str(name) + "_tube.obj", meshNew.VTKObject)
    meshNew = dsa.WrapDataObject(sphere_1.GetOutput())
    vtk_obj_writer(job.ID + "/bridges/" + str(name) + "_sphere_1.obj", meshNew.VTKObject)
    meshNew = dsa.WrapDataObject(sphere_2.GetOutput())
    vtk_obj_writer(job.ID + "/bridges/" + str(name) + "_sphere_2.obj", meshNew.VTKObject)


def create_pts(array_points, array_name, mesh_dir):
    write_to_pts(f"{mesh_dir}{array_name}.pts", array_points)


def to_polydata(mesh):
    polydata = apply_vtk_geom_filter(mesh)

    return polydata


def writer_vtk(mesh, filename):
    vtk_polydata_writer(filename, to_polydata(mesh))
