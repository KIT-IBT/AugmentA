#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:55:02 2021

@author: Luca Azzolin
"""

import numpy as np
from glob import glob
import pandas as pd
import vtk
from vtk.util import numpy_support
import scipy.spatial as spatial
from vtk.numpy_interface import dataset_adapter as dsa


def to_polydata(mesh):
    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(mesh)
    geo_filter.Update()
    polydata = geo_filter.GetOutput()
    return polydata


def get_mean_point(ring_points):
    center_point = [np.mean(ring_points[:, 0]), np.mean(ring_points[:, 1]), np.mean(ring_points[:, 2])]
    center_point = np.array(center_point)
    return center_point


def get_farthest_point_pair(point_array_1, point_array_2):
    # convex hull algorithm
    pointarray = np.vstack((point_array_1, point_array_2))
    candidates = pointarray[spatial.ConvexHull(pointarray).vertices]
    # get distances between each pair of candidate points
    dist_mat = spatial.distance_matrix(candidates, candidates)
    # get indices of candidates that are furthest apart
    i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
    if candidates[i] in point_array_1:
        res_1 = candidates[i]
        res_2 = candidates[j]
    else:
        res_1 = candidates[j]
        res_2 = candidates[i]
    return res_1, res_2


def get_closest_point(vtk_points_1, vtk_points_2):
    points_array_1 = vtk.util.numpy_support.vtk_to_numpy(vtk_points_1.GetData())
    points_array_2 = vtk.util.numpy_support.vtk_to_numpy(vtk_points_2.GetData())
    center_1 = get_mean_point(points_array_1)
    center_2 = get_mean_point(points_array_2)

    kDTree = vtk.vtkKdTree()
    kDTree.BuildLocatorFromPoints(vtk_points_1)
    id_list = vtk.vtkIdList()
    kDTree.FindClosestNPoints(1, center_2, id_list)
    index = id_list.GetId(0)
    res_1 = points_array_1[index]

    kDTree = vtk.vtkKdTree()
    kDTree.BuildLocatorFromPoints(vtk_points_2)
    id_list = vtk.vtkIdList()
    kDTree.FindClosestNPoints(1, center_1, id_list)
    index = id_list.GetId(0)
    res_2 = points_array_2[index]

    return res_1, res_2


def find_points_on_mv(mv_points, center_lpv):
    mv_points_array_1 = vtk.util.numpy_support.vtk_to_numpy(mv_points.GetData())
    kDTree = vtk.vtkKdTree()
    kDTree.BuildLocatorFromPoints(mv_points)
    id_list = vtk.vtkIdList()
    kDTree.FindClosestNPoints(1, center_lpv, id_list)
    index = id_list.GetId(0)
    res_1 = mv_points_array_1[index]
    distance = []
    for i in range(len(mv_points_array_1)):
        d = np.linalg.norm(mv_points_array_1[i] - res_1, ord=None, axis=None, keepdims=False)
        distance += [d]
    res_2 = mv_points_array_1[distance.index(max(distance))]

    distance_diff = []
    for i in range(len(mv_points_array_1)):
        d_mv_l = np.linalg.norm(mv_points_array_1[i] - res_1, ord=None, axis=None, keepdims=False)
        d_mv_r = np.linalg.norm(mv_points_array_1[i] - res_2, ord=None, axis=None, keepdims=False)
        distance_diff += [abs(d_mv_l - d_mv_r)]
    res_3 = mv_points_array_1[distance_diff.index(min(distance_diff))]

    distance_2 = []
    for i in range(len(mv_points_array_1)):
        d_2 = np.linalg.norm(mv_points_array_1[i] - res_3, ord=None, axis=None, keepdims=False)
        distance_2 += [d_2]
    res_4 = mv_points_array_1[distance_2.index(max(distance_2))]

    return res_1, res_2, res_3, res_4


def cut_a_band_from_model(polydata, point_1, point_2, point_3, width):
    v1 = point_2 - point_1
    v2 = point_3 - point_1
    norm = np.cross(v1, v2)
    #
    # # normlize norm
    n = np.linalg.norm([norm], axis=1, keepdims=True)
    norm_1 = norm / n

    point_pass = point_1 + 0.5 * width * norm_1
    plane = vtk.vtkPlane()
    plane.SetNormal(norm_1[0][0], norm_1[0][1], norm_1[0][2])
    plane.SetOrigin(point_pass[0][0], point_pass[0][1], point_pass[0][2])

    meshExtractFilter1 = vtk.vtkExtractGeometry()
    meshExtractFilter1.SetInputData(polydata)
    meshExtractFilter1.SetImplicitFunction(plane)
    meshExtractFilter1.Update()

    point_moved = point_1 - 0.5 * width * norm_1
    # print(point_moved[0][0])
    plane2 = vtk.vtkPlane()
    plane2.SetNormal(-norm_1[0][0], -norm_1[0][1], -norm_1[0][2])
    plane2.SetOrigin(point_moved[0][0], point_moved[0][1], point_moved[0][2])

    meshExtractFilter2 = vtk.vtkExtractGeometry()
    meshExtractFilter2.SetInputData(meshExtractFilter1.GetOutput())
    meshExtractFilter2.SetImplicitFunction(plane2)
    meshExtractFilter2.Update()
    band = meshExtractFilter2.GetOutput()
    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(band)
    geo_filter.Update()
    band = geo_filter.GetOutput()

    return band


def cut_into_two_parts(polydata, point_1, point_2, point_3):
    v1 = point_2 - point_1
    v2 = point_3 - point_1
    norm = np.cross(v1, v2)
    #
    # # normlize norm
    n = np.linalg.norm([norm], axis=1, keepdims=True)
    norm_1 = norm / n

    plane = vtk.vtkPlane()
    plane.SetNormal(norm_1[0][0], norm_1[0][1], norm_1[0][2])
    plane.SetOrigin(point_1[0], point_1[1], point_1[2])

    plane2 = vtk.vtkPlane()
    plane2.SetNormal(-norm_1[0][0], -norm_1[0][1], -norm_1[0][2])
    plane2.SetOrigin(point_1[0], point_1[1], point_1[2])

    meshExtractFilter1 = vtk.vtkExtractGeometry()
    meshExtractFilter1.SetInputData(polydata)
    meshExtractFilter1.SetImplicitFunction(plane)
    meshExtractFilter1.Update()
    sub_1 = meshExtractFilter1.GetOutput()

    meshExtractFilter2 = vtk.vtkExtractGeometry()
    meshExtractFilter2.SetInputData(polydata)
    meshExtractFilter2.ExtractBoundaryCellsOn()
    meshExtractFilter2.SetImplicitFunction(plane2)
    meshExtractFilter2.Update()
    sub_2 = meshExtractFilter2.GetOutput()

    return sub_1, sub_2


def dijkstra_path(polydata, StartVertex, EndVertex):
    path = vtk.vtkDijkstraGraphGeodesicPath()
    path.SetInputData(polydata)
    # attention the return value will be reversed
    path.SetStartVertex(EndVertex)
    path.SetEndVertex(StartVertex)
    path.Update()
    points_data = path.GetOutput().GetPoints().GetData()
    points_data = vtk.util.numpy_support.vtk_to_numpy(points_data)
    return points_data


def get_mv_l_and_r(mv_band, center_lpv):
    connect = vtk.vtkConnectivityFilter()
    connect.SetInputData(mv_band)
    connect.SetExtractionModeToAllRegions()
    connect.Update()
    connect.SetExtractionModeToSpecifiedRegions()
    connect.AddSpecifiedRegion(1)
    connect.Update()

    # Clean unused points
    surface = to_polydata(connect.GetOutput())
    cln = vtk.vtkCleanPolyData()
    cln.SetInputData(surface)
    cln.Update()
    points_data = cln.GetOutput().GetPoints().GetData()
    ring = vtk.util.numpy_support.vtk_to_numpy(points_data)
    center_point_1 = np.asarray([np.mean(ring[:, 0]), np.mean(ring[:, 1]), np.mean(ring[:, 2])])

    connect.DeleteSpecifiedRegion(1)
    connect.AddSpecifiedRegion(0)
    connect.Update()

    # Clean unused points
    surface = to_polydata(connect.GetOutput())
    cln = vtk.vtkCleanPolyData()
    cln.SetInputData(surface)
    cln.Update()
    points_data = cln.GetOutput().GetPoints().GetData()
    ring = vtk.util.numpy_support.vtk_to_numpy(points_data)
    center_point_2 = np.asarray([np.mean(ring[:, 0]), np.mean(ring[:, 1]), np.mean(ring[:, 2])])
    dis_1 = np.linalg.norm(center_point_1 - center_lpv)
    dis_2 = np.linalg.norm(center_point_2 - center_lpv)
    if dis_1 > dis_2:
        mv_l = center_point_2
        mv_r = center_point_1
    else:
        mv_l = center_point_1
        mv_r = center_point_2
    return mv_l, mv_r


def get_closest_point_id_from_polydata(polydata, coordiante):
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(polydata)
    loc.BuildLocator()
    return loc.FindClosestPoint(coordiante)


def multidim_intersect(arr1, arr2):
    arr1_view = arr1.view([('', arr1.dtype)] * arr1.shape[1])
    arr2_view = arr2.view([('', arr2.dtype)] * arr2.shape[1])
    intersected = np.intersect1d(arr1_view, arr2_view)
    return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])
