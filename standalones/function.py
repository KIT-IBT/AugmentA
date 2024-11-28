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
import scipy.spatial as spatial
import vtk

from vtk_opencarp_helper_methods.mathematical_operations.vector_operations import get_normalized_cross_product
from vtk_opencarp_helper_methods.vtk_methods.converters import vtk_to_numpy
from vtk_opencarp_helper_methods.vtk_methods.filters import apply_vtk_geom_filter, clean_polydata, \
    get_elements_above_plane
from vtk_opencarp_helper_methods.vtk_methods.finder import find_closest_point
from vtk_opencarp_helper_methods.vtk_methods.init_objects import initialize_plane, init_connectivity_filter, \
    ExtractionModes


def to_polydata(mesh):
    return apply_vtk_geom_filter(mesh)


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
    points_array_1 = vtk_to_numpy(vtk_points_1.GetData())
    points_array_2 = vtk_to_numpy(vtk_points_2.GetData())
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
    mv_points_array_1 = vtk_to_numpy(mv_points.GetData())
    kDTree = vtk.vtkKdTree()
    kDTree.BuildLocatorFromPoints(mv_points)
    id_list = vtk.vtkIdList()
    kDTree.FindClosestNPoints(1, center_lpv, id_list)
    index = id_list.GetId(0)
    res_1 = mv_points_array_1[index]
    distances = np.linalg.norm(mv_points_array_1 - res_1, axis=1)
    res_2 = mv_points_array_1[np.argmax(distances)]

    d_mv_l = np.linalg.norm(mv_points_array_1 - res_1, axis=1)  # Distances to res_1
    d_mv_r = np.linalg.norm(mv_points_array_1 - res_2, axis=1)  # Distances to res_2
    distance_diff = np.abs(d_mv_l - d_mv_r)  # Absolute differences

    res_3 = mv_points_array_1[distance_diff.index(min(distance_diff))]

    distance_2 = np.linalg.norm(mv_points_array_1 - res_3, axis=1)

    res_4 = mv_points_array_1[distance_2.index(max(distance_2))]

    return res_1, res_2, res_3, res_4


def cut_a_band_from_model(polydata, point_1, point_2, point_3, width):
    norm_1 = get_normalized_cross_product(point_1, point_2, point_3)

    point_pass = point_1 + 0.5 * width * norm_1

    plane = initialize_plane(norm_1[0], point_pass[0])

    extract_mesh_1 = get_elements_above_plane(polydata, plane)

    point_moved = point_1 - 0.5 * width * norm_1

    plane2 = initialize_plane(-norm_1[0], point_moved[0])

    band = apply_vtk_geom_filter(get_elements_above_plane(extract_mesh_1, plane2))

    return band


def cut_into_two_parts(polydata, point_1, point_2, point_3):
    norm_1 = get_normalized_cross_product(point_1, point_2, point_3)

    plane = initialize_plane(norm_1[0], point_1)
    plane2 = initialize_plane(-norm_1[0], point_1)

    sub_1 = get_elements_above_plane(polydata, plane)
    sub_2 = get_elements_above_plane(polydata, plane2, extract_boundary_cells_on=True)

    return sub_1, sub_2


def dijkstra_path(polydata, start_vertex, end_vertex):
    path = vtk.vtkDijkstraGraphGeodesicPath()
    path.SetInputData(polydata)
    # attention the return value will be reversed
    path.SetStartVertex(end_vertex)
    path.SetEndVertex(start_vertex)
    path.Update()
    points_data = path.GetOutput().GetPoints().GetData()
    points_data = vtk_to_numpy(points_data)
    return points_data


def get_mv_l_and_r(mv_band, center_lpv):
    connect = init_connectivity_filter(mv_band, ExtractionModes.ALL_REGIONS)
    connect.SetExtractionModeToSpecifiedRegions()
    connect.AddSpecifiedRegion(1)
    connect.Update()

    # Clean unused points
    surface = to_polydata(connect.GetOutput())

    ring = vtk_to_numpy(clean_polydata(surface).GetPoints().GetData())
    center_point_1 = np.mean(ring, axis=0)

    connect.DeleteSpecifiedRegion(1)
    connect.AddSpecifiedRegion(0)
    connect.Update()

    # Clean unused points
    surface = to_polydata(connect.GetOutput())
    ring = vtk_to_numpy(clean_polydata(surface).GetPoints().GetData())
    center_point_2 = np.mean(ring, axis=0)

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
    return find_closest_point(polydata, coordiante)


def multidim_intersect(arr1, arr2):
    arr1_view = arr1.view([('', arr1.dtype)] * arr1.shape[1])
    arr2_view = arr2.view([('', arr2.dtype)] * arr2.shape[1])
    intersected = np.intersect1d(arr1_view, arr2_view)
    return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])
