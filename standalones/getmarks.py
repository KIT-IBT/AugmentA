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
from glob import glob
import pandas as pd
import vtk
from vtk.util import numpy_support
from vtk.numpy_interface import dataset_adapter as dsa
from scipy import spatial
import function
from sklearn.neighbors import NearestNeighbors
import argparse


def parser():
    parser = argparse.ArgumentParser(description='Generate landmarks for fitting SSM.')
    parser.add_argument('--mesh',
                        type=str,
                        default="meanshape",
                        help='path to meshname')
    parser.add_argument('--prealigned',
                        type=int,
                        default=1,
                        help='set to 1 if the mesh name is LA_prealigned, 0 if it is LA_boundaries_tagged')
    parser.add_argument('--scale',
                        type=float,
                        default=1,
                        help='path to meshname')

    return parser


def get_landmarks(mesh, prealigned=1, scale=1):
    mesh_dir = "{}_surf".format(mesh)
    reader = vtk.vtkPolyDataReader()
    if prealigned:
        reader.SetFileName(mesh_dir + '/LA_prealigned.vtk')
        df = pd.read_csv(mesh + "_surf/rings_centroids_prealigned.csv")
    else:
        reader.SetFileName(mesh_dir + '/LA_boundaries_tagged.vtk')
        df = pd.read_csv(mesh + "_surf/rings_centroids.csv")
    reader.Update()
    model = reader.GetOutput()

    model_polydata = function.to_polydata(model)

    thr = vtk.vtkThreshold()
    thr.SetInputData(model)
    thr.AllScalarsOff()
    thr.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_POINTS", "boundary_tag")
    thr.ThresholdBetween(5, 5)
    thr.Update()
    rsv = thr.GetOutput()

    thr = vtk.vtkThreshold()
    thr.SetInputData(model)
    thr.AllScalarsOff()
    thr.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_POINTS", "boundary_tag")
    thr.ThresholdBetween(4, 4)
    thr.Update()
    riv = thr.GetOutput()

    thr = vtk.vtkThreshold()
    thr.SetInputData(model)
    thr.AllScalarsOff()
    thr.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_POINTS", "boundary_tag")
    thr.ThresholdBetween(3, 3)
    thr.Update()
    lsv = thr.GetOutput()

    thr = vtk.vtkThreshold()
    thr.SetInputData(model)
    thr.AllScalarsOff()
    thr.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_POINTS", "boundary_tag")
    thr.ThresholdBetween(2, 2)
    thr.Update()
    liv = thr.GetOutput()

    thr = vtk.vtkThreshold()
    thr.SetInputData(model)
    thr.AllScalarsOff()
    thr.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_POINTS", "boundary_tag")
    thr.ThresholdBetween(1, 1)
    thr.Update()
    mv = thr.GetOutput()

    rsv_points = vtk.util.numpy_support.vtk_to_numpy(rsv.GetPoints().GetData())
    riv_points = vtk.util.numpy_support.vtk_to_numpy(riv.GetPoints().GetData())
    lsv_points = vtk.util.numpy_support.vtk_to_numpy(lsv.GetPoints().GetData())
    liv_points = vtk.util.numpy_support.vtk_to_numpy(liv.GetPoints().GetData())
    mv_points = vtk.util.numpy_support.vtk_to_numpy(mv.GetPoints().GetData())

    # get farthest away and clostest points on PVs
    rs_o, ri_o = function.get_farthest_point_pair(rsv_points, riv_points)
    ls_o, li_o = function.get_farthest_point_pair(lsv_points, liv_points)

    rs_i, ri_i = function.get_closest_point(rsv.GetPoints(), riv.GetPoints())
    ls_i, li_i = function.get_closest_point(lsv.GetPoints(), liv.GetPoints())

    # Centroids of LPV and RPV
    center_lpv = function.get_mean_point(np.vstack((ls_i, li_i)))
    center_rpv = function.get_mean_point(np.vstack((rs_i, ri_i)))

    # mitral valve
    center_mv = function.get_mean_point(mv_points)

    mv_polydata = function.to_polydata(mv)
    mv_anterior, mv_posterior = function.cut_into_two_parts(mv_polydata, center_mv, center_rpv, center_lpv)

    mv_band = function.cut_a_band_from_model(function.to_polydata(mv_anterior), center_mv, center_rpv, center_lpv,
                                             10 * scale)
    mv_l_fuzzy, mv_r_fuzzy = function.get_mv_l_and_r(mv_band, center_lpv)

    # mv_anterior
    loc_an = vtk.vtkPointLocator()
    loc_an.SetDataSet(mv_anterior)
    loc_an.BuildLocator()
    mv_an_l = mv_anterior.GetPoint(loc_an.FindClosestPoint(mv_l_fuzzy))
    mv_an_r = mv_anterior.GetPoint(loc_an.FindClosestPoint(mv_r_fuzzy))

    # mv_posterior
    loc_po = vtk.vtkPointLocator()
    loc_po.SetDataSet(mv_posterior)
    loc_po.BuildLocator()
    mv_po_l = mv_posterior.GetPoint(loc_po.FindClosestPoint(mv_l_fuzzy))
    mv_po_r = mv_posterior.GetPoint(loc_po.FindClosestPoint(mv_r_fuzzy))

    path_an = function.dijkstra_path(function.to_polydata(mv_anterior), loc_an.FindClosestPoint(mv_l_fuzzy),
                                     loc_an.FindClosestPoint(mv_r_fuzzy))
    path_po = function.dijkstra_path(function.to_polydata(mv_posterior), loc_po.FindClosestPoint(mv_l_fuzzy),
                                     loc_po.FindClosestPoint(mv_r_fuzzy))

    length_an = len(path_an)
    mv_an_middle = path_an[int(length_an * 0.5)]
    length_po = len(path_po)
    mv_po_middle = path_po[int(length_po * 0.5)]

    mv_l = mv_an_l
    mv_r = mv_an_r

    band = function.cut_a_band_from_model(model_polydata, center_mv, center_rpv, center_lpv, 10 * scale)

    ls_i_id = function.get_closest_point_id_from_polydata(model_polydata, ls_i)
    li_i_id = function.get_closest_point_id_from_polydata(model_polydata, li_i)
    rs_i_id = function.get_closest_point_id_from_polydata(model_polydata, rs_i)
    ri_i_id = function.get_closest_point_id_from_polydata(model_polydata, ri_i)

    lpv_path = function.dijkstra_path(model_polydata, ls_i_id, li_i_id)
    rpv_path = function.dijkstra_path(model_polydata, rs_i_id, ri_i_id)

    lpv_base_temp = function.multidim_intersect(vtk.util.numpy_support.vtk_to_numpy(band.GetPoints().GetData()),
                                                lpv_path)
    rpv_base_temp = function.multidim_intersect(vtk.util.numpy_support.vtk_to_numpy(band.GetPoints().GetData()),
                                                rpv_path)

    lpv_base_fuzzy = np.asarray(
        [np.mean(lpv_base_temp[:, 0]), np.mean(lpv_base_temp[:, 1]), np.mean(lpv_base_temp[:, 2])])
    rpv_base_fuzzy = np.asarray(
        [np.mean(rpv_base_temp[:, 0]), np.mean(rpv_base_temp[:, 1]), np.mean(rpv_base_temp[:, 2])])

    lpv_base_id = function.get_closest_point_id_from_polydata(band, lpv_base_fuzzy)
    rpv_base_id = function.get_closest_point_id_from_polydata(band, rpv_base_fuzzy)

    lpv_base = band.GetPoint(lpv_base_id)
    rpv_base = band.GetPoint(rpv_base_id)

    roof_path = function.dijkstra_path(band, lpv_base_id, rpv_base_id)

    length_roof = len(roof_path)
    roof_25 = roof_path[int(length_roof * 0.25)]
    roof_50 = roof_path[int(length_roof * 0.50)]
    roof_75 = roof_path[int(length_roof * 0.75)]

    septum_path = function.dijkstra_path(band, function.get_closest_point_id_from_polydata(band, mv_r), rpv_base_id)

    length_roof = len(septum_path)
    sep_25 = septum_path[int(length_roof * 0.25)]
    sep_50 = septum_path[int(length_roof * 0.50)]
    sep_75 = septum_path[int(length_roof * 0.75)]

    lateral_path = function.dijkstra_path(band, function.get_closest_point_id_from_polydata(band, mv_l), lpv_base_id)

    length_roof = len(lateral_path)
    lat_25 = lateral_path[int(length_roof * 0.25)]
    lat_50 = lateral_path[int(length_roof * 0.50)]
    lat_75 = lateral_path[int(length_roof * 0.75)]

    ls_o_id = function.get_closest_point_id_from_polydata(model_polydata, ls_o)
    rs_o_id = function.get_closest_point_id_from_polydata(model_polydata, rs_o)
    sup_path = function.dijkstra_path(model_polydata, ls_o_id, rs_o_id)

    li_o_id = function.get_closest_point_id_from_polydata(model_polydata, li_o)
    ri_o_id = function.get_closest_point_id_from_polydata(model_polydata, ri_o)
    inf_path = function.dijkstra_path(model_polydata, li_o_id, ri_o_id)

    length_sup = len(sup_path)
    p1 = sup_path[int(length_sup * 0.5)]
    p1_25 = sup_path[int(length_sup * 0.25)]
    p1_75 = sup_path[int(length_sup * 0.75)]

    length_inf = len(inf_path)
    p2 = inf_path[int(length_inf * 0.5)]
    p2_25 = inf_path[int(length_inf * 0.25)]
    p2_75 = inf_path[int(length_inf * 0.75)]

    #
    p1_id = function.get_closest_point_id_from_polydata(model_polydata, p1)
    roof_50_id = function.get_closest_point_id_from_polydata(model_polydata, roof_50)
    p1_roof_path = function.dijkstra_path(model_polydata, roof_50_id, p1_id)

    p2_id = function.get_closest_point_id_from_polydata(model_polydata, p2)
    p2_roof_path = function.dijkstra_path(model_polydata, roof_50_id, p2_id)

    #
    mv_an_middle_id = function.get_closest_point_id_from_polydata(model_polydata, mv_an_middle)
    mv_po_middle_id = function.get_closest_point_id_from_polydata(model_polydata, mv_po_middle)

    p1_mv_path = function.dijkstra_path(model_polydata, p1_id, mv_an_middle_id)
    p2_mv_path = function.dijkstra_path(model_polydata, p2_id, mv_po_middle_id)

    length_p1_mv_path = len(p1_mv_path)

    p1_mv_25 = p1_mv_path[int(length_p1_mv_path * 0.25)]
    p1_mv_50 = p1_mv_path[int(length_p1_mv_path * 0.50)]
    p1_mv_75 = p1_mv_path[int(length_p1_mv_path * 0.75)]

    length_p2_mv_path = len(p2_mv_path)
    p2_mv_33 = p2_mv_path[int(length_p2_mv_path * 0.33)]
    p2_mv_66 = p2_mv_path[int(length_p2_mv_path * 0.66)]

    p1_mv_sep_50_path = function.dijkstra_path(model_polydata,
                                               function.get_closest_point_id_from_polydata(model_polydata, p1_mv_50),
                                               function.get_closest_point_id_from_polydata(model_polydata, sep_50))

    p1_mv_sep_50 = p1_mv_sep_50_path[int(len(p1_mv_sep_50_path) * 0.5)]

    p1_mv_sep_75_path = function.dijkstra_path(model_polydata,
                                               function.get_closest_point_id_from_polydata(model_polydata, p1_mv_75),
                                               function.get_closest_point_id_from_polydata(model_polydata, sep_25))

    p1_mv_sep_75 = p1_mv_sep_75_path[int(len(p1_mv_sep_75_path) * 0.5)]

    p2_mv_sep_50_path = function.dijkstra_path(model_polydata,
                                               function.get_closest_point_id_from_polydata(model_polydata, p2_mv_33),
                                               function.get_closest_point_id_from_polydata(model_polydata, sep_50))

    p2_mv_sep_50 = p2_mv_sep_50_path[int(len(p2_mv_sep_50_path) * 0.5)]

    p2_mv_sep_75_path = function.dijkstra_path(model_polydata,
                                               function.get_closest_point_id_from_polydata(model_polydata, p2_mv_66),
                                               function.get_closest_point_id_from_polydata(model_polydata, sep_25))

    p2_mv_sep_75 = p2_mv_sep_75_path[int(len(p2_mv_sep_75_path) * 0.5)]

    # Extra pts:
    rspvo_p1_mv_50_path = function.dijkstra_path(model_polydata,
                                                 function.get_closest_point_id_from_polydata(model_polydata, rs_o),
                                                 function.get_closest_point_id_from_polydata(model_polydata, p1_mv_50))

    rspvo_p1_mv_30 = rspvo_p1_mv_50_path[int(len(rspvo_p1_mv_50_path) * 0.3)]

    rspvo_p1_mv_50 = rspvo_p1_mv_50_path[int(len(rspvo_p1_mv_50_path) * 0.5)]

    rspvo_p1_mv_70 = rspvo_p1_mv_50_path[int(len(rspvo_p1_mv_50_path) * 0.7)]

    rspvo_p1_mv_sep_50_path = function.dijkstra_path(model_polydata,
                                                     function.get_closest_point_id_from_polydata(model_polydata, rs_o),
                                                     function.get_closest_point_id_from_polydata(model_polydata,
                                                                                                 p1_mv_sep_50))

    rspvo_p1_mv_sep_30 = rspvo_p1_mv_sep_50_path[int(len(rspvo_p1_mv_sep_50_path) * 0.3)]

    rspvo_p1_mv_sep_50 = rspvo_p1_mv_sep_50_path[int(len(rspvo_p1_mv_sep_50_path) * 0.5)]

    rspvo_p1_mv_sep_70 = rspvo_p1_mv_sep_50_path[int(len(rspvo_p1_mv_sep_50_path) * 0.7)]

    # write in json format
    name_lst = ['lspv_o', 'lspv_i', 'lipv_i', 'lipv_o', 'rspv_o', 'rspv_i', 'ripv_i', 'ripv_o',
                'mv_l', 'mv_r', 'mv_an_middle', 'mv_po_middle',
                'lpv_base', 'roof_25', 'roof_50', 'roof_75', 'rpv_base',
                'lat_50',
                'laa',
                'sep_50', 'sep_25',
                'p1', 'p1_mv_50',
                'p1_mv_sep_50', 'p1_mv_sep_75', 'p2_mv_sep_50', 'p2_mv_sep_75',
                'rspvo_p1_mv_30', 'rspvo_p1_mv_50', 'rspvo_p1_mv_70', 'rspvo_p1_mv_sep_30', 'rspvo_p1_mv_sep_50',
                'rspvo_p1_mv_sep_70',
                'p2', 'p2_mv_33', 'p2_mv_66']

    coord_lst = np.vstack((ls_o, ls_i, li_i, li_o, rs_o, rs_i, ri_i, ri_o,
                           mv_l, mv_r, mv_an_middle, mv_po_middle,
                           lpv_base, roof_25, roof_50, roof_75, rpv_base,
                           lat_50,
                           df["LAA"],
                           sep_50, sep_25,
                           p1, p1_mv_50,
                           p1_mv_sep_50, p1_mv_sep_75, p2_mv_sep_50, p2_mv_sep_75,
                           rspvo_p1_mv_30, rspvo_p1_mv_50, rspvo_p1_mv_70, rspvo_p1_mv_sep_30, rspvo_p1_mv_sep_50,
                           rspvo_p1_mv_sep_70,
                           p2, p2_mv_33, p2_mv_66))

    json2 = '['
    for i in range(len(name_lst)):
        json2 = json2 + "{\"id\":\"" + "{}".format(name_lst[i]) + "\",\"coordinates\":[" + "{},{},{}".format(
            coord_lst[i][0], coord_lst[i][1], coord_lst[i][2]) + "]},"
    json2 = json2[:-1] + ']'

    f = open("{}/landmarks.json".format(mesh_dir), "w")
    f.write(json2)
    f.close()

    # txt file to open in Paraview to check
    f = open("{}/landmarks.txt".format(mesh_dir), "w")
    for i in range(len(name_lst)):
        f.write("{} {} {} {}\n".format(coord_lst[i][0], coord_lst[i][1], coord_lst[i][2], name_lst[i]))


def run():
    args = parser().parse_args()
    get_landmarks(args.mesh, args.prealigned, args.scale)


if __name__ == '__main__':
    run()
