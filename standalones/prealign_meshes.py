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
import os
import numpy as np
from glob import glob
import pandas as pd
import vtk
from vtk.util import numpy_support
from vtk.numpy_interface import dataset_adapter as dsa
import datetime
import transformations as tf

import argparse

from vtk_opencarp_helper_methods.vtk_methods.exporting import vtk_polydata_writer


def parser():
    parser = argparse.ArgumentParser(description='Prealign meshes using landmarks.')
    parser.add_argument('--mesh1',
                        type=str,
                        default="mwk03",
                        help='path to meshname to prealign')
    parser.add_argument('--case',
                        type=str,
                        default="LA",
                        help='choose between LA, RA or both')
    parser.add_argument('--mesh2',
                        type=str,
                        default="meanshape",
                        help='path to meshname to prealign mesh1 to')
    parser.add_argument('--scale',
                        type=int,
                        default=0,
                        help='set 1 to scale mesh1 to mesh2')

    return parser


def prealign_meshes(mesh1_name, mesh2_name, case="LA", scale=0):
    # Landmarks to use to prealign both meshes
    if case == "both":
        names = ["MV", "RSPV", "LSPV", "RIPV", "LIPV", "TV", "SVC", "IVC"]
        meshLA = vtkreader(f"{mesh1_name}_surf/LA_boundaries_tagged")
        meshRA = vtkreader(f"{mesh1_name}_surf/RA_boundaries_tagged")
        appendFilter = vtk.vtkAppendFilter()
        appendFilter.AddInputData(meshLA)
        appendFilter.AddInputData(meshRA)
        appendFilter.Update()
        extract_surf = vtk.vtkGeometryFilter()
        extract_surf.SetInputData(appendFilter.GetOutput())
        extract_surf.Update()

        mesh1 = extract_surf.GetOutput()

    elif case == "LA":
        names = ["MV", "RSPV", "LSPV", "RIPV", "LIPV"]  # Prealign MRI
        mesh1 = vtkreader(f"{mesh1_name}_surf/LA_boundaries_tagged")
    else:
        names = ["TV", "SVC", "IVC"]
        mesh1 = vtkreader(f"{mesh1_name}_surf/RA_boundaries_tagged")

    df_tot = pd.read_csv(f"{mesh1_name}_surf/rings_centroids.csv")
    A_tot = df_tot.to_numpy()

    df1 = pd.read_csv(f"{mesh1_name}_surf/rings_centroids.csv", usecols=names)
    df2 = pd.read_csv(f"{mesh2_name}_surf/rings_centroids.csv", usecols=names)
    df2 = df2[df1.columns]

    A = df1.to_numpy()
    B = df2.to_numpy()

    if scale:
        M = tf.transformations.superimposition_matrix(A, B, scale=True, usesvd=True)
    else:
        M = tf.transformations.superimposition_matrix(A, B, scale=False, usesvd=True)
    ret_R = M[:3, :3]
    ret_t = M[:3, 3].reshape(3, 1)

    new_pts = (ret_R @ A) + ret_t
    new_centroids = (ret_R @ A_tot) + ret_t

    # Write the prealigned landmarks as .json files
    df_new = pd.DataFrame(data=new_pts, columns=df1.columns)
    df_new_tot = pd.DataFrame(data=new_centroids, columns=df_tot.columns)

    df_new_tot.to_csv(f"{mesh1_name}_surf/rings_centroids_prealigned.csv", index=False)
    json1 = '['
    json2 = '['
    for i in df_new.columns:
        json1 = json1 + "{\"id\":\"" + f"{i}" + "\",\"coordinates\":[" + "{},{},{}".format(df_new[i][0],
                                                                                           df_new[i][1],
                                                                                           df_new[i][2]) + "]},"
    json1 = json1[:-1] + ']'
    for i in df2.columns:
        json2 = json2 + "{\"id\":\"" + f"{i}" + "\",\"coordinates\":[" + "{},{},{}".format(df2[i][0], df2[i][1],
                                                                                           df2[i][2]) + "]},"
    json2 = json2[:-1] + ']'

    f = open(f"{mesh1_name}_surf/prealigned_landmarks.json", "w")
    f.write(json1)
    f.close()
    f = open(f"{mesh2_name}_surf/landmarks_to_prealign.json", "w")
    f.write(json2)
    f.close()

    M_l = list(M.flatten())

    transformFilter = vtk.vtkTransform()
    transformFilter.SetMatrix(M_l)

    transform_poly = vtk.vtkTransformPolyDataFilter()
    transform_poly.SetInputData(mesh1)
    transform_poly.SetTransform(transformFilter)
    transform_poly.Update()

    writer = vtk.vtkSTLWriter()
    writer.SetInputConnection(transform_poly.GetOutputPort())
    if case == "both":
        writer.SetFileName(f'{mesh1_name}_surf/LA_RA_prealigned.stl')
    elif case == "LA":
        writer.SetFileName(f'{mesh1_name}_surf/LA_prealigned.stl')
    else:
        writer.SetFileName(f'{mesh1_name}_surf/RA_prealigned.stl')
    writer.Write()

    meshNew = dsa.WrapDataObject(transform_poly.GetOutput())
    if case == "both":
        vtk_polydata_writer(f'{mesh1_name}_surf/LA_RA_prealigned.vtk', meshNew.VTKObject)
    elif case == "LA":
        vtk_polydata_writer(f'{mesh1_name}_surf/LA_prealigned.vtk', meshNew.VTKObject)
    else:
        vtk_polydata_writer(f'{mesh1_name}_surf/RA_prealigned.vtk', meshNew.VTKObject)



def run():
    args = parser().parse_args()

    prealign_meshes(args.mesh1, args.mesh2, args.case, args.scale)


def vtkreader(meshname):
    reader = vtk.vtkPolyDataReader()

    reader.SetFileName(f'{meshname}.vtk')
    reader.Update()

    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputConnection(reader.GetOutputPort())
    geo_filter.Update()

    polydata = geo_filter.GetOutput()
    return polydata


if __name__ == '__main__':
    run()
