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
import os, sys
import numpy as np
import pathlib
from glob import glob
import pandas as pd
import vtk
from vtk.util import numpy_support
import scipy.spatial as spatial
from vtk.numpy_interface import dataset_adapter as dsa
import datetime
from sklearn.cluster import KMeans
import argparse
from scipy.spatial import cKDTree

import pymeshfix
from pymeshfix import _meshfix
import pyvista as pv
import collections

pv.set_plot_theme('dark')

sys.path.append('../Atrial_LDRBM/Generate_Boundaries')
import extract_rings

vtk_version = vtk.vtkVersion.GetVTKSourceVersion().split()[-1].split('.')[0]


# def create_sphere(value):
#     radius = int(value)
#     sphere = pv.Sphere(center=center, radius=radius)
#     p.add_mesh(sphere, name='sphere', show_edges=True)
#     return
def parser():
    parser = argparse.ArgumentParser(description='Cut veins manually')
    parser.add_argument('--mesh',
                        type=str,
                        default="",
                        help='path to mesh')
    parser.add_argument('--atrium',
                        type=str,
                        default="",
                        help='write LA or RA')
    parser.add_argument('--size',
                        type=float,
                        default=30,
                        help='patch radius in mesh units for curvature estimation')
    parser.add_argument('--min_cutting_radius',
                        type=float,
                        default=7.5,
                        help='radius to cut veins/valves in mm')
    parser.add_argument('--max_cutting_radius',
                        type=float,
                        default=17.5,
                        help='radius to cut veins/valves in mm')
    parser.add_argument('--scale',
                        type=int,
                        default=1,
                        help='normal unit is mm, set scaling factor if different')
    parser.add_argument('--LAA',
                        type=str,
                        default="",
                        help='LAA apex point index, leave empty if no LA')
    parser.add_argument('--RAA',
                        type=str,
                        default="",
                        help='RAA apex point index, leave empty if no RA')
    parser.add_argument('--debug',
                        type=int,
                        default=0,
                        help='set to 1 to check the predicted location of the appendage apex')
    parser.add_argument('--MRI',
                        type=int,
                        default=0,
                        help='set to 1 if the input is an MRI segmentation')
    return parser


def open_orifices_manually(meshpath, atrium, MRI, scale=1, size=30, min_cutting_radius=7.5, max_cutting_radius=17.5,
                           LAA="", RAA="", debug=0):
    meshname = meshpath.split("/")[-1]
    full_path = meshpath[:-len(meshname)]

    # Clean the mesh from holes and self intersecting triangles
    meshin = pv.read(meshpath)
    meshfix = pymeshfix.MeshFix(meshin)
    meshfix.repair()
    meshfix.mesh.save("{}/{}_clean.vtk".format(full_path, atrium))
    pv.save_meshio("{}/{}_clean.obj".format(full_path, atrium), meshfix.mesh, "obj")

    mesh_with_data = smart_reader(meshpath)

    mesh_clean = smart_reader("{}/{}_clean.vtk".format(full_path, atrium))

    # Map point data to cleaned mesh
    mesh = point_array_mapper(mesh_with_data, mesh_clean, "all")

    if atrium == "LA":
        orifices = ['mitral valve', 'left inferior pulmonary vein', 'left superior pulmonary vein',
                    'right inferior pulmonary vein', 'right superior pulmonary vein']
    else:
        orifices = ['tricuspid valve', 'inferior vena cava', 'superior vena cava', 'coronary sinus']

    for r in orifices:
        picked_pt = None
        while picked_pt is None:
            p = pv.Plotter(notebook=False)
            p.add_mesh(meshfix.mesh, 'r')
            p.add_text('Select the center of the {} and close the window to cut, otherwise just close'.format(r),
                       position='lower_left')
            p.enable_point_picking(meshfix.mesh, use_mesh=True)
            p.show()

            picked_pt = p.picked_point
            p.close()
        if r == 'mitral valve' or r == 'tricuspid valve':
            el_to_del_tot = find_elements_within_radius(mesh, picked_pt, max_cutting_radius)
        else:
            el_to_del_tot = find_elements_within_radius(mesh, picked_pt, min_cutting_radius)

        model_new_el = vtk.vtkIdList()
        cell_id_all = list(range(mesh.GetNumberOfCells()))
        el_diff = list(set(cell_id_all).difference(el_to_del_tot))

        for var in el_diff:
            model_new_el.InsertNextId(var)

        extract = vtk.vtkExtractCells()
        extract.SetInputData(mesh)
        extract.SetCellList(model_new_el)
        extract.Update()

        geo_filter = vtk.vtkGeometryFilter()
        geo_filter.SetInputConnection(extract.GetOutputPort())
        geo_filter.Update()

        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputConnection(geo_filter.GetOutputPort())
        cleaner.Update()

        mesh = cleaner.GetOutput()

    model = extract_largest_region(mesh)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName("{}/{}_cutted.vtk".format(full_path, atrium))
    writer.SetInputData(model)
    writer.Write()

    p = pv.Plotter(notebook=False)
    mesh_from_vtk = pv.PolyData("{}/{}_cutted.vtk".format(full_path, atrium))
    p.add_mesh(mesh_from_vtk, 'r')
    p.add_text('Select the atrial appendage apex', position='lower_left')
    p.enable_point_picking(meshfix.mesh, use_mesh=True)

    p.show()

    if p.picked_point is not None:
        apex = p.picked_point

    p.close()
    model = smart_reader("{}/{}_cutted.vtk".format(full_path, atrium))
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(model)
    loc.BuildLocator()
    apex_id = loc.FindClosestPoint(apex)
    if atrium == "LA":
        LAA = apex_id
    elif atrium == "RA":
        RAA = apex_id

    meshpath = "{}/{}_cutted".format(full_path, atrium)
    extract_rings.run(["--mesh", meshpath, "--LAA", str(LAA), "--RAA", str(RAA)])

    return apex_id


def run():
    args = parser().parse_args()

    apex_id = open_orifices_manually(args.mesh, args.atrium, args.MRI, args.scale, args.size, args.min_cutting_radius,
                                     args.max_cutting_radius, args.LAA, args.RAA, args.debug)


def smart_reader(path):
    extension = str(path).split(".")[-1]

    if extension == "vtk":
        data_checker = vtk.vtkDataSetReader()
        data_checker.SetFileName(str(path))
        data_checker.Update()

        if data_checker.IsFilePolyData():
            reader = vtk.vtkPolyDataReader()
        elif data_checker.IsFileUnstructuredGrid():
            reader = vtk.vtkUnstructuredGridReader()

    elif extension == "vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif extension == "vtu":
        reader = vtk.vtkXMLUnstructuredGridReader()
    elif extension == "obj":
        reader = vtk.vtkOBJReader()
    else:
        print("No polydata or unstructured grid")

    reader.SetFileName(str(path))
    reader.Update()
    output = reader.GetOutput()

    return output


def vtk_thr(model, mode, points_cells, array, thr1, thr2="None"):
    thresh = vtk.vtkThreshold()
    thresh.SetInputData(model)
    if mode == 0:
        thresh.ThresholdByUpper(thr1)
    elif mode == 1:
        thresh.ThresholdByLower(thr1)
    elif mode == 2:
        if int(vtk_version) >= 9:
            thresh.ThresholdBetween(thr1, thr2)
        else:
            thresh.ThresholdByUpper(thr1)
            thresh.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_" + points_cells, array)
            thresh.Update()
            thr = thresh.GetOutput()
            thresh = vtk.vtkThreshold()
            thresh.SetInputData(thr)
            thresh.ThresholdByLower(thr2)
    thresh.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_" + points_cells, array)
    thresh.Update()

    output = thresh.GetOutput()

    return output


def find_elements_within_radius(mesh, points_data, radius):
    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()

    mesh_id_list = vtk.vtkIdList()
    locator.FindPointsWithinRadius(radius, points_data, mesh_id_list)

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


def extract_largest_region(mesh):
    connect = vtk.vtkConnectivityFilter()
    connect.SetInputData(mesh)
    connect.SetExtractionModeToLargestRegion()
    connect.Update()
    surface = connect.GetOutput()

    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(surface)
    geo_filter.Update()
    surface = geo_filter.GetOutput()

    cln = vtk.vtkCleanPolyData()
    cln.SetInputData(surface)
    cln.Update()
    res = cln.GetOutput()

    return res


def point_array_mapper(mesh1, mesh2, idat):
    pts1 = vtk.util.numpy_support.vtk_to_numpy(mesh1.GetPoints().GetData())
    pts2 = vtk.util.numpy_support.vtk_to_numpy(mesh2.GetPoints().GetData())

    tree = cKDTree(pts1)

    dd, ii = tree.query(pts2, n_jobs=-1)

    meshNew = dsa.WrapDataObject(mesh2)
    if idat == "all":
        for i in range(mesh1.GetPointData().GetNumberOfArrays()):
            data = vtk.util.numpy_support.vtk_to_numpy(
                mesh1.GetPointData().GetArray(mesh1.GetPointData().GetArrayName(i)))
            if isinstance(data[0], collections.Sized):
                data2 = np.zeros((len(pts2), len(data[0])), dtype=data.dtype)
            else:
                data2 = np.zeros((len(pts2),), dtype=data.dtype)

            data2 = data[ii]
            data2 = np.where(np.isnan(data2), 10000, data2)

            meshNew.PointData.append(data2, mesh1.GetPointData().GetArrayName(i))
    else:
        data = vtk.util.numpy_support.vtk_to_numpy(mesh1.GetPointData().GetArray(idat))
        if isinstance(data[0], collections.Sized):
            data2 = np.zeros((len(pts2), len(data[0])), dtype=data.dtype)
        else:
            data2 = np.zeros((len(pts2),), dtype=data.dtype)

        data2 = data[ii]
        meshNew.PointData.append(data2, idat)

    return meshNew.VTKObject


if __name__ == '__main__':
    run()
