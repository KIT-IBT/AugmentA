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

import pymeshfix
import pyvista as pv
import vtk

import vtk_opencarp_helper_methods.AugmentA_methods.vtk_operations
from Atrial_LDRBM.Generate_Boundaries import extract_rings
from vtk_opencarp_helper_methods.AugmentA_methods.point_selection import pick_point
from vtk_opencarp_helper_methods.vtk_methods.exporting import vtk_polydata_writer
from vtk_opencarp_helper_methods.vtk_methods.finder import find_closest_point
from vtk_opencarp_helper_methods.vtk_methods.helper_methods import cut_mesh_with_radius
from vtk_opencarp_helper_methods.vtk_methods.mapper import point_array_mapper

pv.set_plot_theme('dark')

from vtk_opencarp_helper_methods.vtk_methods.reader import smart_reader
from vtk_opencarp_helper_methods.AugmentA_methods.vtk_operations import extract_largest_region

vtk_version = vtk.vtkVersion.GetVTKSourceVersion().split()[-1].split('.')[0]


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


def open_orifices_manually(meshpath, atrium, MRI, scale=1, size=30, vessels_cutting_radius=7.5,
                           valve_cutting_radius=17.5, LAA="", RAA="", debug=0):
    meshname = meshpath.split("/")[-1]
    full_path = meshpath[:-len(meshname)]

    # Clean the mesh from holes and self intersecting triangles
    meshin = pv.read(meshpath)
    meshfix = pymeshfix.MeshFix(meshin)
    meshfix.repair()
    meshfix.mesh.save(f"{full_path}/{atrium}_clean.vtk")
    pv.save_meshio(f"{full_path}/{atrium}_clean.obj", meshfix.mesh, "obj")

    mesh_with_data = smart_reader(meshpath)

    mesh_clean = smart_reader(f"{full_path}/{atrium}_clean.vtk")

    # Map point data to cleaned mesh
    mesh = point_array_mapper(mesh_with_data, mesh_clean, "all")

    if atrium == "LA":
        orifices = ['mitral valve', 'left inferior pulmonary vein', 'left superior pulmonary vein',
                    'right inferior pulmonary vein', 'right superior pulmonary vein']
    else:
        orifices = ['tricuspid valve', 'inferior vena cava', 'superior vena cava', 'coronary sinus']

    for r in orifices:
        picked_pt = pick_point(meshfix.mesh, f"center of the {r}")
        if r == 'mitral valve' or r == 'tricuspid valve':
            selected_radius = valve_cutting_radius
        else:
            selected_radius = vessels_cutting_radius

        mesh = cut_mesh_with_radius(mesh, picked_pt, selected_radius)

    model = extract_largest_region(mesh)

    vtk_polydata_writer(f"{full_path}/{atrium}_cutted.vtk", model)

    mesh_from_vtk = pv.PolyData(f"{full_path}/{atrium}_cutted.vtk")

    apex = pick_point(mesh_from_vtk, "atrial appendage apex")

    model = smart_reader(f"{full_path}/{atrium}_cutted.vtk")

    apex_id = find_closest_point(model, apex)
    if atrium == "LA":
        LAA = apex_id
    elif atrium == "RA":
        RAA = apex_id

    meshpath = f"{full_path}/{atrium}_cutted.vtk"

    command = ["--mesh", meshpath, "--LAA", str(LAA), "--RAA", str(RAA)]
    print(f"extract rings with:{command}")
    extract_rings.run(command)

    return apex_id


def run():
    args = parser().parse_args()

    apex_id = open_orifices_manually(args.mesh, args.atrium, args.MRI, args.scale, args.size, args.min_cutting_radius,
                                     args.max_cutting_radius, args.LAA, args.RAA, args.debug)


def vtk_thr(model, mode, points_cells, array, thr1, thr2="None"):
    return vtk_opencarp_helper_methods.AugmentA_methods.vtk_operations.vtk_thr(model, mode, points_cells, array, thr1,
                                                                               thr2)


if __name__ == '__main__':
    run()
