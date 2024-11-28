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

import numpy as np
import pandas as pd
import pymeshfix
import pymeshlab
import pyvista as pv
import vtk
from scipy.spatial import cKDTree

from Atrial_LDRBM.LDRBM.Fiber_RA.Methods_RA import find_elements_around_path_within_radius
from vtk_opencarp_helper_methods.AugmentA_methods.point_selection import pick_point_with_preselection, pick_point
from vtk_opencarp_helper_methods.vtk_methods.converters import vtk_to_numpy
from vtk_opencarp_helper_methods.vtk_methods.exporting import vtk_obj_writer
from vtk_opencarp_helper_methods.vtk_methods.filters import apply_vtk_geom_filter, clean_polydata, get_cells_with_ids, \
    get_feature_edges
from vtk_opencarp_helper_methods.vtk_methods.init_objects import init_connectivity_filter, ExtractionModes

pv.set_plot_theme('dark')
vtk_version = vtk.vtkVersion.GetVTKSourceVersion().split()[-1].split('.')[0]


def parser():
    parser = argparse.ArgumentParser(description='Generate boundaries.')
    parser.add_argument('--mesh',
                        type=str,
                        default="",
                        help='path to meshname')
    parser.add_argument('--scale',
                        type=int,
                        default=1,
                        help='normal unit is mm, set scaling factor if different')
    parser.add_argument('--size',
                        type=float,
                        default=30,
                        help='patch radius in mesh units for curvature estimation')
    parser.add_argument('--target_mesh_resolution',
                        type=float,
                        default=0.4,
                        help='target mesh resolution in mm')
    parser.add_argument('--find_apex_with_curv',
                        type=int,
                        default=0,
                        help='set to 1 to predict location of the appendage apex using max curvature, else pick manually')

    return parser


def resample_surf_mesh(meshname, target_mesh_resolution=0.4, find_apex_with_curv=0, scale=1, size=30, apex_id=-1,
                       atrium='LA'):
    mesh_data = dict()

    ms = pymeshlab.MeshSet()

    ms.load_new_mesh(f'{meshname}.obj')
    # ms.apply_filter('turn_into_a_pure_triangular_mesh')  # if polygonal mesh
    # ms.save_current_mesh('{}.obj'.format(meshname))

    ms.compute_selection_by_self_intersections_per_face()

    m = ms.current_mesh()

    if apex_id > -1:
        apex = m.vertex_matrix()[apex_id, :]

    self_intersecting_faces = m.selected_face_number()

    if self_intersecting_faces:
        reader = vtk.vtkOBJReader()
        reader.SetFileName(f'{meshname}.obj')
        reader.Update()

        boundary_edges = get_feature_edges(reader.GetOutput(), boundary_edges_on=True, feature_edges_on=False,
                                           manifold_edges_on=False,
                                           non_manifold_edges_on=False)

        boundary_pts = vtk_to_numpy(boundary_edges.GetPoints().GetData())

        # Clean the mesh from holes and self intersecting triangles
        meshin = pv.read(f'{meshname}.obj')
        meshfix = pymeshfix.MeshFix(meshin)  # Be careful with biatrial geometries as it might delete one chamber
        meshfix.repair()
        vol = meshfix.mesh.volume

        pv.save_meshio(f'{meshname}_meshfix.obj', meshfix.mesh, "obj")

        reader = vtk.vtkOBJReader()
        reader.SetFileName(f'{meshname}_meshfix.obj')
        reader.Update()

        Mass = vtk.vtkMassProperties()
        Mass.SetInputData(reader.GetOutput())
        Mass.Update()

        print("Volume = ", Mass.GetVolume())
        print("Surface = ", Mass.GetSurfaceArea())

        bd_ids = find_elements_around_path_within_radius(reader.GetOutput(), boundary_pts, 0.5 * scale)

        tot_cells = set(list(range(reader.GetOutput().GetNumberOfCells())))
        cells_no_bd = tot_cells - bd_ids

        earth = apply_vtk_geom_filter(get_cells_with_ids(reader.GetOutput(), cells_no_bd))

        connect = init_connectivity_filter(clean_polydata(earth), ExtractionModes.LARGEST_REGION)

        vtk_obj_writer(f'{meshname}_cleaned.obj', clean_polydata(connect.GetOutput()))
        mesh_data["vol"] = [vol]

        ms = pymeshlab.MeshSet()

        ms.load_new_mesh(f'{meshname}_cleaned.obj')

    else:

        ms = pymeshlab.MeshSet()

        ms.load_new_mesh(f'{meshname}.obj')

    # compute the geometric measures of the current mesh
    # and save the results in the out_dict dictionary
    out_dict = ms.get_geometric_measures()

    # get the average edge length from the dictionary
    avg_edge_length = out_dict['avg_edge_length']

    tgt_edge_length = target_mesh_resolution * scale

    loc_tgt_edge_length = target_mesh_resolution * scale
    it = 1
    print(f"Current resolution: {avg_edge_length / scale} mm")
    print(f"Target resolution: {tgt_edge_length / scale} mm")
    while avg_edge_length > tgt_edge_length * 1.05 or avg_edge_length < tgt_edge_length * 0.95 or it < 3:

        ms.meshing_isotropic_explicit_remeshing(iterations=5, targetlen=pymeshlab.PureValue(loc_tgt_edge_length))
        if it == 1:
            ms.apply_coord_laplacian_smoothing()
        out_dict = ms.get_geometric_measures()

        avg_edge_length = out_dict['avg_edge_length']
        print(f"Current resolution: {avg_edge_length / scale} mm")
        if avg_edge_length > tgt_edge_length * 1.05:
            loc_tgt_edge_length = tgt_edge_length * 0.95
            print(f"New target resolution: {loc_tgt_edge_length / scale} mm")
        elif avg_edge_length < tgt_edge_length * 0.95:
            loc_tgt_edge_length = tgt_edge_length * 1.05
            print(f"New target resolution: {loc_tgt_edge_length / scale} mm")
        else:
            break
        it += 1

    mesh_data["avg_edge_length"] = [out_dict['avg_edge_length']]
    mesh_data["surf"] = [out_dict['surface_area']]

    # Better to save as .ply
    ms.save_current_mesh(f'{meshname}_res.ply', save_vertex_color=False, save_vertex_normal=False,
                         save_face_color=False, save_wedge_texcoord=False, save_wedge_normal=False)
    meshin = pv.read(f'{meshname}_res.ply')

    if find_apex_with_curv and apex_id == -1:
        if self_intersecting_faces:
            os.system(f"meshtool query curvature -msh={meshname}_cleaned.obj -size={size * scale}")
            curv = np.loadtxt(f'{meshname}_cleaned.curv.dat')
            mesh_curv = pv.read(f'{meshname}_cleaned.obj')
        else:
            os.system(f"meshtool query curvature -msh={meshname}.obj -size={size * scale}")
            curv = np.loadtxt(f'{meshname}.curv.dat')
            mesh_curv = pv.read(f'{meshname}.obj')

        apex = mesh_curv.points[np.argmax(curv), :]
        apex = pick_point_with_preselection(meshin, "appendage apex", apex)
        print("Apex coordinates: ", apex)

    elif find_apex_with_curv == 0 and apex_id == -1:
        apex = pick_point(meshin, "appendage apex")
        print("Apex coordinates: ", apex)

    tree = cKDTree(meshin.points.astype(np.double))
    dist, apex_id = tree.query(apex)

    if atrium == 'LA_RA':
        mesh_data["LAA_id"] = [apex_id]  # change accordingly
    else:
        mesh_data[f"{atrium}A_id"] = [apex_id]  # change accordingly

    if atrium == 'LA_RA':
        atrium = 'RA'

        apex = pick_point(meshin, "RA appendage apex")
        print("Apex coordinates: ", apex)

        tree = cKDTree(meshin.points.astype(np.double))
        dist, apex_id = tree.query(apex)

        mesh_data[f"{atrium}A_id"] = [apex_id]  # change accordingly

        reader = vtk.vtkPLYReader()
        reader.SetFileName(f'{meshname}_res.ply')
        reader.Update()

        Mass = vtk.vtkMassProperties()
        Mass.SetInputData(reader.GetOutput())
        Mass.Update()

        print("Volume = ", Mass.GetVolume())
        print("Surface = ", Mass.GetSurfaceArea())
        mesh_data["vol_bi"] = Mass.GetVolume()  # Biatrial volume

    fname = f'{meshname}_res_mesh_data.csv'
    df = pd.DataFrame(mesh_data)
    df.to_csv(fname, float_format="%.2f", index=False)


def run():
    args = parser().parse_args()
    resample_surf_mesh(args.mesh, args.target_mesh_resolution, args.find_apex_with_curv, args.scale, args.size)


if __name__ == '__main__':
    run()
