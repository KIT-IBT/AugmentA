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
import pymeshlab
import pymeshfix
import pyvista as pv
import vtk
import argparse
from scipy.spatial import KDTree
from vtk.util import numpy_support
import os
import numpy as np
import pandas as pd
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

def resample_surf_mesh(meshname, target_mesh_resolution=0.4, find_apex_with_curv=0, scale=1, size=30, apex_id=-1):

    mesh_data = dict()

    ms = pymeshlab.MeshSet()

    ms.load_new_mesh('{}.obj'.format(meshname))

    ms.select_self_intersecting_faces()

    m = ms.current_mesh()

    if apex_id is not None and apex_id>-1:
        apex = m.vertex_matrix()[apex_id,:]

    self_intersecting_faces = m.selected_face_number()

    if self_intersecting_faces:
        reader = vtk.vtkOBJReader()
        reader.SetFileName('{}.obj'.format(meshname))
        reader.Update()

        boundaryEdges = vtk.vtkFeatureEdges()
        boundaryEdges.SetInputData(reader.GetOutput())
        boundaryEdges.BoundaryEdgesOn()
        boundaryEdges.FeatureEdgesOff()
        boundaryEdges.ManifoldEdgesOff()
        boundaryEdges.NonManifoldEdgesOff()
        boundaryEdges.Update()

        boundary_pts = vtk.util.numpy_support.vtk_to_numpy(boundaryEdges.GetOutput().GetPoints().GetData())

        # Clean the mesh from holes and self intersecting triangles
        meshin = pv.read('{}.obj'.format(meshname))
        meshfix = pymeshfix.MeshFix(meshin)
        meshfix.repair()
        vol = meshfix.mesh.volume
        
        pv.save_meshio('{}_meshfix.obj'.format(meshname),meshfix.mesh, "obj")

        reader = vtk.vtkOBJReader()
        reader.SetFileName('{}_meshfix.obj'.format(meshname))
        reader.Update()

        Mass = vtk.vtkMassProperties()
        Mass.SetInputData(reader.GetOutput())
        Mass.Update() 

        print("Volume = ", Mass.GetVolume())
        print("Surface = ", Mass.GetSurfaceArea())

        bd_ids = find_elements_around_path_within_radius(reader.GetOutput(), boundary_pts, 0.5*scale)

        tot_cells = set(list(range(reader.GetOutput().GetNumberOfCells())))
        cells_no_bd = tot_cells - bd_ids
        cell_ids_no_bd = vtk.vtkIdList()
        for i in cells_no_bd:
            cell_ids_no_bd.InsertNextId(i)
        extract = vtk.vtkExtractCells()
        extract.SetInputData(reader.GetOutput())
        extract.SetCellList(cell_ids_no_bd)
        extract.Update()
        
        geo_filter = vtk.vtkGeometryFilter()
        geo_filter.SetInputData(extract.GetOutput())
        geo_filter.Update()
        earth = geo_filter.GetOutput()
        
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(earth)
        cleaner.Update()

        connect = vtk.vtkConnectivityFilter()
        connect.SetInputConnection(cleaner.GetOutputPort())
        connect.SetExtractionModeToLargestRegion()
        connect.Update()

        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(connect.GetOutput())
        cleaner.Update()

        writer = vtk.vtkOBJWriter()
        writer.SetInputData(cleaner.GetOutput())
        writer.SetFileName('{}_cleaned.obj'.format(meshname))
        writer.Write()

        mesh_data["vol"]=[vol]

        ms = pymeshlab.MeshSet()

        ms.load_new_mesh('{}_cleaned.obj'.format(meshname))

    else:

        ms = pymeshlab.MeshSet()

        ms.load_new_mesh('{}.obj'.format(meshname))

    # compute the geometric measures of the current mesh
    # and save the results in the out_dict dictionary
    out_dict = ms.compute_geometric_measures()

    # get the average edge length from the dictionary
    avg_edge_length = out_dict['avg_edge_length']

    tgt_edge_length = target_mesh_resolution*scale

    loc_tgt_edge_length = target_mesh_resolution*scale
    it = 1
    print("Current resolution: {} mm".format(avg_edge_length/scale))
    print("Target resolution: {} mm".format(tgt_edge_length/scale))
    while avg_edge_length > tgt_edge_length*1.05 or avg_edge_length < tgt_edge_length*0.95 or it < 3 :
        
        ms.remeshing_isotropic_explicit_remeshing(iterations=5, targetlen=pymeshlab.AbsoluteValue(loc_tgt_edge_length))
        if it == 1:
            ms.laplacian_smooth()
        out_dict = ms.compute_geometric_measures()

        avg_edge_length = out_dict['avg_edge_length']
        print("Current resolution: {} mm".format(avg_edge_length/scale))
        if avg_edge_length > tgt_edge_length*1.05:
            loc_tgt_edge_length = tgt_edge_length*0.95
            print("New target resolution: {} mm".format(loc_tgt_edge_length/scale))
        elif avg_edge_length < tgt_edge_length*0.95:
            loc_tgt_edge_length = tgt_edge_length*1.05
            print("New target resolution: {} mm".format(loc_tgt_edge_length/scale))
        else:
            break
        it += 1
    
    mesh_data["surf"]=[out_dict['surface_area']]

    ms.save_current_mesh('{}_res.obj'.format(meshname),\
     save_vertex_color=False, save_vertex_normal=False, save_face_color=False, save_wedge_texcoord=False, save_wedge_normal=False)

    meshin = pv.read('{}_res.obj'.format(meshname))

    if find_apex_with_curv and apex_id==-1:
        if self_intersecting_faces:
            os.system("meshtool query curvature -msh={}_cleaned.obj -size={}".format(meshname, size*scale))
            curv = np.loadtxt('{}_cleaned.curv.dat'.format(meshname))
            mesh_curv = pv.read('{}_cleaned.obj'.format(meshname))
        else:
            os.system("meshtool query curvature -msh={}.obj -size={}".format(meshname, size*scale))
            curv = np.loadtxt('{}.curv.dat'.format(meshname))
            mesh_curv = pv.read('{}.obj'.format(meshname))

        apex = mesh_curv.points[np.argmax(curv),:]

        point_cloud = pv.PolyData(apex)

        p = pv.Plotter(notebook=False)

        p.add_mesh(meshin,color='r')
        p.add_mesh(point_cloud, color='w', point_size=30.*scale, render_points_as_spheres=True)
        p.enable_point_picking(meshin, use_mesh=True)
        p.add_text('Select the appendage apex and close the window',position='lower_left')

        p.show()

        if p.picked_point is None:
            print("Please pick a point as apex")
        else:
            apex = p.picked_point
            print("Apex coordinates: ",apex)

    elif find_apex_with_curv==0 and apex_id==-1:

        p = pv.Plotter(notebook=False)

        p.add_mesh(meshin,color='r')
        p.enable_point_picking(meshin, use_mesh=True)
        p.add_text('Select the appendage apex and close the window',position='lower_left')

        p.show()
        if p.picked_point is None:
            print("Please pick a point as apex")
        else:
            apex = p.picked_point
            print("Apex coordinates: ",apex)

    tree = KDTree(meshin.points.astype(np.double))
    dist, apex_id = tree.query(apex)

    mesh_data["LAA_id"] = [apex_id]

    fname = '{}_res_mesh_data.csv'.format(meshname)
    df = pd.DataFrame(mesh_data)
    df.to_csv(fname, float_format="%.2f", index=False)

def run():

    args = parser().parse_args()
    resample_surf_mesh(args.mesh, target_mesh_resolution, find_apex_with_curv, scale, size)

if __name__ == '__main__':
    run()
