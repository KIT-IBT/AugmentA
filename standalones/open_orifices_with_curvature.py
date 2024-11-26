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
import collections
import os
import sys
import warnings

import numpy as np
import pymeshfix
import pyvista as pv
import vtk
from scipy.spatial import cKDTree
from vtk.numpy_interface import dataset_adapter as dsa

import vtk_opencarp_helper_methods.AugmentA_methods.vtk_operations
from standalones.open_orifices_manually import open_orifices_manually
from vtk_opencarp_helper_methods.vtk_methods import filters
from vtk_opencarp_helper_methods.vtk_methods.converters import vtk_to_numpy
from vtk_opencarp_helper_methods.vtk_methods.exporting import vtk_unstructured_grid_writer, vtk_polydata_writer
from vtk_opencarp_helper_methods.vtk_methods.helper_methods import get_maximum_distance_of_points, cut_mesh_with_radius, \
    cut_elements_from_mesh
from vtk_opencarp_helper_methods.vtk_methods.reader import smart_reader

pv.set_plot_theme('dark')

sys.path.append('./Atrial_LDRBM/Generate_Boundaries')
from extract_rings import label_atrial_orifices

vtk_version = vtk.vtkVersion.GetVTKSourceVersion().split()[-1].split('.')[0]


def parser():
    parser = argparse.ArgumentParser(description='Cut veins detected as high curvature areas')
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


def open_orifices_with_curvature(meshpath, atrium, MRI, scale=1, size=30, min_cutting_radius=7.5,
                                 max_cutting_radius=17.5, LAA="", RAA="", debug=0):
    meshname = meshpath.split("/")[-1]
    full_path = meshpath[:-len(meshname)]

    # Clean the mesh from holes and self intersecting triangles
    meshin = pv.read(meshpath)
    meshfix = pymeshfix.MeshFix(meshin)
    meshfix.repair()
    meshfix.mesh.save(f"{full_path}/{atrium}_clean.vtk")
    pv.save_meshio(f"{full_path}/{atrium}_clean.obj", meshfix.mesh, "obj")

    # Compute surface curvature
    os.system(f"meshtool query curvature -msh={full_path}/{atrium}_clean.obj -size={size * scale}")

    # Verify if the mesh curvature is not nan

    mesh_with_data = smart_reader(meshpath)

    curv = np.loadtxt(f'{full_path}/{atrium}_clean.curv.dat')

    mesh_clean = smart_reader(f"{full_path}/{atrium}_clean.vtk")

    # Map point data to cleaned mesh
    mesh = point_array_mapper(mesh_with_data, mesh_clean, "all")

    model = dsa.WrapDataObject(mesh)

    model.PointData.append(curv, "curv")

    model = model.VTKObject

    if debug:
        vtk_polydata_writer(f"{full_path}/{atrium}_clean_with_curv.vtk", model)
    apex = None
    if not MRI:

        valve = vtk_thr(model, 0, "POINTS", "valve", 0.5)
        if valve.GetPoints() is None or valve.GetNumberOfPoints() == 0:
            # do manually orifice opening when automatically does not find valves
            warnings.warn("No points for valve found. Should default to manual assignment")
            return open_orifices_manually(meshpath, atrium, MRI, scale, size, min_cutting_radius, max_cutting_radius,
                                          LAA, RAA,
                                          debug)
        valve = extract_largest_region(valve)

        centerOfMassFilter = vtk.vtkCenterOfMass()
        centerOfMassFilter.SetInputData(valve)
        centerOfMassFilter.SetUseScalarsAsWeights(False)
        centerOfMassFilter.Update()

        valve_center = np.array(centerOfMassFilter.GetCenter())

        max_dist = get_maximum_distance_of_points(valve, valve_center)

        if max_dist > max_cutting_radius * 2:
            print(f"Valve bigger than {max_cutting_radius * 2} cm")

        model = cut_mesh_with_radius(model, valve_center, max_cutting_radius)

    else:
        valve = vtk_thr(model, 0, "POINTS", "curv", 0.05)
        valve = extract_largest_region(valve)
        if valve.GetPoints() is None or valve.GetNumberOfPoints() == 0:
            # do manually orifice opening when automatically does not find valves
            warnings.warn("No points for valve found. Should default to manual assignment")
            return open_orifices_manually(meshpath, atrium, MRI, scale, size, min_cutting_radius, max_cutting_radius,
                                          LAA, RAA,
                                          debug)

        if debug and atrium == 'RA':
            writer_vtk(valve, f"{full_path}/{atrium}_clean_with_curv_" + "valve.vtk")

        center_of_mass = filters.get_center_of_mass(valve, False)

        valve_center = np.array(center_of_mass)

        max_dist = get_maximum_distance_of_points(valve, valve_center)

        # Cutting valve with fixed radius to ensure that it is the biggest ring
        model = cut_mesh_with_radius(model, valve_center, max_cutting_radius)

    # model = smart_reader("{}/{}_valve.vtk".format(full_path, atrium))
    cellid = vtk.vtkIdFilter()
    cellid.CellIdsOn()
    cellid.SetInputData(model)
    cellid.PointIdsOn()
    if int(vtk_version) >= 9:
        cellid.SetPointIdsArrayName('Ids')
        cellid.SetCellIdsArrayName('Ids')
    else:
        cellid.SetIdsArrayName('Ids')
    cellid.Update()

    model = cellid.GetOutput()

    vtk_polydata_writer(f"{full_path}/{atrium}_curv.vtk", model, True)

    curv = vtk_to_numpy(model.GetPointData().GetArray('curv'))

    Gl_pt_id = list(vtk_to_numpy(model.GetPointData().GetArray('Ids')))
    Gl_cell_id = list(vtk_to_numpy(model.GetCellData().GetArray('Ids')))

    if not MRI:
        low_v = vtk_thr(model, 1, "POINTS", "bi", 0.5)

        pts_low_v = set(list(vtk_to_numpy(low_v.GetPointData().GetArray('Ids'))))

        high_v = vtk_thr(model, 0, "POINTS", "bi", 0.5001)

    high_c = vtk_thr(model, 0, "POINTS", "curv", np.median(curv) * 1.15)  # (np.min(curv)+np.max(curv))/2)

    vtk_unstructured_grid_writer(f"{full_path}/{atrium}_h_curv.vtk", high_c, True)

    connect = vtk.vtkConnectivityFilter()
    connect.SetInputData(high_c)
    connect.SetExtractionModeToAllRegions()
    connect.Update()
    num = connect.GetNumberOfExtractedRegions()

    connect = vtk.vtkConnectivityFilter()
    connect.SetInputData(high_c)
    connect.SetExtractionModeToSpecifiedRegions()

    rings = []

    el_to_del_tot = set()
    old_max = 0

    if MRI:
        cc = pv.PolyData(valve_center)
        p = pv.Plotter(notebook=False)
        p.add_mesh(meshfix.mesh, 'r')
        p.add_text('Select the appendage apex and close the window', position='lower_left')
        p.add_mesh(cc, color='w', point_size=30., render_points_as_spheres=True)
        p.enable_point_picking(meshfix.mesh, use_picker=True)

        p.show()

        apex = p.picked_point
        p.close()
        loc = vtk.vtkPointLocator()
        loc.SetDataSet(model)
        loc.BuildLocator()
        apex_id = loc.FindClosestPoint(apex)

        if atrium == "LA":
            LAA = apex_id
        elif atrium == "RA":
            RAA = apex_id
    else:
        transeptal_punture_id = -1
        p = pv.Plotter(notebook=False)
        mesh_from_vtk = pv.PolyData(f"{full_path}/{atrium}_clean.vtk")
        p.add_mesh(mesh_from_vtk, 'r')
        p.add_text('Select the transeptal punture and close the window', position='lower_left')
        p.enable_point_picking(meshfix.mesh, use_picker=True)

        p.show()

        if p.picked_point is not None:
            loc = vtk.vtkPointLocator()
            loc.SetDataSet(model)
            loc.BuildLocator()
            transeptal_punture_id = vtk_to_numpy(model.GetPointData().GetArray('Ids'))[
                loc.FindClosestPoint(p.picked_point)]
        p.close()

    for i in range(num):
        connect.AddSpecifiedRegion(i)
        connect.Update()
        surface = connect.GetOutput()

        # Clean unused points
        geo_filter = vtk.vtkGeometryFilter()
        geo_filter.SetInputData(surface)
        geo_filter.Update()
        surface = geo_filter.GetOutput()

        cln = vtk.vtkCleanPolyData()
        cln.SetInputData(surface)
        cln.Update()
        surface = cln.GetOutput()

        pt_high_c = list(vtk_to_numpy(surface.GetPointData().GetArray('Ids')))
        curv_s = vtk_to_numpy(surface.GetPointData().GetArray('curv'))

        if not MRI:
            if transeptal_punture_id not in pt_high_c:
                if len(set(pt_high_c).intersection(pts_low_v)) > 0:  # the region is both high curvature and low voltage
                    pt_max_curv = np.asarray(model.GetPoint(Gl_pt_id.index(pt_high_c[np.argmax(curv_s)])))
                    el_low_vol = set()
                    connect2 = vtk.vtkConnectivityFilter()
                    connect2.SetInputData(low_v)
                    connect2.SetExtractionModeToAllRegions()
                    connect2.Update()
                    num2 = connect2.GetNumberOfExtractedRegions()

                    connect2.SetExtractionModeToSpecifiedRegions()

                    for ii in range(num2):
                        connect2.AddSpecifiedRegion(ii)
                        connect2.Update()
                        surface2 = connect2.GetOutput()

                        # Clean unused points
                        geo_filter = vtk.vtkGeometryFilter()
                        geo_filter.SetInputData(surface2)
                        geo_filter.Update()
                        surface2 = geo_filter.GetOutput()

                        cln = vtk.vtkCleanPolyData()
                        cln.SetInputData(surface2)
                        cln.Update()
                        surface2 = cln.GetOutput()
                        pt_surf_2 = list(vtk_to_numpy(surface2.GetPointData().GetArray('Ids')))
                        if len(set(pt_high_c).intersection(pt_surf_2)) > 0:

                            for el in vtk_to_numpy(surface2.GetCellData().GetArray('Ids')):
                                el_low_vol.add(Gl_cell_id.index(el))

                        connect2.DeleteSpecifiedRegion(ii)
                        connect2.Update()

                    model_new_el = vtk.vtkIdList()

                    for var in el_low_vol:
                        model_new_el.InsertNextId(var)

                    extract = vtk.vtkExtractCells()
                    extract.SetInputData(model)
                    extract.SetCellList(model_new_el)
                    extract.Update()

                    geo_filter = vtk.vtkGeometryFilter()
                    geo_filter.SetInputConnection(extract.GetOutputPort())
                    geo_filter.Update()

                    cleaner = vtk.vtkCleanPolyData()
                    cleaner.SetInputConnection(geo_filter.GetOutputPort())
                    cleaner.Update()

                    loc_low_V = cleaner.GetOutput()  # local low voltage area

                    loc_low_V = extract_largest_region(loc_low_V)

                    max_dist = get_maximum_distance_of_points(loc_low_V, pt_max_curv)

                    el_to_del = find_elements_within_radius(model, pt_max_curv, min_cutting_radius * 2 * scale)

                    el_to_del_tot = el_to_del_tot.union(set(el_to_del))


                else:  # Possible appendage

                    if np.max(curv_s) > old_max:  # The max curvature without low voltage should be the appendage
                        old_max = np.max(curv_s)
                        apex = np.asarray(model.GetPoint(Gl_pt_id.index(pt_high_c[np.argmax(curv_s)])))
        else:
            if not apex_id in pt_high_c:
                for el in vtk_to_numpy(surface.GetCellData().GetArray('Ids')):
                    el_to_del_tot.add(Gl_cell_id.index(el))

        connect.DeleteSpecifiedRegion(i)
        connect.Update()

    model = cut_elements_from_mesh(model, el_to_del_tot)

    model = extract_largest_region(model)

    vtk_polydata_writer(f"{full_path}/{atrium}_cutted.vtk", model)
    if debug:
        if apex is not None:
            point_cloud = pv.PolyData(apex)

            p = pv.Plotter(notebook=False)
            mesh_from_vtk = pv.PolyData(f"{full_path}/{atrium}_cutted.vtk")
            p.add_mesh(mesh_from_vtk, 'r')
            p.add_mesh(point_cloud, color='w', point_size=30., render_points_as_spheres=True)
            p.enable_point_picking(meshfix.mesh, use_picker=True)
            p.add_text('Select the appendage apex and close the window', position='lower_left')
            p.show()

            if p.picked_point is not None:
                apex = p.picked_point
            p.close()
        else:
            p = pv.Plotter(notebook=False)
            mesh_from_vtk = pv.PolyData(f"{full_path}/{atrium}_cutted.vtk")
            p.add_mesh(mesh_from_vtk, 'r')
            p.enable_point_picking(meshfix.mesh, use_picker=True)
            p.add_text('Select the appendage apex and close the window', position='lower_left')
            p.show()

            if p.picked_point is not None:
                apex = p.picked_point
            p.close()

    model = smart_reader(f"{full_path}/{atrium}_cutted.vtk")
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(model)
    loc.BuildLocator()
    apex_id = loc.FindClosestPoint(apex)
    if atrium == "LA":
        LAA = apex_id
    elif atrium == "RA":
        RAA = apex_id

    label_atrial_orifices(f"{full_path}/{atrium}_cutted.vtk", LAA, RAA)

    return apex_id


def run():
    args = parser().parse_args()

    apex_id = open_orifices_with_curvature(args.mesh, args.atrium, args.MRI, args.scale, args.size,
                                           args.min_cutting_radius, args.max_cutting_radius, args.LAA, args.RAA,
                                           args.debug)


def vtk_thr(model, mode, points_cells, array, thr1, thr2="None"):
    return vtk_opencarp_helper_methods.AugmentA_methods.vtk_operations.vtk_thr(model, mode, points_cells, array, thr1,
                                                                               thr2)


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
    pts1 = vtk_to_numpy(mesh1.GetPoints().GetData())
    pts2 = vtk_to_numpy(mesh2.GetPoints().GetData())

    tree = cKDTree(pts1)

    dd, ii = tree.query(pts2, workers=-1)

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

            meshNew.PointData.append(data2, mesh1.GetPointData().GetArrayName(i))
    else:
        data = vtk_to_numpy(mesh1.GetPointData().GetArray(idat))
        if isinstance(data[0], collections.abc.Sized):
            data2 = np.zeros((len(pts2), len(data[0])), dtype=data.dtype)
        else:
            data2 = np.zeros((len(pts2),), dtype=data.dtype)

        data2 = data[ii]
        meshNew.PointData.append(data2, idat)

    return meshNew.VTKObject


def create_pts(array_points, array_name, mesh_dir):
    f = open(f"{mesh_dir}{array_name}.pts", "w")
    f.write("0 0 0\n")
    for i in range(len(array_points)):
        f.write(f"{array_points[i][0]} {array_points[i][1]} {array_points[i][2]}\n")
    f.close()


def to_polydata(mesh):
    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(mesh)
    geo_filter.Update()
    polydata = geo_filter.GetOutput()
    return polydata


def writer_vtk(mesh, filename):
    vtk_polydata_writer(filename, to_polydata(mesh))


if __name__ == '__main__':
    run()
