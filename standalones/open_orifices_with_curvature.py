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
import sys
import warnings

import numpy as np
import pymeshfix
import pyvista as pv
import vtk
from vtk.numpy_interface import dataset_adapter as dsa

from standalones.open_orifices_manually import open_orifices_manually
from vtk_openCARP_methods_ibt.AugmentA_methods.point_selection import pick_point_with_preselection, pick_point
from vtk_openCARP_methods_ibt.AugmentA_methods.vtk_operations import extract_largest_region, vtk_thr
from vtk_openCARP_methods_ibt.vtk_methods import filters
from vtk_openCARP_methods_ibt.vtk_methods.converters import vtk_to_numpy
from vtk_openCARP_methods_ibt.vtk_methods.exporting import vtk_unstructured_grid_writer, vtk_polydata_writer
from vtk_openCARP_methods_ibt.vtk_methods.filters import apply_vtk_geom_filter, get_vtk_geom_filter_port, \
    clean_polydata, generate_ids, get_cells_with_ids, get_center_of_mass
from vtk_openCARP_methods_ibt.vtk_methods.finder import find_closest_point
from vtk_openCARP_methods_ibt.vtk_methods.helper_methods import get_maximum_distance_of_points, cut_mesh_with_radius, \
    cut_elements_from_mesh, find_elements_within_radius
from vtk_openCARP_methods_ibt.vtk_methods.init_objects import init_connectivity_filter, ExtractionModes
from vtk_openCARP_methods_ibt.vtk_methods.mapper import point_array_mapper
from vtk_openCARP_methods_ibt.vtk_methods.reader import smart_reader

pv.set_plot_theme('dark')

sys.path.append('./Atrial_LDRBM/Generate_Boundaries')
from Atrial_LDRBM.Generate_Boundaries.extract_rings import label_atrial_orifices

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

        valve_center = np.array(get_center_of_mass(valve, False))

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
            vtk_polydata_writer(valve, f"{full_path}/{atrium}_clean_with_curv_" + "valve.vtk")

        center_of_mass = filters.get_center_of_mass(valve, False)

        valve_center = np.array(center_of_mass)

        max_dist = get_maximum_distance_of_points(valve, valve_center)

        # Cutting valve with fixed radius to ensure that it is the biggest ring
        model = cut_mesh_with_radius(model, valve_center, max_cutting_radius)

    model = generate_ids(model, "Ids", "Ids")

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

    connect = init_connectivity_filter(high_c, ExtractionModes.ALL_REGIONS)
    num = connect.GetNumberOfExtractedRegions()

    connect = init_connectivity_filter(high_c, ExtractionModes.SPECIFIED_REGIONS)

    rings = []

    el_to_del_tot = set()
    old_max = 0

    if MRI:
        cc = pv.PolyData(valve_center)

        apex = pick_point_with_preselection(meshfix.mesh, "appendage apex", cc)

        apex_id = find_closest_point(model, apex)

        if atrium == "LA":
            LAA = apex_id
        elif atrium == "RA":
            RAA = apex_id
    else:
        transeptal_punture_id = -1

        mesh_from_vtk = pv.PolyData(f"{full_path}/{atrium}_clean.vtk")

        picked_point = pick_point(mesh_from_vtk, "transeptal puncture")
        if picked_point is not None:
            transeptal_punture_id = vtk_to_numpy(model.GetPointData().GetArray('Ids'))[
                find_closest_point(model, picked_point)]

    for i in range(num):
        connect.AddSpecifiedRegion(i)
        connect.Update()
        surface = connect.GetOutput()

        # Clean unused points
        surface = apply_vtk_geom_filter(surface)
        surface = clean_polydata(surface)

        pt_high_c = list(vtk_to_numpy(surface.GetPointData().GetArray('Ids')))
        curv_s = vtk_to_numpy(surface.GetPointData().GetArray('curv'))

        if not MRI:
            if transeptal_punture_id not in pt_high_c:
                if len(set(pt_high_c).intersection(pts_low_v)) > 0:  # the region is both high curvature and low voltage
                    pt_max_curv = np.asarray(model.GetPoint(Gl_pt_id.index(pt_high_c[np.argmax(curv_s)])))
                    el_low_vol = set()

                    connect2 = init_connectivity_filter(low_v, ExtractionModes.ALL_REGIONS)
                    num2 = connect2.GetNumberOfExtractedRegions()

                    connect2.SetExtractionModeToSpecifiedRegions()

                    for ii in range(num2):
                        connect2.AddSpecifiedRegion(ii)
                        connect2.Update()
                        surface2 = connect2.GetOutput()

                        # Clean unused points
                        surface2 = apply_vtk_geom_filter(surface2)
                        surface2 = clean_polydata(surface2)
                        pt_surf_2 = list(vtk_to_numpy(surface2.GetPointData().GetArray('Ids')))
                        if len(set(pt_high_c).intersection(pt_surf_2)) > 0:

                            for el in vtk_to_numpy(surface2.GetCellData().GetArray('Ids')):
                                el_low_vol.add(Gl_cell_id.index(el))

                        connect2.DeleteSpecifiedRegion(ii)
                        connect2.Update()

                    geo_port, _geo_filter = get_vtk_geom_filter_port(get_cells_with_ids(model, el_low_vol))

                    loc_low_V = clean_polydata(geo_port, input_is_connection=True)  # local low voltage area

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
        mesh_from_vtk = pv.PolyData(f"{full_path}/{atrium}_cutted.vtk")
        apex = pick_point_with_preselection(mesh_from_vtk, "appendage apex", apex)

    model = smart_reader(f"{full_path}/{atrium}_cutted.vtk")

    apex_id = find_closest_point(model, apex)
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

if __name__ == '__main__':
    run()
