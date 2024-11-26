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
import vtk
from scipy.spatial import cKDTree
from vtk.numpy_interface import dataset_adapter as dsa

from vtk_opencarp_helper_methods.AugmentA_methods.vtk_operations import vtk_thr
from vtk_opencarp_helper_methods.vtk_methods.exporting import vtk_xml_unstructured_grid_writer
from vtk_opencarp_helper_methods.vtk_methods.reader import smart_reader

vtk_version = vtk.vtkVersion.GetVTKSourceVersion().split()[-1].split('.')[0]


def low_vol_LAT(args, path):
    # Read mesh
    meshname = f'{args.mesh}_fibers/result_LA/LA_bilayer_with_fiber_with_data_um'  # in um
    model = smart_reader(f'{meshname}.vtk')

    bilayer_n_cells = model.GetNumberOfCells()

    # Transfer lat and bipolar voltage from points to elements
    pt_cell = vtk.vtkPointDataToCellData()
    pt_cell.SetInputData(model)
    pt_cell.AddPointDataArray("bi")
    pt_cell.AddPointDataArray("lat")
    pt_cell.PassPointDataOn()
    pt_cell.CategoricalDataOff()
    pt_cell.ProcessAllArraysOff()
    pt_cell.Update()

    model = pt_cell.GetOutput()

    # Create Points and Cells ids
    cellid = vtk.vtkIdFilter()
    cellid.CellIdsOn()
    cellid.SetInputData(model)
    cellid.PointIdsOn()
    cellid.FieldDataOn()
    if int(vtk_version) >= 9:
        cellid.SetPointIdsArrayName('Global_ids')
        cellid.SetCellIdsArrayName('Global_ids')
    else:
        cellid.SetIdsArrayName('Global_ids')
    cellid.Update()

    model = cellid.GetOutput()

    # Compute elements centroids
    filter_cell_centers = vtk.vtkCellCenters()
    filter_cell_centers.SetInputData(model)
    filter_cell_centers.Update()
    centroids = vtk.util.numpy_support.vtk_to_numpy(filter_cell_centers.GetOutput().GetPoints().GetData())

    # Low voltage in the model
    low_vol = vtk_thr(model, 1, "CELLS", "bi", args.low_vol_thr)
    low_vol_ids = vtk.util.numpy_support.vtk_to_numpy(low_vol.GetCellData().GetArray('Global_ids')).astype(int)

    if args.debug:

        meshbasename = args.mesh.split("/")[-1]
        debug_dir = f'{args.init_state_dir}/{meshbasename}/debug'
        try:
            os.makedirs(debug_dir)
        except OSError:
            print(f"Creation of the directory {debug_dir} failed")
        else:
            print(f"Successfully created the directory {debug_dir} ")

        vtk_xml_unstructured_grid_writer(f'{debug_dir}/low_vol.vtu', low_vol)
    # Endo

    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(model)
    geo_filter.Update()

    endo = vtk_thr(geo_filter.GetOutput(), 1, "CELLS", "elemTag", 10)

    if args.debug:
        vtk_xml_unstructured_grid_writer(f'{debug_dir}/endo.vtu', endo)
    # Get point LAT map in endocardium
    LAT_endo = vtk.util.numpy_support.vtk_to_numpy(endo.GetPointData().GetArray('lat'))
    endo_ids = vtk.util.numpy_support.vtk_to_numpy(endo.GetCellData().GetArray('Global_ids')).astype(int)
    endo_pts = vtk.util.numpy_support.vtk_to_numpy(endo.GetPoints().GetData())

    # Get elements LAT map in endocardium
    LAT_map = vtk.util.numpy_support.vtk_to_numpy(endo.GetCellData().GetArray('lat'))

    # Extract "healthy" high voltage endocardium
    not_low_volt_endo = vtk_thr(endo, 0, "POINTS", "bi", 0.5 + 0.01)
    LAT_not_low_volt = vtk.util.numpy_support.vtk_to_numpy(not_low_volt_endo.GetPointData().GetArray('lat'))
    not_low_volt_endo_pts = vtk.util.numpy_support.vtk_to_numpy(not_low_volt_endo.GetPoints().GetData())
    not_low_volt_ids = vtk.util.numpy_support.vtk_to_numpy(
        not_low_volt_endo.GetPointData().GetArray('Global_ids')).astype(int)

    if args.debug:
        vtk_xml_unstructured_grid_writer(f'{debug_dir}/not_low_volt_endo.vtu', not_low_volt_endo)
    # Extract LA wall from SSM to be sure that no veins or LAA is included when selecting the earliest activated point
    if args.SSM_fitting:
        LA_wall = smart_reader(args.SSM_basename + '/LA_wall.vtk')
        LA_wall_pt_ids = vtk.util.numpy_support.vtk_to_numpy(LA_wall.GetPointData().GetArray('PointIds'))

        # See create_SSM_instance standalone to create LA_fit.obj
        reader = vtk.vtkOBJReader()
        reader.SetFileName(f'{args.mesh}/LA_fit.obj')
        reader.Update()
        LA_fit = reader.GetOutput()

        LA_fit_wall_pts = vtk.util.numpy_support.vtk_to_numpy(LA_fit.GetPoints().GetData())[LA_wall_pt_ids, :] * 1000

        tree = cKDTree(not_low_volt_endo_pts)

        dd, ii = tree.query(LA_fit_wall_pts)
    else:
        tree = cKDTree(not_low_volt_endo_pts)
        dd, ii = tree.query(endo_pts)

    healthy_endo = not_low_volt_endo  # vtk_thr(not_low_volt_endo,0,"POINTS","CV_mag", args.low_CV_thr)
    LAT_healthy = vtk.util.numpy_support.vtk_to_numpy(healthy_endo.GetPointData().GetArray('lat'))
    healthy_ids = vtk.util.numpy_support.vtk_to_numpy(healthy_endo.GetPointData().GetArray('Global_ids')).astype(int)

    if args.max_LAT_pt == "max":

        # Selecting the location of earliest/latest activation as the very first activated map point 
        # or electrogram point can be error-prone since it can happen that there is a single point which was annotated too early/late
        # Latest activated point is the center of mass of the 97.5 percentile of LAT

        perc_975 = np.percentile(LAT_not_low_volt[ii], 97.5)

        ids = np.where(LAT_not_low_volt[ii] >= perc_975)[0]

        max_pt = np.mean(not_low_volt_endo_pts[ii][ids], axis=0)

        loc = vtk.vtkPointLocator()
        loc.SetDataSet(not_low_volt_endo)
        loc.BuildLocator()
        args.max_LAT_id = loc.FindClosestPoint(max_pt)
        max_pt = np.array(not_low_volt_endo.GetPoint(args.max_LAT_id))
        args.LaAT = np.max(LAT_not_low_volt)

        # Earliest activated point is the center of mass of the 2.5 percentile of LAT
        perc_25 = np.percentile(LAT_not_low_volt[ii], 2.5)

        ids = np.where(LAT_not_low_volt[ii] <= perc_25)[0]

        stim_pt = np.mean(not_low_volt_endo_pts[ii][ids], axis=0)

        loc = vtk.vtkPointLocator()
        loc.SetDataSet(not_low_volt_endo)
        loc.BuildLocator()
        stim_pt_id = loc.FindClosestPoint(stim_pt)
        stim_pt = np.array(not_low_volt_endo.GetPoint(stim_pt_id))
        min_LAT = np.min(LAT_not_low_volt[ii])

        # Comp
        fit_LAT = []
        steps = list(np.arange(min_LAT, args.LaAT, args.step))
        for i in range(1, len(steps)):
            fit_LAT.append(steps[i] - min_LAT)

        fit_LAT.append(args.LaAT - min_LAT)

    # Before proceeding with the iterative fitting of the clinical LAT, we detect the nodes 
    # with an earlier activation than the neighboring vertices and mark them as wrong annotations
    el_to_clean, el_border = areas_to_clean(endo, args, min_LAT, stim_pt)

    return bilayer_n_cells, low_vol_ids, endo, endo_ids, centroids, LAT_map - min_LAT, min_LAT, el_to_clean, el_border, stim_pt, fit_LAT, healthy_endo


def areas_to_clean(endo, args, min_LAT, stim_pt):
    # Really fine LAT bands with time step of 5 ms
    steps = list(np.arange(min_LAT, args.LaAT, 5))
    steps.append(args.LaAT)
    el_to_clean = []
    el_border = []
    tot_el_to_clean = np.array([], dtype=int)

    meshNew = dsa.WrapDataObject(endo)
    print("Starting creation of bands ... ")
    for i in range(1, len(steps)):

        # Extract LAT band from min LAT to step i and remove all areas not connected with EAP
        band = vtk_thr(endo, 2, "CELLS", "lat", min_LAT, steps[i])

        b_ids = vtk.util.numpy_support.vtk_to_numpy(band.GetCellData().GetArray('Global_ids')).astype(int)

        connect = vtk.vtkConnectivityFilter()
        connect.SetInputData(band)
        connect.SetExtractionModeToClosestPointRegion()
        connect.SetClosestPoint(stim_pt)
        connect.Update()
        largest_band = connect.GetOutput()

        l_b_ids = vtk.util.numpy_support.vtk_to_numpy(largest_band.GetCellData().GetArray('Global_ids')).astype(int)

        if len(b_ids) > len(l_b_ids):
            cell_diff = set()

            # Find all elements which are not belonging to the clean band
            el_diff = np.setdiff1d(b_ids, l_b_ids)
            b_ids = list(b_ids)
            for el in el_diff:
                cell_diff.add(b_ids.index(el))

            model_new_el = vtk.vtkIdList()

            for var in cell_diff:
                model_new_el.InsertNextId(var)

            extract = vtk.vtkExtractCells()
            extract.SetInputData(band)
            extract.SetCellList(model_new_el)
            extract.Update()

            geo_filter = vtk.vtkGeometryFilter()
            geo_filter.SetInputConnection(extract.GetOutputPort())
            geo_filter.Update()

            cleaner = vtk.vtkCleanPolyData()
            cleaner.SetInputConnection(geo_filter.GetOutputPort())
            cleaner.Update()

            # Mesh of all elements which are not belonging to the clean band
            el_removed = cleaner.GetOutput()

            # Compute centroids of all elements which are not belonging to the clean band
            filter_cell_centers = vtk.vtkCellCenters()
            filter_cell_centers.SetInputData(largest_band)
            filter_cell_centers.Update()
            centroids2 = filter_cell_centers.GetOutput().GetPoints()
            pts = vtk.util.numpy_support.vtk_to_numpy(centroids2.GetData())

            tree = cKDTree(pts)

            connect = vtk.vtkConnectivityFilter()
            connect.SetInputData(el_removed)
            connect.SetExtractionModeToSpecifiedRegions()
            connect.Update()
            num = connect.GetNumberOfExtractedRegions()
            for n in range(num):
                connect.AddSpecifiedRegion(n)
                connect.Update()

                geo_filter = vtk.vtkGeometryFilter()
                geo_filter.SetInputConnection(connect.GetOutputPort())
                geo_filter.Update()
                # Clean unused points
                cln = vtk.vtkCleanPolyData()
                cln.SetInputConnection(geo_filter.GetOutputPort())
                cln.Update()
                surface = cln.GetOutput()

                filter_cell_centers = vtk.vtkCellCenters()
                filter_cell_centers.SetInputData(surface)
                filter_cell_centers.Update()
                centroids1 = filter_cell_centers.GetOutput().GetPoints()
                centroids1_array = vtk.util.numpy_support.vtk_to_numpy(centroids1.GetData())

                dd, ii = tree.query(centroids1_array, n_jobs=-1)

                # Set as elements to clean only if they are at least 1 um away from the biggest band
                if np.min(dd) > 1:
                    loc_el_to_clean = vtk.util.numpy_support.vtk_to_numpy(
                        surface.GetCellData().GetArray('Global_ids')).astype(int)

                    tot_el_to_clean = np.union1d(tot_el_to_clean, loc_el_to_clean)

                # delete added region id
                connect.DeleteSpecifiedRegion(n)
                connect.Update()

    print("Bands to clean ready ... ")

    idss = np.zeros((endo.GetNumberOfCells(),))
    idss[tot_el_to_clean] = 1

    meshNew.CellData.append(idss, "idss")

    endo_clean = vtk_thr(meshNew.VTKObject, 1, "CELLS", "idss", 0)

    el_cleaned = vtk.util.numpy_support.vtk_to_numpy(endo_clean.GetCellData().GetArray('Global_ids')).astype(int)

    endo_to_interpolate = vtk_thr(meshNew.VTKObject, 0, "CELLS", "idss", 1)

    filter_cell_centers = vtk.vtkCellCenters()
    filter_cell_centers.SetInputData(endo_clean)
    filter_cell_centers.Update()
    centroids2 = filter_cell_centers.GetOutput().GetPoints()
    pts = vtk.util.numpy_support.vtk_to_numpy(centroids2.GetData())

    tree = cKDTree(pts)

    # Find elements at the boundary of the areas to clean, which are gonna be used for the fitting of the conductivities
    connect = vtk.vtkConnectivityFilter()
    connect.SetInputData(endo_to_interpolate)
    connect.SetExtractionModeToSpecifiedRegions()
    connect.Update()
    num = connect.GetNumberOfExtractedRegions()
    for n in range(num):
        connect.AddSpecifiedRegion(n)
        connect.Update()

        geo_filter = vtk.vtkGeometryFilter()
        geo_filter.SetInputConnection(connect.GetOutputPort())
        geo_filter.Update()
        # Clean unused points
        cln = vtk.vtkCleanPolyData()
        cln.SetInputConnection(geo_filter.GetOutputPort())
        cln.Update()
        surface = cln.GetOutput()

        loc_el_to_clean = vtk.util.numpy_support.vtk_to_numpy(surface.GetCellData().GetArray('Global_ids')).astype(int)

        el_to_clean.append(np.unique(loc_el_to_clean))

        filter_cell_centers = vtk.vtkCellCenters()
        filter_cell_centers.SetInputData(surface)
        filter_cell_centers.Update()
        centroids1 = filter_cell_centers.GetOutput().GetPoints()
        centroids1_array = vtk.util.numpy_support.vtk_to_numpy(centroids1.GetData())

        dd, ii = tree.query(centroids1_array, n_jobs=-1)  # Find distance to endo_clean pts

        el_border.append(np.unique(el_cleaned[ii]))  # Give id of the closest point to the endo_clean

        # delete added region id
        connect.DeleteSpecifiedRegion(n)
        connect.Update()

    if args.debug:

        meshbasename = args.mesh.split("/")[-1]
        debug_dir = f'{args.init_state_dir}/{meshbasename}/debug'
        try:
            os.makedirs(debug_dir)
        except OSError:
            print(f"Creation of the directory {debug_dir} failed")
        else:
            print(f"Successfully created the directory {debug_dir} ")

        el_border_array = np.concatenate(el_border)  # convert to linear array
        border = np.zeros((endo.GetNumberOfCells(),))
        border[el_border_array] = 1
        meshNew.CellData.append(border, "border")

        vtk_xml_unstructured_grid_writer(f'{debug_dir}/endo_with_clean_tag.vtu', meshNew.VTKObject)
    return el_to_clean, el_border


def create_regele(endo, args):
    # Low voltage in the model
    low_vol = vtk_thr(endo, 1, "CELLS", "bi", args.low_vol_thr)
    low_vol_ids = vtk.util.numpy_support.vtk_to_numpy(low_vol.GetCellData().GetArray('Global_ids')).astype(int)
    not_low_volt_endo = vtk_thr(endo, 0, "POINTS", "bi", 0.5 + 0.01)

    f_slow_conductive = f"{args.init_state_dir}/{args.mesh.split('/')[-1]}/elems_slow_conductive"
    file = open(f_slow_conductive + '.regele', 'w')
    file.write(str(len(low_vol_ids)) + '\n')
    for i in low_vol_ids:
        file.write(str(i) + '\n')
    file.close()

    print('Regele file done ...')


def low_CV(model, low_CV_thr, meshfold):
    low_CV = vtk_thr(model, 1, "CELLS", "CV_mag", low_CV_thr)

    low_CV_ids = vtk.util.numpy_support.vtk_to_numpy(low_CV.GetCellData().GetArray('Global_ids')).astype(int)

    low_CV_c = vtk.util.numpy_support.vtk_to_numpy(low_CV.GetCellData().GetArray('CV_mag')) / 1000

    low_sigma = low_CV_c ** 2

    sigma = np.ones((model.GetNumberOfCells(),))

    sigma[low_CV_ids] = 0.6 ** 2  # low_sigma

    f = open(meshfold + '/low_CV.dat', 'w')
    for i in sigma:
        f.write(f"{i:.4f}\n")
    f.close()

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


def get_EAP(path_mod, path_fib):
    model = smart_reader(path_mod)
    mod_fib = smart_reader(path_fib)

    cellid = vtk.vtkIdFilter()
    cellid.CellIdsOn()
    cellid.SetInputData(mod_fib)
    cellid.PointIdsOn()
    cellid.FieldDataOn()
    if int(vtk_version) >= 9:
        cellid.SetPointIdsArrayName('Global_ids')
        cellid.SetCellIdsArrayName('Global_ids')
    else:
        cellid.SetIdsArrayName('Global_ids')
    cellid.Update()

    mod_fib = cellid.GetOutput()
    LA_MV = vtk_thr(mod_fib, 1, "CELLS", "elemTag", 2)
    LAT_map = vtk.util.numpy_support.vtk_to_numpy(model.GetPointData().GetArray('LAT'))

    LA_MV_ids = vtk.util.numpy_support.vtk_to_numpy(LA_MV.GetPointData().GetArray('Global_ids')).astype(int)

    print(LA_MV_ids[np.argmin(LAT_map[LA_MV_ids])])
    stim_pt = model.GetPoint(LA_MV_ids[np.argmin(LAT_map[LA_MV_ids])])

    return stim_pt


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
