#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 14:50:34 2021

@author: luca
"""
import os,sys
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

sys.path.append('Atrial_LDRBM/Generate_Boundaries')
import extract_rings

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

def open_orifices_with_curvature(meshpath, atrium, MRI, scale=1, size=30, min_cutting_radius=7.5, max_cutting_radius=17.5,  LAA="", RAA="", debug=0):

    meshname = meshpath.split("/")[-1]
    full_path = meshpath[:-len(meshname)]

    # Clean the mesh from holes and self intersecting triangles
    meshin = pv.read(meshpath)
    meshfix = pymeshfix.MeshFix(meshin)
    meshfix.repair()
    meshfix.mesh.save("{}/{}_clean.vtk".format(full_path, atrium))
    pv.save_meshio("{}/{}_clean.obj".format(full_path, atrium),meshfix.mesh, "obj")

    # Compute surface curvature
    os.system("meshtool query curvature -msh={}/{}_clean.obj -size={}".format(full_path, atrium, size*scale))

    mesh_with_data = smart_reader(meshpath)
    
    curv = np.loadtxt('{}/{}_clean.curv.dat'.format(full_path, atrium))

    mesh_clean = smart_reader("{}/{}_clean.vtk".format(full_path, atrium))
    
    # Map point data to cleaned mesh
    mesh = point_array_mapper(mesh_with_data, mesh_clean, "all")
    
    model = dsa.WrapDataObject(mesh)

    model.PointData.append(curv, "curv")

    model = model.VTKObject

    apex = None
    if not MRI:

        valve = vtk_thr(model,0,"POINTS","valve",0.5)
        valve = extract_largest_region(valve)

        centerOfMassFilter = vtk.vtkCenterOfMass()
        centerOfMassFilter.SetInputData(valve)
        centerOfMassFilter.SetUseScalarsAsWeights(False)
        centerOfMassFilter.Update()

        valve_center = np.array(centerOfMassFilter.GetCenter())

        valve_pts = vtk.util.numpy_support.vtk_to_numpy(valve.GetPoints().GetData())
        max_dist = 0
        for l in range(len(valve_pts)):
            if np.sqrt(np.sum((valve_center-valve_pts[l])**2, axis=0)) > max_dist:
                max_dist = np.sqrt(np.sum((valve_center-valve_pts[l])**2, axis=0))
        
        if max_dist > max_cutting_radius*2:
            print("Valve bigger than {} cm".format(max_cutting_radius*2))
        el_to_del_tot = find_elements_within_radius(model,valve_center,max_cutting_radius)

        model_new_el = vtk.vtkIdList()
        cell_id_all = list(range(model.GetNumberOfCells()))
        el_diff =  list(set(cell_id_all).difference(el_to_del_tot))
        
        for var in el_diff:
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

        model = cleaner.GetOutput()

    else:
        valve = vtk_thr(model,0,"POINTS","valve",0.5)
        valve = extract_largest_region(valve)

        centerOfMassFilter = vtk.vtkCenterOfMass()
        centerOfMassFilter.SetInputData(valve)
        centerOfMassFilter.SetUseScalarsAsWeights(False)
        centerOfMassFilter.Update()

        valve_center = np.array(centerOfMassFilter.GetCenter())

        valve_pts = vtk.util.numpy_support.vtk_to_numpy(valve.GetPoints().GetData())
        max_dist = 0
        for l in range(len(valve_pts)):
            if np.sqrt(np.sum((valve_center-valve_pts[l])**2, axis=0)) > max_dist:
                max_dist = np.sqrt(np.sum((valve_center-valve_pts[l])**2, axis=0))
        
        # Cutting valve with fixed radius to ensure that it is the biggest ring
        el_to_del_tot = find_elements_within_radius(model,valve_center,max_cutting_radius)

        model_new_el = vtk.vtkIdList()
        cell_id_all = list(range(model.GetNumberOfCells()))
        el_diff =  list(set(cell_id_all).difference(el_to_del_tot))
        
        for var in el_diff:
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

        model = cleaner.GetOutput()

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
    
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName("{}/{}_curv.vtk".format(full_path, atrium))
    writer.SetInputData(model)
    writer.SetFileTypeToBinary()
    writer.Write()

    curv = vtk.util.numpy_support.vtk_to_numpy(model.GetPointData().GetArray('curv'))
    
    Gl_pt_id = list(vtk.util.numpy_support.vtk_to_numpy(model.GetPointData().GetArray('Ids')))
    Gl_cell_id = list(vtk.util.numpy_support.vtk_to_numpy(model.GetCellData().GetArray('Ids')))
    
    if not MRI:
        low_v = vtk_thr(model,1,"POINTS","bi",0.5)

        pts_low_v = set(list(vtk.util.numpy_support.vtk_to_numpy(low_v.GetPointData().GetArray('Ids'))))

        high_v = vtk_thr(model,0,"POINTS","bi",0.5001)

    high_c = vtk_thr(model,0,"POINTS","curv",np.median(curv)*1.15)#(np.min(curv)+np.max(curv))/2)

    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName("{}/{}_h_curv.vtk".format(full_path, atrium))
    writer.SetInputData(high_c)
    writer.SetFileTypeToBinary()
    writer.Write()

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
        p.add_mesh(meshfix.mesh)
        p.add_text('Select the appendage apex and close the window',position='lower_left')
        p.add_mesh(cc, color='blue', point_size=30., render_points_as_spheres=True)
        p.enable_point_picking(meshfix.mesh, use_mesh=True)

        p.show()

        apex = p.picked_point
    
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
        mesh_from_vtk = pv.PolyData("{}/{}_clean.vtk".format(full_path, atrium))
        p.add_mesh(mesh_from_vtk, 'r')
        p.add_text('Select the transeptal punture and close the window',position='lower_left')
        p.enable_point_picking(meshfix.mesh, use_mesh=True)

        p.show()

        if p.picked_point is not None:
            loc = vtk.vtkPointLocator()
            loc.SetDataSet(model)
            loc.BuildLocator()
            transeptal_punture_id = vtk.util.numpy_support.vtk_to_numpy(model.GetPointData().GetArray('Ids'))[loc.FindClosestPoint(p.picked_point)]

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
                            
        pt_high_c = list(vtk.util.numpy_support.vtk_to_numpy(surface.GetPointData().GetArray('Ids')))
        curv_s = vtk.util.numpy_support.vtk_to_numpy(surface.GetPointData().GetArray('curv'))
        
        if not MRI:
            if transeptal_punture_id not in pt_high_c:
                if len(set(pt_high_c).intersection(pts_low_v))>0: # the region is both high curvature and low voltage
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
                        pt_surf_2 = list(vtk.util.numpy_support.vtk_to_numpy(surface2.GetPointData().GetArray('Ids')))
                        if len(set(pt_high_c).intersection(pt_surf_2))>0:

                            for el in vtk.util.numpy_support.vtk_to_numpy(surface2.GetCellData().GetArray('Ids')):
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
            
                    loc_low_V_pts = vtk.util.numpy_support.vtk_to_numpy(loc_low_V.GetPoints().GetData())

                    max_dist = 0
                    for l in range(len(loc_low_V_pts)):
                        if np.sqrt(np.sum((pt_max_curv-loc_low_V_pts[l])**2, axis=0)) > max_dist:
                            max_dist = np.sqrt(np.sum((pt_max_curv-loc_low_V_pts[l])**2, axis=0))

                    el_to_del = find_elements_within_radius(model,pt_max_curv,min_cutting_radius*2*scale)

                    el_to_del_tot = el_to_del_tot.union(set(el_to_del))
                    
                
                else: # Possible appendage

                    if np.max(curv_s) > old_max: # The max curvature without low voltage should be the appendage
                        old_max = np.max(curv_s)
                        apex = np.asarray(model.GetPoint(Gl_pt_id.index(pt_high_c[np.argmax(curv_s)])))
        else:
            if not apex_id in pt_high_c:
                for el in vtk.util.numpy_support.vtk_to_numpy(surface.GetCellData().GetArray('Ids')):
                    el_to_del_tot.add(Gl_cell_id.index(el))

        connect.DeleteSpecifiedRegion(i)
        connect.Update()

    model_new_el = vtk.vtkIdList()
    cell_id_all = list(range(model.GetNumberOfCells()))
    el_diff =  list(set(cell_id_all).difference(el_to_del_tot))
    
    for var in el_diff:
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

    model = cleaner.GetOutput()
    
    model = extract_largest_region(model)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName("{}/{}_cutted.vtk".format(full_path, atrium))
    writer.SetInputData(model)
    writer.SetFileTypeToBinary()
    writer.Write()
    
    if debug:
        if apex is not None:
            point_cloud = pv.PolyData(apex)

            p = pv.Plotter(notebook=False)
            mesh_from_vtk = pv.PolyData("{}/{}_cutted.vtk".format(full_path, atrium))
            p.add_mesh(mesh_from_vtk, 'r')
            p.add_mesh(point_cloud, color='black', point_size=30., render_points_as_spheres=True)
            p.enable_point_picking(meshfix.mesh, use_mesh=True)
            p.add_text('Select the appendage apex and close the window',position='lower_left')
            p.show()

            if p.picked_point is not None:
                apex = p.picked_point
        else:
            p = pv.Plotter(notebook=False)
            mesh_from_vtk = pv.PolyData("{}/{}_cutted.vtk".format(full_path, atrium))
            p.add_mesh(mesh_from_vtk, 'r')
            p.enable_point_picking(meshfix.mesh, use_mesh=True)
            p.add_text('Select the appendage apex and close the window',position='lower_left')
            p.show()

            if p.picked_point is not None:
                apex = p.picked_point

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
    extract_rings.run(["--mesh",meshpath,"--LAA",str(LAA),"--RAA",str(RAA)])

def run():

    args = parser().parse_args()

    open_orifices_with_curvature(args.mesh, args.atrium, args.MRI, args.scale, args.size, args.min_cutting_radius, args.max_cutting_radius, args.LAA, args.RAA, args.debug)
        
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

def vtk_thr(model,mode,points_cells,array,thr1,thr2="None"):
    thresh = vtk.vtkThreshold()
    thresh.SetInputData(model)
    if mode == 0:
        thresh.ThresholdByUpper(thr1)
    elif mode == 1:
        thresh.ThresholdByLower(thr1)
    elif mode ==2:
        if int(vtk_version) >= 9:
            thresh.ThresholdBetween(thr1,thr2)
        else:
            thresh.ThresholdByUpper(thr1)
            thresh.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_"+points_cells, array)
            thresh.Update()
            thr = thresh.GetOutput()
            thresh = vtk.vtkThreshold()
            thresh.SetInputData(thr)
            thresh.ThresholdByLower(thr2)
    thresh.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_"+points_cells, array)
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
            data = vtk.util.numpy_support.vtk_to_numpy(mesh1.GetPointData().GetArray(mesh1.GetPointData().GetArrayName(i)))
            if isinstance(data[0], collections.Sized):
                data2 = np.zeros((len(pts2),len(data[0])), dtype=data.dtype)
            else:
                data2 = np.zeros((len(pts2),), dtype=data.dtype)
            
            data2 = data[ii]
            data2 = np.where(np.isnan(data2), 10000, data2)
            
            meshNew.PointData.append(data2, mesh1.GetPointData().GetArrayName(i))
    else:
        data = vtk.util.numpy_support.vtk_to_numpy(mesh1.GetPointData().GetArray(idat))
        if isinstance(data[0], collections.Sized):
            data2 = np.zeros((len(pts2),len(data[0])), dtype=data.dtype)
        else:
            data2 = np.zeros((len(pts2),), dtype=data.dtype)
        
        data2 = data[ii]
        meshNew.PointData.append(data2, idat)
    
    return meshNew.VTKObject

if __name__ == '__main__':
    run()
