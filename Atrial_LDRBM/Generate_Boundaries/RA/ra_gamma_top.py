#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 14:05:05 2020

@author: tz205
get the top_epi and top_endo
"""

from mayavi import mlab
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
from vtk.numpy_interface import dataset_adapter as dsa

def run():
    data_checker = vtk.vtkDataSetReader()
    data_checker.SetFileName('result/gamma_top.vtk')
    data_checker.Update()
    
    if data_checker.IsFilePolyData():
        reader = vtk.vtkPolyDataReader()
    elif data_checker.IsFileUnstructuredGrid():
        reader = vtk.vtkUnstructuredGridReader()

    reader.SetFileName('result/gamma_top.vtk')
    reader.Update()
    gamma_top = reader.GetOutput()
    gamma_top_points= gamma_top.GetPoints().GetData()
    gamma_top_points = vtk.util.numpy_support.vtk_to_numpy(gamma_top_points)
    
    
    reader_ivc = vtk.vtkPolyDataReader()
    reader_ivc.SetFileName('result/ra_ivc_surface.vtk')
    reader_ivc.Update()
    ivc = reader_ivc.GetOutput()
    ivc_points = ivc.GetPoints().GetData()
    ivc_points = vtk.util.numpy_support.vtk_to_numpy(ivc_points)
    
    reader_svc = vtk.vtkPolyDataReader()
    reader_svc.SetFileName('result/ra_svc_surface.vtk')
    reader_svc.Update()
    svc = reader_svc.GetOutput()
    svc_points = svc.GetPoints().GetData()
    svc_points = vtk.util.numpy_support.vtk_to_numpy(svc_points)
    
    reader_epi = vtk.vtkPolyDataReader()
    reader_epi.SetFileName('result/ra_epi_surface.vtk')
    reader_epi.Update()
    epi = reader_epi.GetOutput()
    epi_points = epi.GetPoints().GetData()
    epi_points = vtk.util.numpy_support.vtk_to_numpy(epi_points)
    
    gamma_top_points = gamma_top_points.tolist()
    ivc_points = ivc_points.tolist()
    svc_points = svc_points.tolist()
    epi_points = epi_points.tolist()

    connect = vtk.vtkConnectivityFilter()
    connect.SetInputData(gamma_top)
    connect.SetExtractionModeToSpecifiedRegions()
    connect.Update()
    num = connect.GetNumberOfExtractedRegions()
    for i in range(num):
        connect.AddSpecifiedRegion(i)
        connect.Update()
        surface = connect.GetOutput()
        # Clean unused points
        cln = vtk.vtkCleanPolyData()
        cln.SetInputData(surface)
        cln.Update()
        surface = cln.GetOutput()
        points = surface.GetPoints().GetData()
        points = vtk.util.numpy_support.vtk_to_numpy(points)
        points = points.tolist()
        
        # for var in points:
        #     if var in ivc_points:
        #         for var in points:
        #             if var in svc_points:
        #                 print(i)
        #                 break
        in_ivc = False
        in_svc = False
        # if there is point of group i in both svc and ivc then it is the "top_endo+epi" we need
        while in_ivc == False and in_svc == False:
            for var in points:
                if var in ivc_points:
                    in_ivc = True
                if var in svc_points:
                    in_svc = True
            if in_ivc and in_svc:
                top_endo_epi_id = i
                break
            else:
                break
    
        # delete added region id
        connect.DeleteSpecifiedRegion(i)
        connect.Update()
        
    print(top_endo_epi_id)
    connect.AddSpecifiedRegion(top_endo_epi_id)
    connect.Update()
    surface = connect.GetOutput()
    # Clean unused points
    cln = vtk.vtkCleanPolyData()
    cln.SetInputData(surface)
    cln.Update()
    top_epi_endo = cln.GetOutput()
    
    
    cellid = vtk.vtkIdFilter()
    cellid.CellIdsOn()
    cellid.SetInputData(top_epi_endo) # vtkPolyData()
    cellid.PointIdsOff()
    cellid.SetIdsArrayName('Cell_ids')
    cellid.Update()
    temp_cell_ids = cellid.GetOutput().GetCellData().GetArray('Cell_ids')
    ID_all = vtk.util.numpy_support.vtk_to_numpy(temp_cell_ids)
    ID_all = ID_all.tolist()
    
    endo_cell_idlist = vtk.vtkIdList()
    for i in range(len(ID_all)):
        endo_cell_idlist.InsertNextId(ID_all[i])
    
    # get the top_epi point id list
    top_points = cln.GetOutput().GetPoints().GetData()
    top_points = vtk.util.numpy_support.vtk_to_numpy(top_points)
    top_points = top_points.tolist()
    
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(top_epi_endo)
    loc.BuildLocator()
    
    epi_id_list = vtk.vtkIdList()
    for var in top_points:
        if var in epi_points:
            point_id = loc.FindClosestPoint(var)
            epi_id_list.InsertNextId(point_id)

    epi_cell_idlist = vtk.vtkIdList()
    epi_cell_idlist_all = vtk.vtkIdList()
    for i in range(epi_id_list.GetNumberOfIds()):
        top_epi_endo.GetPointCells(epi_id_list.GetId(i), epi_cell_idlist)
        for j in range(epi_cell_idlist.GetNumberOfIds()):
            epi_cell_idlist_all.InsertNextId(epi_cell_idlist.GetId(j))
        for j in range(epi_cell_idlist.GetNumberOfIds()):
            endo_cell_idlist.DeleteId(epi_cell_idlist.GetId(j))
            
    
    print(epi_cell_idlist_all)
    print(endo_cell_idlist)
            
    extractor = vtk.vtkExtractCells()  
    extractor.SetInputData(top_epi_endo)
    extractor.AddCellList(epi_cell_idlist_all)
    extractor.Update()
    extraction = extractor.GetOutput()
    
    meshNew = dsa.WrapDataObject(extraction)
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName("result/ra_top_epi_surface.vtk")
    writer.SetInputData(meshNew.VTKObject)
    writer.Write()
    
    extractor = vtk.vtkExtractCells()  
    extractor.SetInputData(top_epi_endo)
    extractor.AddCellList(endo_cell_idlist)
    extractor.Update()
    extraction = extractor.GetOutput()


    meshNew = dsa.WrapDataObject(extraction)
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName("result/ra_top_endo_surface.vtk")
    writer.SetInputData(meshNew.VTKObject)
    writer.Write()
    
if __name__ == '__main__':
    run()