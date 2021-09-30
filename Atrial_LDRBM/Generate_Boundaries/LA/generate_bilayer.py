#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 15:45:17 2021

@author: la816
"""

import os
import numpy as np
import vtk
from vtk.util import numpy_support

def run():
    
    meshname = 'result/LA_new.vtk'
    
    data_checker = vtk.vtkDataSetReader()
    data_checker.SetFileName(meshname)
    data_checker.Update()
    
    if data_checker.IsFilePolyData():
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(meshname)
        reader.Update()
        polydata = reader.GetOutput()
    elif data_checker.IsFileUnstructuredGrid():
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(meshname)
        polydata = vtk.vtkExtractSurface(Input=reader.Update())
    
    polydata2 = vtk.vtkPolyData()
    polydata2.DeepCopy(polydata)
    
    normalGenerator = vtk.vtkPolyDataNormals()
    normalGenerator.SetInputData(polydata)
    normalGenerator.ComputeCellNormalsOff()
    normalGenerator.ComputePointNormalsOn()
    normalGenerator.ConsistencyOn()
    normalGenerator.AutoOrientNormalsOn()
    normalGenerator.SplittingOff() 
    normalGenerator.Update()
    
    eps = 1
    direction = 1
    PointNormalArray = numpy_support.vtk_to_numpy(normalGenerator.GetOutput().GetPointData().GetNormals())
    atrial_points = numpy_support.vtk_to_numpy(polydata2.GetPoints().GetData())
    
    atrial_points = atrial_points + eps*direction*PointNormalArray
    
    vtkPts = vtk.vtkPoints()
    vtkPts.SetData(numpy_support.numpy_to_vtk(atrial_points))
    polydata2.SetPoints(vtkPts)
    
    appendFilter = vtk.vtkAppendPolyData()
    appendFilter.AddInputData(polydata)
    appendFilter.AddInputData(polydata2)
    appendFilter.Update()
    
    appended_polydata = appendFilter.GetOutput()
    lines = vtk.vtkCellArray()
    
    for i in range(len(atrial_points)):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0,i);
        line.GetPointIds().SetId(1,len(atrial_points)+i);
        lines.InsertNextCell(line)
    
    appended_polydata.SetLines(lines)
    vtk_copy = vtk.vtkUnstructuredGrid()
    vtk_copy.DeepCopy(appended_polydata)
    
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetInputData(vtk_copy)
    writer.SetFileName('result/LA_new2.vtk')
    writer.Write()
    
    
    
    
if __name__ == '__main__':
    run()

