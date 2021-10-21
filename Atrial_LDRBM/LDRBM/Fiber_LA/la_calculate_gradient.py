#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:55:02 2021

@author: Luca Azzolin
"""
import numpy as np
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
import os

def la_calculate_gradient(args, model, job):
    name_list = ['phi', 'r', 'v', 'ab']

    # change the result of Laplace from points data to cell data
    pointDataToCellData = vtk.vtkPointDataToCellData()
    pointDataToCellData.SetInputData(model)
    pointDataToCellData.Update()

    for var in name_list:
        print('Calculating the gradient of ' + str(var) + '...')
        if var =='phi':
            # using the vtkGradientFilter to calculate the gradient
            if args.mesh_type == "vol":
                gradientFilter = vtk.vtkGradientFilter()
                gradientFilter.SetInputConnection(pointDataToCellData.GetOutputPort())
                gradientFilter.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "phie_"+str(var))
                gradientFilter.SetResultArrayName('grad_'+str(var))
                gradientFilter.Update()
                LA_gradient = gradientFilter.GetOutput()
            else:
                normalFilter = vtk.vtkPolyDataNormals()
                normalFilter.SetInputConnection(pointDataToCellData.GetOutputPort())
                normalFilter.ComputeCellNormalsOn()
                normalFilter.ComputePointNormalsOff()
                normalFilter.SplittingOff()
                normalFilter.Update()
                LA_gradient = normalFilter.GetOutput()
                LA_gradient.GetCellData().GetArray("Normals").SetName('grad_'+str(var))
                
        else:
            gradientFilter = vtk.vtkGradientFilter()
            gradientFilter.SetInputData(LA_gradient)
            gradientFilter.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "phie_"+str(var))
            gradientFilter.SetResultArrayName('grad_'+str(var))
            gradientFilter.Update()
            LA_gradient = gradientFilter.GetOutput()
            
        print('Calculating the gradient of ' + str(var) + '... Done!')
    
    output = vtk.vtkUnstructuredGrid()
    output.DeepCopy(LA_gradient)
    if args.debug == 1:
        # write
        simid = job.ID+"/gradient"
        try:
            os.makedirs(simid)
        except OSError:
            print ("Creation of the directory %s failed" % simid)
        else:
            print ("Successfully created the directory %s " % simid)
        # write the file as vtk 
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(simid+"/LA_with_lp_res_gradient.vtk")
        writer.SetInputData(output)
        writer.Write()
    
    return output

