#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 15:54:50 2020

@author: tz205

Get the matrial for the laplace:
    points id list of:
        lpv
        rpv
        mv
        ap
"""
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import shutil
import os

def get_surf_ids(name,polydata):
    data_checker = vtk.vtkDataSetReader()
    data_checker.SetFileName('result/la_'+str(name)+'_surface.vtk')
    data_checker.Update()
    
    if data_checker.IsFilePolyData():
        reader = vtk.vtkPolyDataReader()
    elif data_checker.IsFileUnstructuredGrid():
        reader = vtk.vtkUnstructuredGridReader()

    reader.SetFileName('result/la_'+str(name)+'_surface.vtk')
    reader.Update()
    points_data = reader.GetOutput().GetPoints().GetData()
    points_list = vtk.util.numpy_support.vtk_to_numpy(points_data)
    
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(polydata)
    loc.BuildLocator()
    
    point_id_list = []
    for i in range(len(points_list)):
        temp_point_id_list = loc.FindClosestPoint(points_list[i,:])
        point_id_list += [temp_point_id_list]
        
    # if name == 'epi':
    #     reader = vtk.vtkPolyDataReader()
    #     reader.SetFileName('result/la_epi_surface.vtk')
    #     reader.Update()
    #     epi_surface = reader.GetOutput()
    
    #     # geo_filter = vtk.vtkGeometryFilter()
    #     # geo_filter.SetInputData(epi_surface)
    #     # geo_filter.Update()
    #     # epi_surface = geo_filter.GetOutput()
    #     #print(polydata)
    #     # print('Extracting the surface close to contour...')
    #     # starttime = datetime.datetime.now()
    #     # endtime = datetime.datetime.now()
    #     # running_time = endtime - starttime
    #     # print('Extracting the surface close to contour...Done \n takes: '+str(running_time))
    
    #     boundaryEdges = vtk.vtkFeatureEdges()
    #     boundaryEdges.SetInputData(epi_surface)
    #     boundaryEdges.BoundaryEdgesOn()
    #     boundaryEdges.FeatureEdgesOff()
    #     boundaryEdges.ManifoldEdgesOff()
    #     boundaryEdges.NonManifoldEdgesOff()
    #     boundaryEdges.Update()
    #     points = boundaryEdges.GetOutput().GetPoints().GetData()
    
    #     points = vtk.util.numpy_support.vtk_to_numpy(points)
    #     #points = np.array(points)
    #     loc = vtk.vtkPointLocator()
    #     loc.SetDataSet(polydata)
    #     loc.BuildLocator()
    #     # Get all the point ids of points in mixed_points_list
    #     ring_point_id_list = []
    #     for i in range(len(points)):
    #         temp_point_id_list = loc.FindClosestPoint(points[i,:])
    #         ring_point_id_list += [temp_point_id_list]
    #     point_id_list = list(set(point_id_list).difference(set(ring_point_id_list)))

    file_write_obj = open("Surf/surf_"+str(name)+".vtx", 'w')
    file_write_obj.writelines(str(len(point_id_list)))
    file_write_obj.write('\n')
    file_write_obj.writelines('extra')
    file_write_obj.write('\n')
    for var in point_id_list:
        file_write_obj.writelines(str(var))
        file_write_obj.write('\n')
    file_write_obj.close()
    shutil.copyfile("Surf/surf_"+str(name)+".vtx", "../../LDRBM/Fiber_LA/Surfs/surf_"+str(name)+".vtx")

def run(appen_point):
    """The whole model"""
    data_checker = vtk.vtkDataSetReader()
    data_checker.SetFileName('model/LA.vtk')
    data_checker.Update()
    
    if data_checker.IsFilePolyData():
        reader = vtk.vtkPolyDataReader()
    elif data_checker.IsFileUnstructuredGrid():
        reader = vtk.vtkUnstructuredGridReader()
        
    reader.SetFileName('model/LA.vtk')
    reader.Update()
    polydata = reader.GetOutput()
    whole_model_points_data = reader.GetOutput().GetPoints().GetData()
    whole_model_points_coordinate = vtk.util.numpy_support.vtk_to_numpy(whole_model_points_data)
    
    """Epi surface"""
    get_surf_ids('epi',polydata)
    
    """Endo surface"""
    get_surf_ids('endo_without_rings',polydata)
    
    """Mitral valve"""
    get_surf_ids('mv',polydata)
    
    """Left pulmonary vein"""
    get_surf_ids('lpv',polydata)
    
    """Right pulmonary vein"""
    get_surf_ids('rpv',polydata)
    
    """Appendage"""
    # appen_point = [22.561380859375, -35.339421875, 34.9769375]
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(polydata)
    loc.BuildLocator()
    ap_point_id_list = []
    temp_point_id_list = loc.FindClosestPoint(appen_point)
    ap_point_id_list += [temp_point_id_list]
    
    file_write_obj = open("Surf/surf_ap.vtx", 'w')
    file_write_obj.writelines(str(len(ap_point_id_list)))
    file_write_obj.write('\n')
    file_write_obj.writelines('extra')
    file_write_obj.write('\n')
    for var in ap_point_id_list:
        file_write_obj.writelines(str(var))
        file_write_obj.write('\n')
    file_write_obj.close()
    # rename endo
    os.rename("../../LDRBM/Fiber_LA/Surfs/surf_endo_without_rings.vtx","../../LDRBM/Fiber_LA/Surfs/surf_endo.vtx")
if __name__ == '__main__':
    run()