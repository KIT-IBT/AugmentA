#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 19:03:22 2020

@author: tz205

separate rings of of left atrium into:
    -mv: mitral valve
    -lpv: left pulmonary vein
    -rpv: right pulmonary vein
"""

from mayavi import mlab
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
from vtk.numpy_interface import dataset_adapter as dsa

class rings:
   def __init__(self, index, name, points_num, center_point, distance):
       self.index = index
       self.name = name
       self.points_num = points_num
       self.center_point = center_point
       self.distance = distance

# appen_point = [22.561380859375, -35.339421875, 34.9769375]

"""
distance(p0,p1,digits=2)

calculate the distance between two points
"""
def distance(p0,p1,digits=4):
    a=map(lambda x: (x[0]-x[1])**2, zip(p0, p1))
    return round(math.sqrt(sum(a)),digits)


def run(appen_point):
    data_checker = vtk.vtkDataSetReader()
    data_checker.SetFileName('result/la_rings.vtk')
    data_checker.Update()
    
    if data_checker.IsFilePolyData():
        reader = vtk.vtkPolyDataReader()
    elif data_checker.IsFileUnstructuredGrid():
        reader = vtk.vtkUnstructuredGridReader()

    reader.SetFileName('result/la_rings.vtk')
    reader.Update()
    
    connect = vtk.vtkConnectivityFilter()
    connect.SetInputData(reader.GetOutput())
    connect.SetExtractionModeToAllRegions()
    # connect.SetExtractionModeToLargestRegion()
    connect.Update()
    num = connect.GetNumberOfExtractedRegions()
    
    connect.SetExtractionModeToSpecifiedRegions()

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
        
        # number of points
        points_num = surface.GetNumberOfPoints()
        
        # coordinate of center points of ring
        points_data = surface.GetPoints().GetData()
        ring = vtk.util.numpy_support.vtk_to_numpy(points_data)
        center_point = [np.mean(ring[:,0]),np.mean(ring[:,1]),np.mean(ring[:,2])]
        
        # class rings(index, name, points_num, center_point)
        globals()[f'ring{i}'] = rings(i, 'unknown', points_num, center_point, 0)
        # delete added region id
        connect.DeleteSpecifiedRegion(i)
        connect.Update()
    
    # extract the biggest ring as mitral valve
    num_p = max(globals()["ring" + str(i)].points_num for i in range(num))
    for i in range(num):
        if globals()["ring" + str(i)].points_num == num_p:
            globals()["ring" + str(i)].name = 'mv'
            mv_index = i

    # # calculate the distance
    # d_list = []
    # for i in range(num):
    #     if i != mv_index:
    #         center = globals()["ring" + str(i)].center_point
    #         d = distance(center,appen_point)
    #         globals()["ring" + str(i)].distance = d
    #         print(d)
    #         d_list.append(d)
    # mean_d = np.mean(d_list)
    # print(mean_d)
    
    # Using k-means to sort the center points
    center_list = []
    index_list = []
    for i in range(num):
        if i != mv_index:
            center = globals()["ring" + str(i)].center_point
            center_list.append(center)
            index_list.append(i)

    center_list.append(appen_point)
    estimator = KMeans(n_clusters=2)
    estimator.fit(center_list)
    label_pred = estimator.labels_
    #print(label_pred)

    lpv_lable = label_pred[-1]
    
    for i in range(len(index_list)):
        index = index_list[i]
        if label_pred[i] == lpv_lable:
            globals()["ring" + str(index)].name = "lpv"
        else:
            globals()["ring" + str(index)].name = "rpv"
    
    for i in range(num):
        ring = globals()["ring" + str(i)]
        print(f"ring "+str(i)+f" index: {ring.index}")
        print(f"ring "+str(i)+f" cluster: {ring.name}")
        print(f"ring "+str(i)+f" number of points: {ring.points_num}")
        print(f"ring "+str(i)+f" center points: {ring.center_point}")
        print(f"ring "+str(i)+f" distance: {ring.distance}\n")
        
    # extracting mv lpv and rpv
    lpv_index = []
    rpv_index = []
    mv_index = []
    for i in range(num):
        cluster = globals()["ring" + str(i)].name
        if cluster == "lpv":
            lpv_index.append(i)
        elif cluster == "rpv":
            rpv_index.append(i)
        else:
            mv_index.append(i)
            
    for var in ["mv", "lpv", "rpv"]:
        if var  == "mv":
            region_index = mv_index
        if var == "lpv":
            region_index = lpv_index
        if var == "rpv":
            region_index = rpv_index
        
        
        for index in region_index:
            connect.AddSpecifiedRegion(index)
        
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
        
        # Write into vtk
        meshNew = dsa.WrapDataObject(surface)
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName("result/la_"+str(var)+"_surface.vtk")
        writer.SetInputData(meshNew.VTKObject)
        writer.Write()
        
        for index in region_index:
            connect.DeleteSpecifiedRegion(index)

    # # show the result
    # global mesh, cursor3d
    # fig = mlab.figure('Atrium, detected rings and centers of rings')
    # mlab.clf()
    # points = mlab.points3d(points_coordinate[:,0], points_coordinate[:,1], points_coordinate[:,2], scale_factor = 1,color=(0.0, 0.9, 0.9))
    # # appendage
    # appendage = mlab.points3d(appen_point[0], appen_point[1], appen_point[2], scale_factor = 3,color=(0.9, 0, 0))
    # mlab.show()
if __name__ == '__main__':
    run()