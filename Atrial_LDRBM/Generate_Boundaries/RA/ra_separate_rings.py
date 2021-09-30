#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 19:03:22 2020

@author: tz205

separate rings of of right atrium into:
    -tv: tricuspid valve
    -svc: superior vena cava
    -ivc: inferior vena cava
    -cs: coronary sinus
    
    further those area are extracted:
        -top_endo
        -top_epi
        -tv_f
        -tv_s
    for details pleas read: <Modeling cardiac muscle fibers in ventricular and atrial electrophysiology simulations>
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



# appen_point = [2.4696, -78.0776, 43.5639]
"""
distance(p0,p1,digits=2)

calculate the distance between two points
"""
def distance(p0,p1,digits=4):
    a=map(lambda x: (x[0]-x[1])**2, zip(p0, p1))
    return round(math.sqrt(sum(a)),digits)


def run(appen_point):
    data_checker = vtk.vtkDataSetReader()
    data_checker.SetFileName('result/ra_rings.vtk')
    data_checker.Update()
    
    if data_checker.IsFilePolyData():
        reader = vtk.vtkPolyDataReader()
    elif data_checker.IsFileUnstructuredGrid():
        reader = vtk.vtkUnstructuredGridReader()

    reader.SetFileName('result/ra_rings.vtk')
    reader.Update()
    
    connect = vtk.vtkConnectivityFilter()
    connect.SetInputData(reader.GetOutput())
    connect.SetExtractionModeToAllRegions ()
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
    
    # extract the biggest ring as tricuspid valve
    num_p = max(globals()["ring" + str(i)].points_num for i in range(num))
    for i in range(num):
        if globals()["ring" + str(i)].points_num == num_p:
            globals()["ring" + str(i)].name = 'tv'
            tv_index = i
    
    # extract the second and third biggest rings as svc and ivc
    index_all = []
    svc_ivc = []
    for i in range(num):
        index_all.append(i)
        
    index_all.remove(tv_index)
    while len(svc_ivc) != 2:
        num_p_next_biggest = max(globals()["ring" + str(i)].points_num for i in index_all)
        next_biggest_index = [i for i in index_all if globals()["ring" + str(i)].points_num == num_p_next_biggest][0]
        svc_ivc.append(next_biggest_index)
        index_all.remove(next_biggest_index)
    
    # the rings that neither tv nor svc ivc are cs
    for i in range(num):
        if i != tv_index and i not in svc_ivc:
            globals()["ring" + str(i)].name = 'cs'


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
    for i in svc_ivc:
            center = globals()["ring" + str(i)].center_point
            center_list.append(center)
            index_list.append(i)

    center_list.append(appen_point)
    estimator = KMeans(n_clusters=2)
    estimator.fit(center_list)
    label_pred = estimator.labels_
    print(label_pred)
    
    # svc and appendage are sorted into same cluster since there are close to each other
    svc_lable = label_pred[-1]
    for i in range(len(index_list)):
        index = index_list[i]
        if label_pred[i] == svc_lable:
            globals()["ring" + str(index)].name = "svc"
        else:
            globals()["ring" + str(index)].name = "ivc"

    for i in range(num):
        ring = globals()["ring" + str(i)]
        print(f"ring "+str(i)+f" index: {ring.index}")
        print(f"ring "+str(i)+f" cluster: {ring.name}")
        print(f"ring "+str(i)+f" number of points: {ring.points_num}")
        print(f"ring "+str(i)+f" center points: {ring.center_point}")
        print(f"ring "+str(i)+f" distance: {ring.distance}\n")
    
    # extracting tv,svc,ivc,cs
    svc_index = []
    ivc_index = []
    tv_index = []
    cs_index = []
    for i in range(num):
        cluster = globals()["ring" + str(i)].name
        if cluster == "svc":
            svc_index.append(i)
        elif cluster == "ivc":
            ivc_index.append(i)
        elif cluster == "cs":
            cs_index.append(i)
        else:
            tv_index.append(i)
            
    for var in ["tv", "svc", "ivc", "cs"]:
        if var == "tv":
            region_index = tv_index
        if var == "svc":
            region_index = svc_index
        if var == "ivc":
            region_index = ivc_index
        if var == "cs":
            region_index = cs_index
        
        
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
        writer.SetFileName("result/ra_"+str(var)+"_surface.vtk")
        writer.SetInputData(meshNew.VTKObject)
        writer.Write()
        
        for index in region_index:
            connect.DeleteSpecifiedRegion(index)
    """
    extracting top_endo_eip_others
    """
    data_checker = vtk.vtkDataSetReader()
    data_checker.SetFileName('model/RA.vtk')
    data_checker.Update()
    
    if data_checker.IsFilePolyData():
        reader = vtk.vtkPolyDataReader()
    elif data_checker.IsFileUnstructuredGrid():
        reader = vtk.vtkUnstructuredGridReader()

    reader.SetFileName('model/RA.vtk')
    reader.Update()
    
    tv_center = globals()["ring" + str(tv_index[0])].center_point
    svc_center = globals()["ring" + str(svc_index[0])].center_point
    ivc_center = globals()["ring" + str(ivc_index[0])].center_point

    tv_center = np.array(tv_center)
    svc_center = np.array(svc_center)
    ivc_center = np.array(ivc_center)
    
    # calculate the norm vector
    v1 = tv_center - svc_center
    v2 = tv_center - ivc_center
    norm = np.cross(v1, v2)
    
    #normlize norm
    n = np.linalg.norm([norm], axis=1, keepdims=True)
    norm_1 = norm/n

    plane = vtk.vtkPlane()
    plane.SetNormal(norm_1[0][0], norm_1[0][1], norm_1[0][2])
    plane.SetOrigin(tv_center[0], tv_center[1], tv_center[2])
    
    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(reader.GetOutput())
    geo_filter.Update()
    surface = geo_filter.GetOutput()
    
    # clip = vtk.vtkClipPolyData()
    # clip.SetClipFunction(plane)
    # clip.SetInputData(surface)
    # clip.Update()
    meshExtractFilter = vtk.vtkExtractGeometry()
    meshExtractFilter.SetInputData(surface)
    meshExtractFilter.SetImplicitFunction(plane)
    meshExtractFilter.Update()
    
    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(meshExtractFilter.GetOutput())
    geo_filter.Update()
    surface = geo_filter.GetOutput()
    
    # save the result
    meshNew = dsa.WrapDataObject(surface)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName("temp_result/test_cut_0.vtk")
    writer.SetInputData(meshNew.VTKObject)
    writer.Write()
    
    """
    this is a back up plan:
        it will let the Origin point of clipping plane move along the norm vector with distance= 0.5 and cut again
    """
    # point_moved = tv_center + 0.5 * norm_1
    # print(point_moved[0][0])
    # plane2 = vtk.vtkPlane()
    # plane2.SetNormal(-norm_1[0][0], -norm_1[0][1], -norm_1[0][2])
    # plane2.SetOrigin(point_moved[0][0], point_moved[0][1], point_moved[0][2])
    
    # clip2 = vtk.vtkClipPolyData()
    # clip2.SetClipFunction(plane2)
    # clip2.SetInputData(clip.GetOutput(0))
    # clip2.Update()
        
    # meshNew = dsa.WrapDataObject(clip2.GetOutput(0))
    # writer = vtk.vtkPolyDataWriter()
    # writer.SetFileName("temp_result/gamma_top.vtk")
    # writer.SetInputData(meshNew.VTKObject)
    # writer.Write()
    
    """
    here we will extract the feature edge 
    """
    boundaryEdges = vtk.vtkFeatureEdges()
    boundaryEdges.SetInputData(surface)
    boundaryEdges.BoundaryEdgesOn()
    boundaryEdges.FeatureEdgesOff()
    boundaryEdges.ManifoldEdgesOff()
    boundaryEdges.NonManifoldEdgesOff()
    boundaryEdges.Update()
    # points = boundaryEdges.GetOutput().GetPoints().GetData()
    
    meshNew = dsa.WrapDataObject(boundaryEdges.GetOutput())
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName("result/gamma_top.vtk")
    writer.SetInputData(meshNew.VTKObject)
    writer.Write()
    
    """
    separate the tv into tv tv-f and tv-f
    """
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName('result/ra_tv_surface.vtk')
    reader.Update()
    tv = reader.GetOutput()
    # calculate the norm vector
    v1 = svc_center - tv_center
    v2 = ivc_center - tv_center
    norm = np.cross(v1, v2)
    
    #normlize norm
    n = np.linalg.norm([norm], axis=1, keepdims=True)
    norm_1 = norm/n
    norm_2 = - norm_1

    plane = vtk.vtkPlane()
    plane.SetNormal(norm_1[0][0], norm_1[0][1], norm_1[0][2])
    plane.SetOrigin(tv_center[0], tv_center[1], tv_center[2])
    
    plane2 = vtk.vtkPlane()
    plane2.SetNormal(norm_2[0][0], norm_2[0][1], norm_2[0][2])
    plane2.SetOrigin(tv_center[0], tv_center[1], tv_center[2])
    
    meshExtractFilter = vtk.vtkExtractGeometry()
    meshExtractFilter.SetInputData(tv)
    meshExtractFilter.SetImplicitFunction(plane)
    meshExtractFilter.Update()
    
    meshExtractFilter2 = vtk.vtkExtractGeometry()
    meshExtractFilter2.SetInputData(tv)
    meshExtractFilter2.ExtractBoundaryCellsOn()
    meshExtractFilter2.SetImplicitFunction(plane2)
    meshExtractFilter2.Update()
    
    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(meshExtractFilter.GetOutput())
    geo_filter.Update()
    surface = geo_filter.GetOutput()
    
    geo_filter2 = vtk.vtkGeometryFilter()
    geo_filter2.SetInputData(meshExtractFilter2.GetOutput())
    geo_filter2.Update()
    surface2 = geo_filter2.GetOutput()
    
    # save the result
    meshNew = dsa.WrapDataObject(surface)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName("result/ra_tv_f_surface.vtk")
    writer.SetInputData(meshNew.VTKObject)
    writer.Write()
    
    # save the result
    meshNew = dsa.WrapDataObject(surface2)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName("result/ra_tv_s_surface.vtk")
    writer.SetInputData(meshNew.VTKObject)
    writer.Write()
    
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