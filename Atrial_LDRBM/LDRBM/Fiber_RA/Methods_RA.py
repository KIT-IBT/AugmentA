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
import vtk
import numpy as np
from vtk.util import numpy_support
from vtk.numpy_interface import dataset_adapter as dsa
from scipy.spatial import cKDTree

vtk_version = vtk.vtkVersion.GetVTKSourceVersion().split()[-1].split('.')[0]

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


def downsample_path(points_data, step):
    # down sampling
    path_all = np.asarray([points_data[i] for i in range(len(points_data)) if i % step == 0 or i == len(points_data) - 1])
   
    # fit a spline
    spline_points = vtk.vtkPoints()
    for i in range(len(path_all)):
        spline_points.InsertPoint(i, path_all[i][0], path_all[i][1], path_all[i][2])
    spline = vtk.vtkParametricSpline()
    spline.SetPoints(spline_points)
    functionSource = vtk.vtkParametricFunctionSource()
    functionSource.SetParametricFunction(spline)
    functionSource.SetUResolution(30 * spline_points.GetNumberOfPoints())
    functionSource.Update()
    points_data = vtk.util.numpy_support.vtk_to_numpy(functionSource.GetOutput().GetPoints().GetData())
    return points_data

def move_surf_along_normals(mesh, eps, direction):

    extract_surf = vtk.vtkGeometryFilter()
    extract_surf.SetInputData(mesh)
    extract_surf.Update()
    polydata = extract_surf.GetOutput()
    
    normalGenerator = vtk.vtkPolyDataNormals()
    normalGenerator.SetInputData(polydata)
    normalGenerator.ComputeCellNormalsOff()
    normalGenerator.ComputePointNormalsOn()
    normalGenerator.ConsistencyOn()
    normalGenerator.SplittingOff() 
    normalGenerator.Update()
    
    PointNormalArray = numpy_support.vtk_to_numpy(normalGenerator.GetOutput().GetPointData().GetNormals())
    atrial_points = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
    
    atrial_points = atrial_points + eps*direction*PointNormalArray
    
    vtkPts = vtk.vtkPoints()
    vtkPts.SetData(numpy_support.numpy_to_vtk(atrial_points))
    polydata.SetPoints(vtkPts)
    
    mesh = vtk.vtkUnstructuredGrid()
    mesh.DeepCopy(polydata)
    
    return mesh
    
def generate_bilayer(endo, epi, max_dist=np.inf):
    
    extract_surf = vtk.vtkGeometryFilter()
    extract_surf.SetInputData(endo)
    extract_surf.Update()
    
    reverse = vtk.vtkReverseSense()
    reverse.ReverseCellsOn()
    reverse.ReverseNormalsOn()
    reverse.SetInputConnection(extract_surf.GetOutputPort())
    reverse.Update()
    
    endo = vtk.vtkUnstructuredGrid()
    endo.DeepCopy(reverse.GetOutput())
            
    endo_pts = numpy_support.vtk_to_numpy(endo.GetPoints().GetData())
    epi_pts = numpy_support.vtk_to_numpy(epi.GetPoints().GetData())
    
    tree = cKDTree(epi_pts)
    dd, ii = tree.query(endo_pts, distance_upper_bound = max_dist, n_jobs=-1)
    
    endo_ids = np.where(dd!=np.inf)[0]
    epi_ids = ii[endo_ids]
    lines = vtk.vtkCellArray()
    
    for i in range(len(endo_ids)):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0,i);
        line.GetPointIds().SetId(1,len(endo_ids)+i);
        lines.InsertNextCell(line)
    
    points = np.vstack((endo_pts[endo_ids], epi_pts[epi_ids]))
    polydata = vtk.vtkUnstructuredGrid()
    vtkPts = vtk.vtkPoints()
    vtkPts.SetData(numpy_support.numpy_to_vtk(points))
    polydata.SetPoints(vtkPts)
    polydata.SetCells(3, lines)
    
    fibers = np.zeros((len(endo_ids),3),dtype="float32")
    fibers[:,0] = 1
    
    tag = np.ones((len(endo_ids),), dtype=int)
    tag[:] = 100
    
    meshNew = dsa.WrapDataObject(polydata)
    meshNew.CellData.append(tag, "elemTag")
    meshNew.CellData.append(fibers, "fiber")
    fibers = np.zeros((len(endo_ids),3),dtype="float32")
    fibers[:,1] = 1
    meshNew.CellData.append(fibers, "sheet")
    
    appendFilter = vtk.vtkAppendFilter()
    appendFilter.AddInputData(endo)
    appendFilter.AddInputData(epi)
    appendFilter.AddInputData(meshNew.VTKObject)
    appendFilter.MergePointsOn()
    appendFilter.Update()
    
    bilayer = appendFilter.GetOutput()
    
    return bilayer

def write_bilayer(bilayer, args, job):
    
    if args.ofmt == 'vtk':
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(job.ID+"/result_RA/LA_RA_bilayer_with_fiber.vtk")
        writer.SetFileTypeToBinary()
    else:
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(job.ID+"/result_RA/LA_RA_bilayer_with_fiber.vtu")
    writer.SetInputData(bilayer)
    writer.Write()
    
    pts = numpy_support.vtk_to_numpy(bilayer.GetPoints().GetData())
    with open(job.ID+'/result_RA/LA_RA_bilayer_with_fiber.pts',"w") as f:
        f.write("{}\n".format(len(pts)))
        for i in range(len(pts)):
            f.write("{} {} {}\n".format(pts[i][0], pts[i][1], pts[i][2]))
    
    tag_epi = vtk.util.numpy_support.vtk_to_numpy(bilayer.GetCellData().GetArray('elemTag'))

    with open(job.ID+'/result_RA/LA_RA_bilayer_with_fiber.elem',"w") as f:
        f.write("{}\n".format(bilayer.GetNumberOfCells()))
        for i in range(bilayer.GetNumberOfCells()):
            cell = bilayer.GetCell(i)
            if cell.GetNumberOfPoints() == 2:
                f.write("Ln {} {} {}\n".format(cell.GetPointIds().GetId(0), cell.GetPointIds().GetId(1), tag_epi[i]))
            elif cell.GetNumberOfPoints() == 3:
                f.write("Tr {} {} {} {}\n".format(cell.GetPointIds().GetId(0), cell.GetPointIds().GetId(1), cell.GetPointIds().GetId(2), tag_epi[i]))
            elif cell.GetNumberOfPoints() == 4:
                f.write("Tt {} {} {} {} {}\n".format(cell.GetPointIds().GetId(0), cell.GetPointIds().GetId(1), cell.GetPointIds().GetId(2), cell.GetPointIds().GetId(3), tag_epi[i]))
            else:
                print("strange "+ str(cell.GetNumberOfPoints()))
    el_epi = vtk.util.numpy_support.vtk_to_numpy(bilayer.GetCellData().GetArray('fiber'))
    sheet_epi = vtk.util.numpy_support.vtk_to_numpy(bilayer.GetCellData().GetArray('sheet'))
    
    with open(job.ID+'/result_RA/LA_RA_bilayer_with_fiber.lon',"w") as f:
        f.write("2\n")
        for i in range(len(el_epi)):
            f.write("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(el_epi[i][0], el_epi[i][1], el_epi[i][2], sheet_epi[i][0], sheet_epi[i][1], sheet_epi[i][2]))

def generate_sheet_dir(args, model, job):

    fiber = model.GetCellData().GetArray('fiber')
    fiber = vtk.util.numpy_support.vtk_to_numpy(fiber)

    fiber = np.where(fiber == [0,0,0], [1,0,0], fiber).astype("float32")
    
    print('reading done!')
    
    '''
    extract the surface
    '''
    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(model)
    geo_filter.Update()
    surface = geo_filter.GetOutput()
    
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(surface)
    cleaner.Update()
    cln_surface = cleaner.GetOutput()
    
    # meshNew = dsa.WrapDataObject(cln_surface)
    # writer = vtk.vtkPolyDataWriter()
    # writer.SetFileName("test_model_surface.vtk")
    # writer.SetInputData(meshNew.VTKObject)
    # writer.Write()
    
    
    '''
    calculate normals of surface cells
    '''
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(surface)
    normals.ComputeCellNormalsOn()
    normals.ComputePointNormalsOff()
    normals.Update()
    normal_vectors = normals.GetOutput().GetCellData().GetArray('Normals')
    normal_vectors = vtk.util.numpy_support.vtk_to_numpy(normal_vectors)
    
    # print('Normal vectors: \n', normal_vectors, '\n')
    print('Number of normals: ', len(normal_vectors))
    print('2nd norm of vectors: ', np.linalg.norm(normal_vectors[0],ord=2), '\n')
    
    if args.mesh_type == "vol":
        
        '''
        calculate the centers of surface mesh cells
        '''
        filter_cell_centers = vtk.vtkCellCenters()
        filter_cell_centers.SetInputData(cln_surface)
        filter_cell_centers.Update()
        center_surface = filter_cell_centers.GetOutput().GetPoints()
        center_surface_array = vtk.util.numpy_support.vtk_to_numpy(center_surface.GetData())
        print('Number of center_surface: ', len(center_surface_array), '\n')
        
        
        '''
        calculate the centers of volume mesh cells
        '''
        filter_cell_centers = vtk.vtkCellCenters()
        filter_cell_centers.SetInputData(model)
        filter_cell_centers.Update()
        center_volume = filter_cell_centers.GetOutput().GetPoints().GetData()
        center_volume = vtk.util.numpy_support.vtk_to_numpy(center_volume)
        print('Number of center_volume: ', len(center_volume), '\n')
        
        
        '''
        Mapping
        '''
        normals = np.ones([len(center_volume), 3], dtype = float)
        kDTree = vtk.vtkKdTree()
        kDTree.BuildLocatorFromPoints(center_surface)
        
        for i in range(len(center_volume)):
            id_list = vtk.vtkIdList()
            kDTree.FindClosestNPoints(1, center_volume[i], id_list)
            index = id_list.GetId(0)
            normals[i] = normal_vectors[index]
        
        print('Mapping done!')
        '''
        calculate the sheet
        '''
    
    elif args.mesh_type == "bilayer":
        normals = normal_vectors
    
    sheet = np.cross(normals, fiber)
    sheet = np.where(sheet == [0,0,0], [1,0,0], sheet).astype("float32")
    # normalize
    abs_sheet = np.linalg.norm(sheet, axis=1, keepdims=True)
    sheet_norm = sheet / abs_sheet
    
    '''
    writing
    '''
    print('writing...')
    # sheet_data = vtk.util.numpy_support.numpy_to_vtk(sheet_norm, deep=True, array_type=vtk.VTK_DOUBLE)
    # sheet_data.SetNumberOfComponents(3)
    # sheet_data.SetName("sheet")
    # model.GetCellData().AddArray(sheet_data)
    
    meshNew = dsa.WrapDataObject(model)
    meshNew.CellData.append(fiber, "fiber")
    meshNew.CellData.append(sheet_norm, "sheet")
    if args.debug:
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(job.ID+"/result_RA/model_with_sheet.vtk")
        writer.SetInputData(meshNew.VTKObject)
        writer.Write()
        print('writing... done!')
    
    return meshNew.VTKObject
             
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

def creat_tube_around_spline(points_data, radius):
    # Creat a points set
    spline_points = vtk.vtkPoints()
    for i in range(len(points_data)):
        spline_points.InsertPoint(i, points_data[i][0], points_data[i][1], points_data[i][2])

    # Fit a spline to the points
    spline = vtk.vtkParametricSpline()
    spline.SetPoints(spline_points)

    functionSource = vtk.vtkParametricFunctionSource()
    functionSource.SetParametricFunction(spline)
    functionSource.SetUResolution(30 * spline_points.GetNumberOfPoints())
    functionSource.Update()

    # Interpolate the scalars
    interpolatedRadius = vtk.vtkTupleInterpolator()
    interpolatedRadius.SetInterpolationTypeToLinear()
    interpolatedRadius.SetNumberOfComponents(1)

    # Generate the radius scalars
    tubeRadius = vtk.vtkDoubleArray()
    n = functionSource.GetOutput().GetNumberOfPoints()
    tubeRadius.SetNumberOfTuples(n)
    tubeRadius.SetName("TubeRadius")

    # TODO make the radius variable???
    tMin = interpolatedRadius.GetMinimumT()
    tMax = interpolatedRadius.GetMaximumT()
    for i in range(n):
        t = (tMax - tMin) / (n - 1) * i + tMin
        r = radius
        # interpolatedRadius.InterpolateTuple(t, r)
        tubeRadius.SetTuple1(i, r)

    # Add the scalars to the polydata
    tubePolyData = functionSource.GetOutput()
    tubePolyData.GetPointData().AddArray(tubeRadius)
    tubePolyData.GetPointData().SetActiveScalars("TubeRadius")

    # Create the tubes TODO: SidesShareVerticesOn()???
    tuber = vtk.vtkTubeFilter()
    tuber.SetInputData(tubePolyData)
    tuber.SetNumberOfSides(40)
    tuber.SidesShareVerticesOn()
    tuber.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
    tuber.SetCapping(1)
    tuber.Update()

    triangle = vtk.vtkTriangleFilter()
    triangle.SetInputData(tuber.GetOutput())
    triangle.Update()

    tuber = triangle
    return tuber


def dijkstra_path(polydata, StartVertex, EndVertex):
    path = vtk.vtkDijkstraGraphGeodesicPath()
    path.SetInputData(polydata)

    path.SetStartVertex(EndVertex)
    path.SetEndVertex(StartVertex)
    path.Update()
    points_data = path.GetOutput().GetPoints().GetData()
    points_data = vtk.util.numpy_support.vtk_to_numpy(points_data)
    return points_data

def dijkstra_path_coord(polydata, StartVertex, EndVertex):
    
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(polydata)
    loc.BuildLocator()
    StartVertex = loc.FindClosestPoint(StartVertex)
    EndVertex = loc.FindClosestPoint(EndVertex)
    
    path = vtk.vtkDijkstraGraphGeodesicPath()
    path.SetInputData(polydata)

    path.SetStartVertex(EndVertex)
    path.SetEndVertex(StartVertex)
    path.Update()
    points_data = path.GetOutput().GetPoints().GetData()
    points_data = vtk.util.numpy_support.vtk_to_numpy(points_data)
    return points_data

def dijkstra_path_on_a_plane(polydata, args, StartVertex, EndVertex, plane_point):
    point_start = np.asarray(polydata.GetPoint(StartVertex))
    point_end = np.asarray(polydata.GetPoint(EndVertex))
    point_third = plane_point

    v1 = point_start - point_end
    v2 = point_start - point_third
    norm = np.cross(v1, v2)
    #
    # # normlize norm
    n = np.linalg.norm(norm)
    norm_1 = norm / n

    plane = vtk.vtkPlane()
    plane.SetNormal(norm_1[0], norm_1[1], norm_1[2])
    plane.SetOrigin(point_start[0], point_start[1], point_start[2])

    meshExtractFilter1 = vtk.vtkExtractGeometry()
    meshExtractFilter1.SetInputData(polydata)
    meshExtractFilter1.SetImplicitFunction(plane)
    meshExtractFilter1.Update()

    point_moved = point_start - 2*args.scale * norm_1
    plane2 = vtk.vtkPlane()
    plane2.SetNormal(-norm_1[0], -norm_1[1], -norm_1[2])
    plane2.SetOrigin(point_moved[0], point_moved[1], point_moved[2])

    meshExtractFilter2 = vtk.vtkExtractGeometry()
    meshExtractFilter2.SetInputData(meshExtractFilter1.GetOutput())
    meshExtractFilter2.SetImplicitFunction(plane2)
    meshExtractFilter2.Update()
    
    band = meshExtractFilter2.GetOutput()
    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(band)
    geo_filter.Update()
    cln = vtk.vtkCleanPolyData()
    cln.SetInputData(geo_filter.GetOutput())
    cln.Update()
    band = cln.GetOutput()
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(band)
    loc.BuildLocator()
    StartVertex = loc.FindClosestPoint(point_start)
    EndVertex = loc.FindClosestPoint(point_end)

    points_data = dijkstra_path(band, StartVertex, EndVertex)
    return points_data


def creat_sphere(center, radius):
    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(center[0], center[1], center[2])
    sphere.SetRadius(radius)
    sphere.SetThetaResolution(40)
    sphere.SetPhiResolution(40)
    sphere.Update()
    return sphere


def creat_tube(center1, center2, radius):
    line = vtk.vtkLineSource()
    line.SetPoint1(center1[0], center1[1], center1[2])
    line.SetPoint2(center2[0], center2[1], center2[2])
    line.Update()

    tube = vtk.vtkTubeFilter()
    tube.SetInputData(line.GetOutput())
    tube.SetRadius(radius)
    tube.SetNumberOfSides(40)
    tube.Update()
    return tube

def find_elements_around_path_within_radius(mesh, points_data, radius):
    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()

    mesh_id_list = vtk.vtkIdList()
    for i in range(len(points_data)):
        temp_result = vtk.vtkIdList()
        locator.FindPointsWithinRadius(radius, points_data[i], temp_result)
        for j in range(temp_result.GetNumberOfIds()):
            mesh_id_list.InsertNextId(temp_result.GetId(j))

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
    
def get_element_ids_around_path_within_radius(mesh, points_data, radius):
    
    gl_ids = vtk.util.numpy_support.vtk_to_numpy(mesh.GetCellData().GetArray('Global_ids'))
    
    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()

    mesh_id_list = vtk.vtkIdList()
    for i in range(len(points_data)):
        temp_result = vtk.vtkIdList()
        locator.FindPointsWithinRadius(radius, points_data[i], temp_result)
        for j in range(temp_result.GetNumberOfIds()):
            mesh_id_list.InsertNextId(temp_result.GetId(j))

    mesh_cell_id_list = vtk.vtkIdList()
    mesh_cell_temp_id_list = vtk.vtkIdList()
    for i in range(mesh_id_list.GetNumberOfIds()):
        mesh.GetPointCells(mesh_id_list.GetId(i), mesh_cell_temp_id_list)
        for j in range(mesh_cell_temp_id_list.GetNumberOfIds()):
            mesh_cell_id_list.InsertNextId(mesh_cell_temp_id_list.GetId(j))
    
    ids = []
    
    for i in range(mesh_cell_id_list.GetNumberOfIds()):
        index = mesh_cell_id_list.GetId(i)
        ids.append(gl_ids[index])

    return ids

def assign_element_tag_around_path_within_radius(mesh, points_data, radius, tag, element_tag):
    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()

    mesh_id_list = vtk.vtkIdList()
    for i in range(len(points_data)):
        temp_result = vtk.vtkIdList()
        locator.FindPointsWithinRadius(radius, points_data[i], temp_result)
        for j in range(temp_result.GetNumberOfIds()):
            mesh_id_list.InsertNextId(temp_result.GetId(j))

    mesh_cell_id_list = vtk.vtkIdList()
    mesh_cell_temp_id_list = vtk.vtkIdList()
    for i in range(mesh_id_list.GetNumberOfIds()):
        mesh.GetPointCells(mesh_id_list.GetId(i), mesh_cell_temp_id_list)
        for j in range(mesh_cell_temp_id_list.GetNumberOfIds()):
            mesh_cell_id_list.InsertNextId(mesh_cell_temp_id_list.GetId(j))

    for i in range(mesh_cell_id_list.GetNumberOfIds()):
        index = mesh_cell_id_list.GetId(i)
        tag[index] = element_tag

    return tag


def normalize_vector(vector):
    abs = np.linalg.norm(vector)
    if abs != 0:
        vector_norm = vector / abs
    else:
        vector_norm = vector

    return vector_norm


def assign_element_fiber_around_path_within_radius(mesh, points_data, radius, fiber, smooth=True):
    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()
    if smooth:
        for i in range(len(points_data)):
            if i % 5 == 0 and i < 5:
                vector = points_data[5] - points_data[0]
            else:
                vector = points_data[i] - points_data[i - 5]
            vector = normalize_vector(vector)
            mesh_point_temp_id_list = vtk.vtkIdList()
            locator.FindPointsWithinRadius(radius, points_data[i], mesh_point_temp_id_list)
            mesh_cell_temp_id_list = vtk.vtkIdList()
            mesh_cell_id_list = vtk.vtkIdList()
            for j in range(mesh_point_temp_id_list.GetNumberOfIds()):
                mesh.GetPointCells(mesh_point_temp_id_list.GetId(j), mesh_cell_temp_id_list)
                for h in range(mesh_cell_temp_id_list.GetNumberOfIds()):
                    mesh_cell_id_list.InsertNextId(mesh_cell_temp_id_list.GetId(h))

            for k in range(mesh_cell_id_list.GetNumberOfIds()):
                index = mesh_cell_id_list.GetId(k)
                fiber[index] = vector
    else:
        for i in range(len(points_data)):
            if i < 1:
                vector = points_data[1] - points_data[0]
            else:
                vector = points_data[i] - points_data[i - 1]
            vector = normalize_vector(vector)
            mesh_point_temp_id_list = vtk.vtkIdList()
            locator.FindPointsWithinRadius(radius, points_data[i], mesh_point_temp_id_list)
            mesh_cell_temp_id_list = vtk.vtkIdList()
            mesh_cell_id_list = vtk.vtkIdList()
            for j in range(mesh_point_temp_id_list.GetNumberOfIds()):
                mesh.GetPointCells(mesh_point_temp_id_list.GetId(j), mesh_cell_temp_id_list)
                for h in range(mesh_cell_temp_id_list.GetNumberOfIds()):
                    mesh_cell_id_list.InsertNextId(mesh_cell_temp_id_list.GetId(h))

            for k in range(mesh_cell_id_list.GetNumberOfIds()):
                index = mesh_cell_id_list.GetId(k)
                fiber[index] = vector
    return fiber


def get_mean_point(data):
    ring_points = data.GetPoints().GetData()
    ring_points = vtk.util.numpy_support.vtk_to_numpy(ring_points)
    center_point = [np.mean(ring_points[:, 0]), np.mean(ring_points[:, 1]), np.mean(ring_points[:, 2])]
    center_point = np.array(center_point)
    return center_point


def multidim_intersect(arr1, arr2):
    arr1_view = arr1.view([('', arr1.dtype)] * arr1.shape[1])
    arr2_view = arr2.view([('', arr2.dtype)] * arr2.shape[1])
    intersected = np.intersect1d(arr1_view, arr2_view)
    return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])


def multidim_intersect_bool(arr1, arr2):
    arr1_view = arr1.view([('', arr1.dtype)] * arr1.shape[1])
    arr2_view = arr2.view([('', arr2.dtype)] * arr2.shape[1])
    intersected = np.intersect1d(arr1_view, arr2_view)
    if len(intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])) == 0:
        res = 0
    else:
        res = 1
    return res


def get_ct_end_points_id(endo, ct, scv, icv):
    # endo
    points_data = endo.GetPoints().GetData()
    endo_points = vtk.util.numpy_support.vtk_to_numpy(points_data)

    # ct
    points_data = ct.GetPoints().GetData()
    ct_points = vtk.util.numpy_support.vtk_to_numpy(points_data)

    # scv
    points_data = scv.GetPoints().GetData()
    scv_points = vtk.util.numpy_support.vtk_to_numpy(points_data)

    # icv
    points_data = icv.GetPoints().GetData()
    icv_points = vtk.util.numpy_support.vtk_to_numpy(points_data)

    # intersection
    # inter_ct_endo = multidim_intersect(endo_points, ct_points)
    # inter_icv = multidim_intersect(inter_ct_endo, icv_points)
    # inter_scv = multidim_intersect(inter_ct_endo, scv_points)`
    inter_icv = multidim_intersect(ct_points, icv_points)
    inter_scv = multidim_intersect(ct_points, scv_points)

    # calculating mean point
    path_icv = np.asarray([np.mean(inter_icv[:, 0]), np.mean(inter_icv[:, 1]), np.mean(inter_icv[:, 2])])
    path_scv = np.asarray([np.mean(inter_scv[:, 0]), np.mean(inter_scv[:, 1]), np.mean(inter_scv[:, 2])])

    loc = vtk.vtkPointLocator()
    loc.SetDataSet(endo)
    loc.BuildLocator()

    path_ct_id_icv = loc.FindClosestPoint(path_icv)
    path_ct_id_scv = loc.FindClosestPoint(path_scv)

    return path_ct_id_icv, path_ct_id_scv


def get_tv_end_points_id(endo, ra_tv_s_surface, ra_ivc_surface, ra_svc_surface, ra_tv_surface):
    # reader = vtk.vtkPolyDataReader()
    # reader.SetFileName('model_pm/ra_tv_s_surface.vtk')
    # reader.Update()
    # tv_s = reader.GetOutput()

    # reader = vtk.vtkPolyDataReader()
    # reader.SetFileName('model_pm/ra_ivc_surface.vtk')
    # reader.Update()
    # tv_ivc = reader.GetOutput()

    # reader = vtk.vtkPolyDataReader()
    # reader.SetFileName('model_pm/ra_svc_surface.vtk')
    # reader.Update()
    # tv_svc = reader.GetOutput()

    # reader = vtk.vtkPolyDataReader()
    # reader.SetFileName('model_pm/ra_tv_surface.vtk')
    # reader.Update()
    # tv = reader.GetOutput()

    tv_center = get_mean_point(ra_tv_surface)
    tv_ivc_center = get_mean_point(ra_ivc_surface)
    tv_svc_center = get_mean_point(ra_svc_surface)

    v1 = tv_center - tv_ivc_center
    v2 = tv_center - tv_svc_center
    norm = np.cross(v1, v2)

    n = np.linalg.norm([norm], axis=1, keepdims=True)
    norm_1 = norm / n
    moved_center = tv_center - norm_1 * 5

    plane = vtk.vtkPlane()
    plane.SetNormal(-norm_1[0][0], -norm_1[0][1], -norm_1[0][2])
    plane.SetOrigin(moved_center[0][0], moved_center[0][1], moved_center[0][2])

    meshExtractFilter = vtk.vtkExtractGeometry()
    meshExtractFilter.SetInputData(ra_tv_s_surface)
    meshExtractFilter.SetImplicitFunction(plane)
    meshExtractFilter.Update()

    connect = vtk.vtkConnectivityFilter()
    connect.SetInputData(meshExtractFilter.GetOutput())
    connect.SetExtractionModeToAllRegions()
    connect.Update()
    connect.SetExtractionModeToSpecifiedRegions()
    connect.AddSpecifiedRegion(1)
    connect.Update()

    # Clean unused points
    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(connect.GetOutput())
    geo_filter.Update()
    surface = geo_filter.GetOutput()

    cln = vtk.vtkCleanPolyData()
    cln.SetInputData(surface)
    cln.Update()
    points_data = cln.GetOutput().GetPoints().GetData()
    ring = vtk.util.numpy_support.vtk_to_numpy(points_data)
    center_point_1 = np.asarray([np.mean(ring[:, 0]), np.mean(ring[:, 1]), np.mean(ring[:, 2])])

    connect.DeleteSpecifiedRegion(1)
    connect.AddSpecifiedRegion(0)
    connect.Update()

    # Clean unused points
    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(connect.GetOutput())
    geo_filter.Update()
    surface = geo_filter.GetOutput()

    cln = vtk.vtkCleanPolyData()
    cln.SetInputData(surface)
    cln.Update()
    points_data = cln.GetOutput().GetPoints().GetData()
    ring = vtk.util.numpy_support.vtk_to_numpy(points_data)
    center_point_2 = np.asarray([np.mean(ring[:, 0]), np.mean(ring[:, 1]), np.mean(ring[:, 2])])
    dis_1 = np.linalg.norm(center_point_1 - tv_ivc_center)
    dis_2 = np.linalg.norm(center_point_1 - tv_svc_center)
    # print(dis_1)
    # print(dis_2)
    if dis_1 < dis_2:
        center_point_icv = center_point_1
        center_point_scv = center_point_2
    else:
        center_point_icv = center_point_2
        center_point_scv = center_point_1
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(endo)
    loc.BuildLocator()

    path_tv_id_icv = loc.FindClosestPoint(center_point_icv)
    path_tv_id_scv = loc.FindClosestPoint(center_point_scv)

    return path_tv_id_icv, path_tv_id_scv


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


def assign_ra_appendage(model, SCV, appex_point, tag, elemTag):
    appex_point = np.asarray(appex_point)
    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(model)
    locator.BuildLocator()

    locator2 = vtk.vtkStaticPointLocator()
    locator2.SetDataSet(SCV)
    locator2.BuildLocator()
    SCV_id = locator2.FindClosestPoint(appex_point)
    SCV_closed_point = SCV.GetPoint(SCV_id)
    radius = np.linalg.norm(appex_point - SCV_closed_point)
    print(radius)

    mesh_point_temp_id_list = vtk.vtkIdList()
    locator.FindPointsWithinRadius(radius, appex_point, mesh_point_temp_id_list)
    print(mesh_point_temp_id_list.GetNumberOfIds())
    mesh_cell_id_list = vtk.vtkIdList()
    mesh_cell_temp_id_list = vtk.vtkIdList()
    for i in range(mesh_point_temp_id_list.GetNumberOfIds()):
        model.GetPointCells(mesh_point_temp_id_list.GetId(i), mesh_cell_temp_id_list)
        for j in range(mesh_cell_temp_id_list.GetNumberOfIds()):
            mesh_cell_id_list.InsertNextId(mesh_cell_temp_id_list.GetId(j))

    for i in range(mesh_cell_id_list.GetNumberOfIds()):
        index = mesh_cell_id_list.GetId(i)
        tag[index] = elemTag

    return tag


def get_endo_ct_intersection_cells(endo, ct):
    points_data = ct.GetPoints().GetData()
    ct_points = vtk.util.numpy_support.vtk_to_numpy(points_data)

    points_data = endo.GetPoints().GetData()
    endo_points = vtk.util.numpy_support.vtk_to_numpy(points_data)

    intersection = multidim_intersect(ct_points, endo_points)

    loc = vtk.vtkPointLocator()
    loc.SetDataSet(endo)
    loc.BuildLocator()

    endo_id_list = []
    for i in range(len(intersection)):
        endo_id_list.append(loc.FindClosestPoint(intersection[i]))
    endo_cell_id_list = vtk.vtkIdList()
    endo_cell_temp_id_list = vtk.vtkIdList()
    for i in range(len(endo_id_list)):
        endo.GetPointCells(endo_id_list[i], endo_cell_temp_id_list)
        for j in range(endo_cell_temp_id_list.GetNumberOfIds()):
            endo_cell_id_list.InsertNextId(endo_cell_temp_id_list.GetId(j))
    print(endo_cell_id_list.GetNumberOfIds())
    extract = vtk.vtkExtractCells()
    extract.SetInputData(endo)
    extract.SetCellList(endo_cell_id_list)
    extract.Update()
    endo_ct = extract.GetOutput()

    return endo_ct

def get_connection_point_la_and_ra_surface(appen_point, la_mv_surface, la_rpv_inf_surface, la_epi_surface, ra_epi_surface):

    loc_mv = vtk.vtkPointLocator()
    loc_mv.SetDataSet(la_mv_surface)
    loc_mv.BuildLocator()

    point_1_id = loc_mv.FindClosestPoint(appen_point)
    point_1 = la_mv_surface.GetPoint(point_1_id)

    loc_rpv_inf = vtk.vtkPointLocator()
    loc_rpv_inf.SetDataSet(la_rpv_inf_surface)
    loc_rpv_inf.BuildLocator()
    point_2_id = loc_rpv_inf.FindClosestPoint(appen_point)
    point_2 = la_rpv_inf_surface.GetPoint(point_2_id)

    loc = vtk.vtkPointLocator()
    loc.SetDataSet(la_epi_surface)
    loc.BuildLocator()

    point_1_id = loc.FindClosestPoint(point_1)
    point_2_id = loc.FindClosestPoint(point_2)

    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(la_epi_surface)
    geo_filter.Update()

    bb_aux_l_points = dijkstra_path(geo_filter.GetOutput(), point_1_id, point_2_id)
    length = len(bb_aux_l_points)
    la_connect_point = bb_aux_l_points[int(length * 0.5)]

    # ra
    geo_filter_la = vtk.vtkGeometryFilter()
    geo_filter_la.SetInputData(la_epi_surface)
    geo_filter_la.Update()
    la_epi_surface = geo_filter_la.GetOutput()

    geo_filter_ra = vtk.vtkGeometryFilter()
    geo_filter_ra.SetInputData(ra_epi_surface)
    geo_filter_ra.Update()
    ra_epi_surface = geo_filter_ra.GetOutput()

    loc_la_epi = vtk.vtkPointLocator()
    loc_la_epi.SetDataSet(la_epi_surface)
    loc_la_epi.BuildLocator()

    loc_ra_epi = vtk.vtkPointLocator()
    loc_ra_epi.SetDataSet(ra_epi_surface)
    loc_ra_epi.BuildLocator()

    la_connect_point_id = loc_la_epi.FindClosestPoint(la_connect_point)
    la_connect_point = la_epi_surface.GetPoint(la_connect_point_id)

    ra_connect_point_id = loc_ra_epi.FindClosestPoint(la_connect_point)
    ra_connect_point = ra_epi_surface.GetPoint(ra_connect_point_id)

    return la_connect_point, ra_connect_point

def get_connection_point_la_and_ra(appen_point):
    la_mv_surface = smart_reader('../../Generate_Boundaries/LA/result/la_mv_surface.vtk')
    la_rpv_inf_surface = smart_reader('../../Generate_Boundaries/LA/result/la_rpv_inf_surface.vtk')
    endo = smart_reader('../../Generate_Boundaries/LA/result/la_endo_surface.vtk')
    la_epi_surface = smart_reader('../../Generate_Boundaries/LA/result/la_epi_surface.vtk')
    ra_epi_surface = smart_reader('../../Generate_Boundaries/RA/result/ra_epi_surface.vtk')

    loc_mv = vtk.vtkPointLocator()
    loc_mv.SetDataSet(la_mv_surface)
    loc_mv.BuildLocator()

    point_1_id = loc_mv.FindClosestPoint(appen_point)
    point_1 = la_mv_surface.GetPoint(point_1_id)

    loc_rpv_inf = vtk.vtkPointLocator()
    loc_rpv_inf.SetDataSet(la_rpv_inf_surface)
    loc_rpv_inf.BuildLocator()
    point_2_id = loc_rpv_inf.FindClosestPoint(appen_point)
    point_2 = la_rpv_inf_surface.GetPoint(point_2_id)

    loc_endo = vtk.vtkPointLocator()
    loc_endo.SetDataSet(endo)
    loc_endo.BuildLocator()

    point_1_id_endo = loc_endo.FindClosestPoint(point_1)
    point_2_id_endo = loc_endo.FindClosestPoint(point_2)

    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(endo)
    geo_filter.Update()

    bb_aux_l_points = dijkstra_path(geo_filter.GetOutput(), point_1_id_endo, point_2_id_endo)
    length = len(bb_aux_l_points)
    la_connect_point = bb_aux_l_points[int(length * 0.5)]

    # ra
    geo_filter_la = vtk.vtkGeometryFilter()
    geo_filter_la.SetInputData(la_epi_surface)
    geo_filter_la.Update()
    la_epi_surface = geo_filter_la.GetOutput()

    geo_filter_ra = vtk.vtkGeometryFilter()
    geo_filter_ra.SetInputData(ra_epi_surface)
    geo_filter_ra.Update()
    ra_epi_surface = geo_filter_ra.GetOutput()

    loc_la_epi = vtk.vtkPointLocator()
    loc_la_epi.SetDataSet(la_epi_surface)
    loc_la_epi.BuildLocator()

    loc_ra_epi = vtk.vtkPointLocator()
    loc_ra_epi.SetDataSet(ra_epi_surface)
    loc_ra_epi.BuildLocator()

    la_connect_point_id = loc_la_epi.FindClosestPoint(la_connect_point)
    la_connect_point = la_epi_surface.GetPoint(la_connect_point_id)

    ra_connect_point_id = loc_ra_epi.FindClosestPoint(la_connect_point)
    ra_connect_point = ra_epi_surface.GetPoint(ra_connect_point_id)

    return la_connect_point, ra_connect_point

def get_bachmann_path_left(appendage_basis, lpv_sup_basis):
    la_mv_surface = smart_reader('../../Generate_Boundaries/LA/result/la_mv_surface.vtk')
    la_lpv_inf_surface = smart_reader('../../Generate_Boundaries/LA/result/la_lpv_inf_surface.vtk')
    endo = smart_reader('../../Generate_Boundaries/LA/result/la_endo_surface.vtk')
    epi = smart_reader('../../Generate_Boundaries/LA/result/la_epi_surface.vtk')

    loc_mv = vtk.vtkPointLocator()
    loc_mv.SetDataSet(la_mv_surface)
    loc_mv.BuildLocator()

    loc_epi = vtk.vtkPointLocator()
    loc_epi.SetDataSet(epi)
    loc_epi.BuildLocator()

    appendage_basis_id = loc_epi.FindClosestPoint(appendage_basis)
    lpv_sup_basis_id = loc_epi.FindClosestPoint(lpv_sup_basis)

    left_inf_pv_center = get_mean_point(la_lpv_inf_surface)
    point_l1_id = loc_mv.FindClosestPoint(left_inf_pv_center)
    point_l1 = la_mv_surface.GetPoint(point_l1_id)
    bb_mv_id = loc_epi.FindClosestPoint(point_l1)

    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(epi)
    geo_filter.Update()

    bb_1_points = dijkstra_path(geo_filter.GetOutput(), lpv_sup_basis_id, appendage_basis_id)
    bb_2_points = dijkstra_path(geo_filter.GetOutput(), appendage_basis_id, bb_mv_id)
    np.delete(bb_1_points, -1)
    bb_left = np.concatenate((bb_1_points, bb_2_points), axis=0)

    return bb_left, appendage_basis

def create_free_bridge_semi_auto(la_epi, ra_epi, ra_point, radius):
    loc_la_epi = vtk.vtkPointLocator()
    loc_la_epi.SetDataSet(la_epi)
    loc_la_epi.BuildLocator()
    point_end_id = loc_la_epi.FindClosestPoint(ra_point)
    point_end = la_epi.GetPoint(point_end_id)
    start = np.asarray(ra_point)
    end = np.asarray(point_end)
    point_data = np.vstack((start, end))
    tube = creat_tube_around_spline(point_data, radius)

    sphere_a = creat_sphere(start, radius * 1.02)
    sphere_b = creat_sphere(end, radius * 1.02)

    fiber_direction = end - start
    fiber_direction_norm = normalize_vector(fiber_direction)
    return tube, sphere_a, sphere_b, fiber_direction_norm

def creat_center_line(start_end_point):
    spline_points = vtk.vtkPoints()
    for i in range(len(start_end_point)):
        spline_points.InsertPoint(i, start_end_point[i][0], start_end_point[i][1], start_end_point[i][2])

    # Fit a spline to the points
    spline = vtk.vtkParametricSpline()
    spline.SetPoints(spline_points)

    functionSource = vtk.vtkParametricFunctionSource()
    functionSource.SetParametricFunction(spline)
    functionSource.SetUResolution(30 * spline_points.GetNumberOfPoints())
    functionSource.Update()
    tubePolyData = functionSource.GetOutput()
    points = tubePolyData.GetPoints().GetData()
    points = vtk.util.numpy_support.vtk_to_numpy(points)

    return points

def smart_bridge_writer(tube, sphere_1, sphere_2, name, job):
    meshNew = dsa.WrapDataObject(tube.GetOutput())
    writer = vtk.vtkOBJWriter()
    writer.SetFileName(job.ID+"/bridges/" + str(name) + "_tube.obj")
    writer.SetInputData(meshNew.VTKObject)
    writer.Write()

    meshNew = dsa.WrapDataObject(sphere_1.GetOutput())
    writer = vtk.vtkOBJWriter()
    writer.SetFileName(job.ID+"/bridges/" + str(name) + "_sphere_1.obj")
    writer.SetInputData(meshNew.VTKObject)
    writer.Write()

    meshNew = dsa.WrapDataObject(sphere_2.GetOutput())
    writer = vtk.vtkOBJWriter()
    writer.SetFileName(job.ID+"/bridges/" + str(name) + "_sphere_2.obj")
    writer.SetInputData(meshNew.VTKObject)
    writer.Write()