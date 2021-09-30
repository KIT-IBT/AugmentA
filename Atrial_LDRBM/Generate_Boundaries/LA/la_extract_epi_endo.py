#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
take the "equipotential surface" and LA.vtk as input

Output so far is vtk files: epi- and endo-surface

'''

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import scipy.spatial as spatial
from vtk.numpy_interface import dataset_adapter as dsa
import datetime


# appendage [22.561380859375, -35.339421875, 34.9769375]

def extract_epi_endo():

    # Read the atrial model
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName('model/LA.vtk')
    reader.Update()
    polydata = reader.GetOutput()
    points_data = polydata.GetPoints().GetData()
    atrial_points = vtk.util.numpy_support.vtk_to_numpy(points_data)

    # Read the contour or "equal potential surface"
    reader = vtk.vtkPolyDataReader()
    # reader.SetFileName('MATLAB/good_contour.vtk')
    reader.SetFileName('MATLAB/res_contour.vtk')
    reader.Update()
    contour_points_data = reader.GetOutput().GetPoints().GetData()
    contour_points = vtk.util.numpy_support.vtk_to_numpy(contour_points_data)
    print(contour_points)
    # Using cKDTress here, alternative we can use vtkKdTree
    point_tree = spatial.cKDTree(contour_points)
    atrial_points_tree = spatial.cKDTree(atrial_points)
    atrial_points_list = atrial_points.tolist()
    start_time = datetime.datetime.now()
    print('Extracting the surface close to contour... ' + str(start_time))
    # Using the query_ball_point
    # output_points = point_tree.query_ball_point(atrial_points, 0.9)
    output_points = atrial_points_tree.query_ball_tree(point_tree, 0.9)
    point_id_list = vtk.vtkIdList()
    mixed_points_list = []
    for i in range(len(output_points)):
        if len(output_points[i]) == 0:
            point_id_list.InsertNextId(i)
            mixed_points_list += [atrial_points_list[i]]
    end_time = datetime.datetime.now()
    running_time = end_time - start_time
    print('Extracting the surface close to contour...done ' + str(end_time) + '\nIt takes: ' + str(running_time) + '\n')

    # Get cell ids from LA file if cell contains points on the mixed_points_list
    cell_idlist = vtk.vtkIdList()
    cell_idlist_all = vtk.vtkIdList()
    for i in range(point_id_list.GetNumberOfIds()):
        polydata.GetPointCells(point_id_list.GetId(i), cell_idlist)
        for j in range(cell_idlist.GetNumberOfIds()):
            cell_idlist_all.InsertNextId(cell_idlist.GetId(j))

    # extract cells with specified cellids
    extractor = vtk.vtkExtractCells()
    extractor.SetInputData(polydata)
    extractor.AddCellList(cell_idlist_all)
    extractor.Update()
    extraction = extractor.GetOutput()
    print('Extracting epi surface and some pieces from endo...done\n \nWriting as la_epi_and_endo_pieces.vtk...')
    # Write epi surface and broken endo surface as vtk file
    meshNew = dsa.WrapDataObject(extraction)
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName("result_temp/la_epi_and_endo_pieces.vtk")
    writer.SetInputData(meshNew.VTKObject)
    writer.Write()
    print('Writing as la_epi_and_endo_pieces.vtk...done\n ')
    print('Extracting epi surface ...')

    # Extract the largest region, namely the epi
    connect = vtk.vtkConnectivityFilter()
    connect.SetInputData(extraction)
    connect.SetExtractionModeToLargestRegion()
    connect.Update()
    surface = connect.GetOutput()

    # ConnectivityFilter the unused points are not filtered.
    # clean the unused points
    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(surface)
    geo_filter.Update()
    epi_surface = geo_filter.GetOutput()

    cln = vtk.vtkCleanPolyData()
    cln.SetInputData(epi_surface)
    cln.Update()

    epi_new = cln.GetOutput()
    print('Extracting epi surface ... done\n \nWriting as la_epi_sruface.vtk...')

    # Write epi surface as vtk file
    meshNew = dsa.WrapDataObject(epi_new)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName("result/la_epi_surface.vtk")
    writer.SetInputData(meshNew.VTKObject)
    writer.Write()
    print('Writing as la_epi_surface.vtk...done\n ')

    """ Extract Endo surface """
    print('Extracting endo surface ...')
    cleanPolyData = cln.GetOutput()

    epi_surface_points = cleanPolyData.GetPoints().GetData()
    epi_surface_points = vtk.util.numpy_support.vtk_to_numpy(epi_surface_points)
    epi_surface_points = epi_surface_points.tolist()

    mixed_pieces = np.array(mixed_points_list)
    # slow
    # epi_points_list = [var for var in epi_surface_points if var in mixed_pieces]

    # Using Kd-tree
    first_tree = spatial.cKDTree(epi_surface_points)
    second__tree = spatial.cKDTree(mixed_pieces)
    output_points_2 = first_tree.query_ball_tree(second__tree, 0.001)

    pure_epi_point_list = []
    for i in range(len(output_points_2)):
        if len(output_points_2[i]) != 0:
            pure_epi_point_list += [epi_surface_points[i]]

    loc = vtk.vtkPointLocator()
    loc.SetDataSet(polydata)
    loc.BuildLocator()
    epi_point_id_list = vtk.vtkIdList()
    for i in range(len(pure_epi_point_list)):
        epi_point_id_list.InsertNextId(loc.FindClosestPoint(pure_epi_point_list[i]))

    cellid = vtk.vtkIdFilter()
    cellid.CellIdsOn()
    cellid.SetInputData(polydata) # vtkPolyData()
    cellid.PointIdsOff()
    cellid.SetIdsArrayName('Cell_ids')
    cellid.Update()
    temp_cell_ids = cellid.GetOutput().GetCellData().GetArray('Cell_ids')
    atrial_cell_id_all = vtk.util.numpy_support.vtk_to_numpy(temp_cell_ids)
    atrial_cell_id_all = atrial_cell_id_all.tolist()

    temp_epi_cell_id_list = vtk.vtkIdList()
    epi_cell_id_list= []
    for i in range(epi_point_id_list.GetNumberOfIds()):
        polydata.GetPointCells(epi_point_id_list.GetId(i), temp_epi_cell_id_list)
        for j in range(temp_epi_cell_id_list.GetNumberOfIds()):
            epi_cell_id_list += [temp_epi_cell_id_list.GetId(j)]

    endo_ids = list(set(atrial_cell_id_all).difference(set(epi_cell_id_list)))
    endo_cell_id_list = vtk.vtkIdList()
    for var in endo_ids:
        endo_cell_id_list.InsertNextId(var)

    extractor = vtk.vtkExtractCells()
    extractor.SetInputData(polydata)
    extractor.AddCellList(endo_cell_id_list)
    extractor.Update()
    endo_extraction = extractor.GetOutput()
    print('Extracting endo surface ... done\n \nWriting as la_endo_sruface.vtk...')
    meshNew = dsa.WrapDataObject(endo_extraction)
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName("result/la_endo_surface.vtk")
    writer.SetInputData(meshNew.VTKObject)
    writer.Write()
    print('Writing as la_endo_sruface.vtk...done\n ')

if __name__ == '__main__':
    run()
