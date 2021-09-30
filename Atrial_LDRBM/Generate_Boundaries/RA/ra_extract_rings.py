#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Extract the feature edges from extracted endo surface
Then generate the rings

Input: extracted endo surface
Output: rings in vtk form; Ring points Id list
'''

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import scipy.spatial as spatial
from vtk.numpy_interface import dataset_adapter as dsa
import datetime


def run():
    """Extrating Rings"""
    print('Extracting rings...')
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName('result/ra_endo_surface.vtk')
    reader.Update()
    endo_surface = reader.GetOutput()

    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(endo_surface)
    geo_filter.Update()
    endo_surface = geo_filter.GetOutput()

    boundaryEdges = vtk.vtkFeatureEdges()
    boundaryEdges.SetInputData(endo_surface)
    boundaryEdges.BoundaryEdgesOn()
    boundaryEdges.FeatureEdgesOff()
    boundaryEdges.ManifoldEdgesOff()
    boundaryEdges.NonManifoldEdgesOff()
    boundaryEdges.Update()
    points = boundaryEdges.GetOutput().GetPoints().GetData()
    points = vtk.util.numpy_support.vtk_to_numpy(points)

    loc = vtk.vtkPointLocator()
    loc.SetDataSet(endo_surface)
    loc.BuildLocator()
    # Get all the point ids of points in mixed_points_list
    point_id_list = []
    for i in range(len(points)):
        temp_point_id_list = loc.FindClosestPoint(points[i, :])
        point_id_list += [temp_point_id_list]
    """Export Ring points Id list"""
    file_write_obj = open("data/ra_ring_edge_points_id.txt", 'w')
    for var in point_id_list:
        file_write_obj.writelines(str(var))
        file_write_obj.write('\n')
    file_write_obj.close()
    #
    cell_idlist = vtk.vtkIdList()
    cell_idlist_all = vtk.vtkIdList()
    for i in range(len(point_id_list)):
        endo_surface.GetPointCells(point_id_list[i], cell_idlist)
        for j in range(cell_idlist.GetNumberOfIds()):
            cell_idlist_all.InsertNextId(cell_idlist.GetId(j))

    extractor = vtk.vtkExtractCells()
    extractor.SetInputData(endo_surface)
    extractor.AddCellList(cell_idlist_all)
    extractor.Update()
    extraction = extractor.GetOutput()

    # delete the ring cells on the endo surface
    cellid = vtk.vtkIdFilter()
    cellid.CellIdsOn()
    cellid.SetInputData(endo_surface)  # vtkPolyData()
    cellid.PointIdsOff()
    cellid.SetIdsArrayName('Cell_ids')
    cellid.Update()
    temp_cell_ids = cellid.GetOutput().GetCellData().GetArray('Cell_ids')
    ID_all = vtk.util.numpy_support.vtk_to_numpy(temp_cell_ids)
    ID_all = ID_all.tolist()
    ring_idlist_all = []
    for i in range(cell_idlist_all.GetNumberOfIds()):
        ring_idlist_all += [cell_idlist_all.GetId(i)]
    new_endo_id_list = list(set(ID_all).difference(set(ring_idlist_all)))
    new_endo_cell_id_list = vtk.vtkIdList()
    for i in range(len(new_endo_id_list)):
        new_endo_cell_id_list.InsertNextId(new_endo_id_list[i])

    # print(new_endo_cell_id_list)
    extractor = vtk.vtkExtractCells()
    extractor.SetInputData(endo_surface)
    extractor.AddCellList(new_endo_cell_id_list)
    extractor.Update()
    new_endo = extractor.GetOutput()

    print('Extracting rings...done\n ')
    print('Writing the endo surface without rings as ra_endo_without_rings_surface.vtk...')
    meshNew = dsa.WrapDataObject(new_endo)
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName("result/ra_endo_without_rings_surface.vtk")
    writer.SetInputData(meshNew.VTKObject)
    writer.Write()
    print('Writing the endo surface without rings as ra_endo_without_rings_surface.vtk...done\n ')
    print('Writing the rings as rings.vtk...')

    meshNew = dsa.WrapDataObject(extraction)
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName("result/ra_rings.vtk")
    writer.SetInputData(meshNew.VTKObject)
    writer.Write()
    print('Writing the rings as ra_rings.vtk...done\n')


if __name__ == '__main__':
    run()
