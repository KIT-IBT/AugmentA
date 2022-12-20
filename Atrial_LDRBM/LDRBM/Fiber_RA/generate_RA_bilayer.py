#!/usr/bin/env python3

import os
import sys
import numpy as np
# from glob import glob
# import pandas as pd
import vtk
from vtk.util import numpy_support
from vtk.numpy_interface import dataset_adapter as dsa
from scipy.spatial import cKDTree
# from scipy import spatial
# import function
# from sklearn.neighbors import NearestNeighbors
import argparse

parser = argparse.ArgumentParser(description='Create Right Atrium.')

parser.add_argument('--mesh',
                    type=str,
                    default="../../../data/20220203/20220203_res_fibers/result_RA",
                    help='path to meshname')
parser.add_argument('--scale',
                    type=float,
                    default=1,
                    help='set 1 when the mesh is in mm or 1000 if it is in microns')

args = parser.parse_args()


def run(args):

    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(args.mesh + "/RA_CT_PMs.vtk")
    reader.Update()
    ra_endo = reader.GetOutput()

    # reader = vtk.vtkUnstructuredGridReader()
    # reader.SetFileName(args.mesh + "/RA_endo_with_fiber.vtk")
    # reader.Update()
    # ra_endo = reader.GetOutput()


    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(args.mesh + "/RA_epi_with_fiber.vtk")
    reader.Update()
    ra_epi = reader.GetOutput()

    endo = move_surf_along_normals(ra_endo, 0.1*args.scale, 1) # # Warning: set -1 if pts normals are pointing outside
    
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(args.mesh + "/RA_endo_with_fiber_moved.vtk")
    writer.SetInputData(endo)
    writer.Write()

    bilayer = generate_bilayer(endo, ra_epi, 0.12*args.scale)
    #bilayer = generate_bilayer(ra_endo, ra_epi)
    
    write_bilayer(bilayer)

def move_surf_along_normals(mesh, eps, direction):

    extract_surf = vtk.vtkGeometryFilter()
    extract_surf.SetInputData(mesh)
    extract_surf.Update()
    
    # reverse = vtk.vtkReverseSense()
    # reverse.ReverseCellsOn()
    # reverse.ReverseNormalsOn()
    # reverse.SetInputConnection(extract_surf.GetOutputPort())
    # reverse.Update()
    
    # polydata = reverse.GetOutput()
    polydata = extract_surf.GetOutput()
    
    normalGenerator = vtk.vtkPolyDataNormals()
    normalGenerator.SetInputData(polydata)
    normalGenerator.ComputeCellNormalsOff()
    normalGenerator.ComputePointNormalsOn()
    normalGenerator.ConsistencyOn()
    #normalGenerator.AutoOrientNormalsOn()
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


#Creates VTK and CARP files: .pts, .lon, .elem
def write_bilayer(bilayer):
    
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(args.mesh+"/RA_bilayer_with_fiber.vtk")
    writer.SetFileTypeToBinary()
    writer.SetInputData(bilayer)
    writer.Write()
    
    pts = numpy_support.vtk_to_numpy(bilayer.GetPoints().GetData())
    with open(args.mesh+'/RA_bilayer_with_fiber.pts',"w") as f:
        f.write("{}\n".format(len(pts)))
        for i in range(len(pts)):
            f.write("{} {} {}\n".format(pts[i][0], pts[i][1], pts[i][2]))
    
    tag_epi = vtk.util.numpy_support.vtk_to_numpy(bilayer.GetCellData().GetArray('elemTag'))

    with open(args.mesh+'/RA_bilayer_with_fiber.elem',"w") as f:
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
    
    with open(args.mesh+'/RA_bilayer_with_fiber.lon',"w") as f:
        f.write("2\n")
        for i in range(len(el_epi)):
            f.write("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(el_epi[i][0], el_epi[i][1], el_epi[i][2], sheet_epi[i][0], sheet_epi[i][1], sheet_epi[i][2]))

    print('Generated Bilayer RA')

if __name__ == '__main__':
    run(args)
