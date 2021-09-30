#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Extract the feature edges from extracted endo surface
Then generate the rings

Input: extracted endo surface
Output: rings in vtk form; Ring points Id list
'''
import os
import numpy as np
from glob import glob
import vtk
from vtk.util import numpy_support
import scipy.spatial as spatial
from vtk.numpy_interface import dataset_adapter as dsa
import datetime
from sklearn.cluster import KMeans
import argparse

parser = argparse.ArgumentParser(description='Generate boundaries.')
parser.add_argument('--mesh',
                    type=str,
                    default='LA_new',
                    help='path to meshname')

parser.add_argument('--ap',
                    type=int,
                    default=0,
                    help='Apex ponts index')

args = parser.parse_args()


class Ring:
   def __init__(self, index, name, points_num, center_point, distance, polydata):
       self.id = index
       self.name = name
       self.np = points_num
       self.center = center_point
       self.ap_dist = distance
       self.vtk_polydata = polydata
      
def run(args):
    """Extrating Rings"""
    print('Extracting rings...')
    
    data_checker = vtk.vtkDataSetReader()
    data_checker.SetFileName('result/{}.vtk'.format(args.mesh))
    data_checker.Update()
    
    if data_checker.IsFilePolyData():
        reader = vtk.vtkPolyDataReader()
    elif data_checker.IsFileUnstructuredGrid():
        reader = vtk.vtkUnstructuredGridReader()
        
    reader.SetFileName('result/{}.vtk'.format(args.mesh))
    reader.Update()
    
    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputConnection(reader.GetOutputPort())
    geo_filter.Update()
    
    mesh_surf = geo_filter.GetOutput()
    ap_point = mesh_surf.GetPoint(args.ap)
    
    idFilter = vtk.vtkIdFilter()
    idFilter.SetInputConnection(geo_filter.GetOutputPort())
    idFilter.SetIdsArrayName("Ids")
    idFilter.SetPointIds(True)
    idFilter.SetCellIds(False)
    idFilter.Update()
    
    boundaryEdges = vtk.vtkFeatureEdges()
    boundaryEdges.SetInputConnection(idFilter.GetOutputPort())
    boundaryEdges.BoundaryEdgesOn()
    boundaryEdges.FeatureEdgesOff()
    boundaryEdges.ManifoldEdgesOff()
    boundaryEdges.NonManifoldEdgesOff()
    boundaryEdges.Update()
    
    "Splitting rings"
    
    connect = vtk.vtkConnectivityFilter()
    connect.SetInputData(boundaryEdges.GetOutput())
    connect.SetExtractionModeToAllRegions()
    connect.Update()
    num = connect.GetNumberOfExtractedRegions()
    
    connect.SetExtractionModeToSpecifiedRegions()
    
    rings = []
    
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
        
        ring_surf = vtk.vtkPolyData()
        ring_surf.DeepCopy(surface)
        
        centerOfMassFilter = vtk.vtkCenterOfMass()
        centerOfMassFilter.SetInputData(surface)
        centerOfMassFilter.SetUseScalarsAsWeights(False)
        centerOfMassFilter.Update()
        
        c_mass = centerOfMassFilter.GetCenter()
        
        ring = Ring(i,"",surface.GetNumberOfPoints(), c_mass, np.sqrt(np.sum((np.array(ap_point)- \
                    np.array(c_mass))**2, axis=0)), ring_surf)
        rings.append(ring)
        
        connect.DeleteSpecifiedRegion(i)
        connect.Update()
    
    rings[np.argmax([r.np for r in rings])].name = "MV"
    pvs = [i for i in range(len(rings)) if rings[i].name!="MV"]
    
    estimator = KMeans(n_clusters=2)
    estimator.fit([r.center for r in rings if r.name!="MV"])
    label_pred = estimator.labels_
    
    min_ap_dist = np.argmin([r.ap_dist for r in [rings[i] for i in pvs]])
    label_LPV = label_pred[min_ap_dist]
    
    LPVs = [pvs[i] for i in np.where(label_pred == label_LPV)[0]]
    LSPV_id = LPVs.index(min_ap_dist)
    RPVs = [pvs[i] for i in np.where(label_pred != label_LPV)[0]]
    min_ap_dist_RPVs = np.argmin([r.ap_dist for r in [rings[i] for i in RPVs]])
    RSPV_id = LPVs.index(min_ap_dist_RPVs)
    
    estimator = KMeans(n_clusters=2)
    estimator.fit([r.center for r in [rings[i] for i in LPVs]])
    LPV_lab = estimator.labels_
    LSPVs = [LPVs[i] for i in np.where(LPV_lab == LPV_lab[LSPV_id])[0]]
    LIPVs = [LPVs[i] for i in np.where(LPV_lab != LPV_lab[LSPV_id])[0]]
    
    estimator = KMeans(n_clusters=2)
    estimator.fit([r.center for r in [rings[i] for i in RPVs]])
    RPV_lab = estimator.labels_
    RSPVs = [RPVs[i] for i in np.where(RPV_lab == RPV_lab[RSPV_id])[0]]
    RIPVs = [RPVs[i] for i in np.where(RPV_lab != RPV_lab[RSPV_id])[0]]
    
    for i in range(len(pvs)):
        if pvs[i] in LSPVs:
            rings[pvs[i]].name = "LSPV"
        elif pvs[i] in LIPVs:
            rings[pvs[i]].name = "LIPV"
        elif pvs[i] in RIPVs:
            rings[pvs[i]].name = "RIPV"
        else:
            rings[pvs[i]].name = "RSPV"
    
    fname = glob('Surf/{}*'.format(args.mesh))
    
    b_tag = np.zeros((reader.GetOutput().GetNumberOfPoints(),))
    
    for r in fname:
        os.remove(r)
    
    for r in rings:
        id_vec = numpy_support.vtk_to_numpy(r.vtk_polydata.GetPointData().GetArray("Ids"))
        fname = 'Surf/{}_{}.vtx'.format(args.mesh,r.name)
        if os.path.exists(fname):
            f = open(fname, 'a')
        else:
            f = open(fname, 'w')
            f.write('{}\n'.format(len(id_vec)))
            f.write('extra\n')
        for i in id_vec:
            f.write('{}\n'.format(i))
            if r.name == "MV":
                b_tag[i] = 1
            elif r.name == "LIPV":
                b_tag[i] = 2
            elif r.name == "LSPV":
                b_tag[i] = 3
            elif r.name == "RIPV":
                b_tag[i] = 4
            else:
                b_tag[i] = 5
        f.close()
    
    dataSet = dsa.WrapDataObject(mesh_surf)
    dataSet.PointData.append(b_tag, 'boundary_tag')
    
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(dataSet.VTKObject)
    writer.SetFileName('result/{}_boundaries_tagged.vtk'.format(args.mesh))
    writer.Write()
    
if __name__ == '__main__':
    run(args)
