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
import numpy as np
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.util import numpy_support
import Methods_LA as Method
import csv
import datetime
import pandas as pd
from la_laplace import laplace_0_1
import os

EXAMPLE_DIR = os.path.dirname(os.path.realpath(__file__))

vtk_version = vtk.vtkVersion.GetVTKSourceVersion().split()[-1].split('.')[0]
      
def la_generate_fiber(model, args, job):
    
    # size(Radius) of Bachmann Bundle in mm
    w_bb = 2*args.scale
    
    simid = job.ID+"/result_LA"
    try:
        os.makedirs(simid)
    except OSError:
        print ("Creation of the directory %s failed" % simid)
    else:
        print ("Successfully created the directory %s " % simid)
            
    with open(os.path.join(EXAMPLE_DIR,'../../element_tag.csv')) as f:
        tag_dict = {}
        reader = csv.DictReader(f)
        for row in reader:
            tag_dict[row['name']] = row['tag']
    # load epi tags
    mitral_valve_epi = int(tag_dict['mitral_valve_epi'])
    superior_left_pulmonary_vein_epi = int(tag_dict['superior_left_pulmonary_vein_epi'])
    inferior_left_pulmonary_vein_epi = int(tag_dict['inferior_left_pulmonary_vein_epi'])
    superior_right_pulmonary_vein_epi = int(tag_dict['superior_right_pulmonary_vein_epi'])
    inferior_right_pulmonary_vein_epi = int(tag_dict['inferior_right_pulmonary_vein_epi'])
    left_atrial_appendage_epi = int(tag_dict['left_atrial_appendage_epi'])
    left_atrial_wall_epi = int(tag_dict['left_atrial_wall_epi'])
    left_atrial_lateral_wall_epi = int(tag_dict['left_atrial_lateral_wall_epi'])
    left_septum_wall_epi = int(tag_dict['left_septum_wall_epi'])
    left_atrial_roof_epi = int(tag_dict['left_atrial_roof_epi'])

    # load endo tags
    mitral_valve_endo = int(tag_dict['mitral_valve_endo'])
    superior_left_pulmonary_vein_endo = int(tag_dict['superior_left_pulmonary_vein_endo'])
    inferior_left_pulmonary_vein_endo = int(tag_dict['inferior_left_pulmonary_vein_endo'])
    superior_right_pulmonary_vein_endo = int(tag_dict['superior_right_pulmonary_vein_endo'])
    inferior_right_pulmonary_vein_endo = int(tag_dict['inferior_right_pulmonary_vein_endo'])
    left_atrial_appendage_endo = int(tag_dict['left_atrial_appendage_endo'])
    left_atrial_wall_endo = int(tag_dict['left_atrial_wall_endo'])
    left_atrial_lateral_wall_endo = int(tag_dict['left_atrial_lateral_wall_endo'])
    left_septum_wall_endo = int(tag_dict['left_septum_wall_endo'])
    left_atrial_roof_endo = int(tag_dict['left_atrial_roof_endo'])
    bachmann_bundel_left = int(tag_dict['bachmann_bundel_left'])

    tao_mv = 0.85

    # ab
    ab = model.GetCellData().GetArray('phie_ab')
    ab_grad = model.GetCellData().GetArray('grad_ab')
    ab = vtk.util.numpy_support.vtk_to_numpy(ab)
    ab_grad = vtk.util.numpy_support.vtk_to_numpy(ab_grad)

    # v
    v = model.GetCellData().GetArray('phie_v')
    v_grad = model.GetCellData().GetArray('grad_v')
    v = vtk.util.numpy_support.vtk_to_numpy(v)
    v_grad = vtk.util.numpy_support.vtk_to_numpy(v_grad)

    # r
    r = model.GetCellData().GetArray('phie_r')
    r_grad = model.GetCellData().GetArray('grad_r')
    r = vtk.util.numpy_support.vtk_to_numpy(r)
    r_grad = vtk.util.numpy_support.vtk_to_numpy(r_grad)
    
    # r2
    r2 = model.GetCellData().GetArray('phie_r2')
    r2 = vtk.util.numpy_support.vtk_to_numpy(r2)

    # phie
    if args.mesh_type == "vol":
        phie = model.GetCellData().GetArray('phie_phi')
        phie = vtk.util.numpy_support.vtk_to_numpy(phie)
    
    phie_grad = model.GetCellData().GetArray('grad_phi')
    phie_grad = vtk.util.numpy_support.vtk_to_numpy(phie_grad)

    cellid = vtk.vtkIdFilter()
    cellid.CellIdsOn()
    cellid.SetInputData(model)
    cellid.PointIdsOn()
    if int(vtk_version) >= 9:
        cellid.SetPointIdsArrayName('Global_ids')
        cellid.SetCellIdsArrayName('Global_ids')
    else:
        cellid.SetIdsArrayName('Global_ids')
    cellid.Update()
    
    model = cellid.GetOutput()
    
    df = pd.read_csv(args.mesh+"_surf/rings_centroids.csv")
    
    # LPV
    lb = 0
    ub = 0.4
    tao_lpv = Method.find_tau(model, ub, lb, "low", "phie_v")
    print('Calculating tao_lpv done! tap_lpv = ', tao_lpv)
    
    thr = Method.vtk_thr(model, 1, "CELLS","phie_v",tao_lpv)
    connect = vtk.vtkConnectivityFilter()
    connect.SetInputData(thr)
    connect.SetExtractionModeToAllRegions()
    connect.Update()

    PVs = dict()
    # Distinguish between LIPV and LSPV
    PVs = Method.distinguish_PVs(connect, PVs, df, "LIPV", "LSPV")

    model = laplace_0_1(args, job, model, "RPV", "LAA", "phie_ab2")

    thr = Method.vtk_thr(model, 1, "CELLS","phie_v", tao_lpv)
    #thr = Method.vtk_thr(model, 1, "CELLS","phie_v", 0.35)
    #found, val = Method.optimize_shape_PV(thr, 10, 0)
    #print('Calculating opt_tao_lpv done! tap_lpv = ', val)
    #if val!=0.35:
    #    thr = Method.vtk_thr(model, 1, "CELLS","phie_v",val)
    
    phie_r2_tau_lpv = vtk.util.numpy_support.vtk_to_numpy(thr.GetCellData().GetArray('phie_r2'))
    max_phie_r2_tau_lpv = np.max(phie_r2_tau_lpv)
    
    phie_ab_tau_lpv = vtk.util.numpy_support.vtk_to_numpy(thr.GetPointData().GetArray('phie_ab2'))
    max_phie_ab_tau_lpv = np.max(phie_ab_tau_lpv)

    print("max_phie_r2_tau_lpv ",max_phie_r2_tau_lpv)
    print("max_phie_ab_tau_lpv ",max_phie_ab_tau_lpv)
    
    # RPV
    lb = 0.6
    ub = 1
    tao_rpv = Method.find_tau(model, ub, lb, "up", "phie_v")
    print('Calculating tao_rpv done! tap_rpv = ', tao_rpv)
    
    thr = Method.vtk_thr(model, 0, "CELLS","phie_v",tao_rpv)

    connect = vtk.vtkConnectivityFilter()
    connect.SetInputData(thr)
    connect.SetExtractionModeToAllRegions()
    connect.Update()
    
    # Distinguish between RIPV and RSPV
    PVs = Method.distinguish_PVs(connect, PVs, df, "RIPV", "RSPV")

    start_time = datetime.datetime.now()
    print('Calculating fibers... ' + str(start_time))
    
    #### Bundles selection ####
    # # Volume mesh
    # if 0:# args.mesh_type == 'vol':
    #     tag = np.zeros(len(r))
        
    #     for i in range(len(ab_grad)):
    #         # tagging endo-layer
    #         if phie[i] <= 0.5:
    #             if r[i] >= tao_mv:
    #                 tag[i] = mitral_valve_endo
    #             elif ab[i] < max_phie_ab_tau_lpv+0.01 and r2[i]>max_phie_r2_tau_lpv+0.01:
    #                 tag[i] = left_atrial_appendage_endo
    #             elif v[i] <= tao_lpv and i in PVs["LIPV"]:
    #                 tag[i] = inferior_left_pulmonary_vein_endo
    #             elif v[i] <= tao_lpv and i in PVs["LSPV"]:
    #                 tag[i] = superior_left_pulmonary_vein_endo
    #             elif v[i] >= tao_rpv and i in PVs["RIPV"]:
    #                 tag[i] = inferior_right_pulmonary_vein_endo
    #             elif v[i] >= tao_rpv and i in PVs["RSPV"]:
    #                 tag[i] = superior_right_pulmonary_vein_endo
    #             else:
    #                 tag[i] = left_atrial_wall_endo
    #         # tagging epi-layer
    #         else:
    #             if r[i] >= tao_mv:
    #                 ab_grad[i] = r_grad[i]
    #                 tag[i] = mitral_valve_epi
    #             elif ab[i] < max_phie_ab_tau_lpv+0.01 and r2[i]>max_phie_r2_tau_lpv+0.01:
    #                 tag[i] = left_atrial_appendage_epi
    #             elif v[i] <= tao_lpv and i in PVs["LIPV"]:
    #                 ab_grad[i] = v_grad[i]
    #                 tag[i] = inferior_left_pulmonary_vein_epi
    #             elif v[i] <= tao_lpv and i in PVs["LSPV"]:
    #                 ab_grad[i] = v_grad[i]
    #                 tag[i] = superior_left_pulmonary_vein_epi
    #             elif v[i] >= tao_rpv and i in PVs["RIPV"]:
    #                 ab_grad[i] = v_grad[i]
    #                 tag[i] = inferior_right_pulmonary_vein_epi
    #             elif v[i] >= tao_rpv and i in PVs["RSPV"]:
    #                 ab_grad[i] = v_grad[i]
    #                 tag[i] = superior_right_pulmonary_vein_epi
    #             else:
    #                 tag[i] = left_atrial_wall_epi
        
    #     meshNew = dsa.WrapDataObject(model)
    #     meshNew.CellData.append(tag, "elemTag")
    #     model = meshNew.VTKObject
        
    #     # normlize the gradient phie
    #     abs_phie_grad = np.linalg.norm(phie_grad, axis=1, keepdims=True)
    #     abs_phie_grad = np.where(abs_phie_grad != 0, abs_phie_grad, 1)
    #     phie_grad_norm = phie_grad / abs_phie_grad
    
    #     ##### Local coordinate system #####
    #     # et
    #     et = phie_grad_norm
    #     print('############### et ###############')
    #     print(et)
    
    #     # k
    #     k = ab_grad
    #     print('############### k ###############')
    #     print(k)
    
    #     # en
    #     en = ab_grad
    #     #for i in range(len(k)):
    #     #    en[i] = k[i] - np.dot(k[i], et[i]) * et[i]
            
    #     en = k - np.dot(k, et) * et
    #     # normalize the en
    #     abs_en = np.linalg.norm(en, axis=1, keepdims=True)
    #     abs_en = np.where(abs_en != 0, abs_en, 1)
    #     en = en / abs_en
    #     print('############### en ###############')
    #     print(en)
    
    #     # el
    #     el = np.cross(en, et)
    #     print('############### el ###############')
    #     print(el)
    
    #     abs_v_grad = np.linalg.norm(v_grad, axis=1, keepdims=True)
    #     abs_v_grad = np.where(abs_v_grad != 0, abs_v_grad, 1)
    #     v_grad_norm = v_grad / abs_v_grad
    #     ### Subendo bundle selection
    #     for i in range(len(v_grad_norm)):
    #         if phie[i] < 0.5 and v[i] <= 0.2:
    #             el[i] = v_grad_norm[i]
    
    #     end_time = datetime.datetime.now()
    #     running_time = end_time - start_time
    #     print('Calculating fibers... done! ' + str(end_time) + '\nIt takes: ' + str(running_time) + '\n')
    
    #     reader = vtk.vtkUnstructuredGridReader()
    #     reader.SetFileName('Mesh/LA_mesh.vtk')
    #     reader.Update()
    #     model = reader.GetOutput()
    
    #     print("Creating bachmann bundles...")
        
    #     # Extract surface
    #     geo_filter = vtk.vtkGeometryFilter()
    #     geo_filter.SetInputData(model)
    #     geo_filter.PassThroughPointIdsOn()
    #     geo_filter.Update()
    #     # Get only epi
    #     thr = Method.vtk_thr(geo_filter.GetOutput(), 2, "CELLS","elemTag",mitral_valve_epi, left_atrial_roof_epi)
    #     # Bachmann Bundle
        
    #     # Get ending point of wide BB
    #     # Extract the MV
    #     thr = Method.vtk_thr(thr, 2, "CELLS","elemTag",mitral_valve_epi, mitral_valve_epi)
        
    #     # Get the closest point to the inferior appendage base in the MV
    #     loc = vtk.vtkPointLocator()
    #     loc.SetDataSet(thr)
    #     loc.BuildLocator()
    
    #     bb_mv_id =  int(model.GetPointData().GetArray('Global_ids').GetTuple(loc.FindClosestPoint(model.GetPoint(inf_appendage_basis_id)))[0])
        
    #     thresh = vtk.vtkThreshold()
    #     thresh.SetInputData(model)
    #     thresh.ThresholdBetween(max_phie_ab_tau_lpv-0.1, max_phie_ab_tau_lpv+0.1)
    #     thresh.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_CELLS", "phie_ab")
    #     thresh.Update()
        
    #     loc = vtk.vtkPointLocator()
    #     loc.SetDataSet(thresh.GetOutput())
    #     loc.BuildLocator()
    
    #     inf_appendage_basis_id = int(thresh.GetOutput().GetPointData().GetArray('Global_ids').GetTuple(loc.FindClosestPoint(df["MV"].to_numpy()))[0])
        
    #     thresh = vtk.vtkThreshold()
    #     thresh.SetInputData(thresh.GetOutput())
    #     thresh.ThresholdBetween(max_phie_r2_tau_lpv-0.1, max_phie_r2_tau_lpv+0.1)
    #     thresh.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_CELLS", "phie_r2")
    #     thresh.Update()
        
    #     loc = vtk.vtkPointLocator()
    #     loc.SetDataSet(thresh.GetOutput())
    #     loc.BuildLocator()
    
    #     sup_appendage_basis_id = int(thresh.GetOutput().GetPointData().GetArray('Global_ids').GetTuple(loc.FindClosestPoint(df["LAA"].to_numpy()))[0])
    
    
    #     bb_left = Method.get_wide_bachmann_path_left(thresh.GetOutput(), inf_appendage_basis_id, sup_appendage_basis_id, bb_mv_id)
        
    #     tag = Method.assign_element_tag_around_path_within_radius(model, bb_left, 2, tag, bachmann_bundel_left)
    #     el = Method.assign_element_fiber_around_path_within_radius(model, bb_left, 2, el, smooth=True)
    
    #     # # Bachmann_Bundle internal connection
    #     # la_connect_point, ra_connect_point = Method.get_connection_point_la_and_ra(la_appex_point)
    #     # la_connect_point = np.asarray(la_connect_point)
    #     #
    #     # path_start_end = np.vstack((appendage_basis, la_connect_point))
    #     # path_bb_ra = Method.creat_center_line(path_start_end)
    #     # tag = Method.assign_element_tag_around_path_within_radius(model, path_bb_ra, 2, tag, bachmann_bundel_left)
    #     # el = Method.assign_element_fiber_around_path_within_radius(model, path_bb_ra, 2, el, smooth=True)
    
    #     print("Creating bachmann bundles... done")
    #     abs_el = np.linalg.norm(el, axis=1, keepdims=True)
    #     interpolate_arr = np.asarray([0, 0, 1])
    #     index = np.argwhere(abs_el == 0)
    #     print('There is',len(index),'zero vector(s).')
    #     for var in index:
    #         el[var[0]] = interpolate_arr
    
    #     #### save the result into vtk ####
    #     start_time = datetime.datetime.now()
    #     print('Writinga as LA_with_fiber... ' + str(start_time))
    
    #     # fiber_data = vtk.util.numpy_support.numpy_to_vtk(el, deep=True, array_type=vtk.VTK_DOUBLE)
    #     # fiber_data.SetNumberOfComponents(3)
    #     # fiber_data.SetName("fiber")
    #     # model.GetCellData().SetVectors(fiber_data)  # AddArray(fiber_data)
    
    #     # tag_data = vtk.util.numpy_support.numpy_to_vtk(tag, deep=True, array_type=vtk.VTK_DOUBLE)
    #     # tag_data.SetNumberOfComponents(1)
    #     # tag_data.SetName("elemTag")
    #     # model.GetCellData().RemoveArray("Cell_ids")
    #     # model.GetCellData().RemoveArray("elemTag")
    #     # model.GetCellData().SetScalars(tag_data)

    #     meshNew = dsa.WrapDataObject(model)
    #     meshNew.CellData.append(tag, "elemTag")
    #     meshNew.CellData.append(el, "fiber")
    
    #     model = meshNew.VTKObject
    
    #     end_time = datetime.datetime.now()
    #     running_time = end_time - start_time
    #     print('Writing as LA_with_fiber... done! ' + str(end_time) + '\nIt takes: ' + str(running_time) + '\n')
        
    
    # Bilayer mesh
    if args.mesh_type == 'bilayer':
        epi = vtk.vtkUnstructuredGrid()
        epi.DeepCopy(model)
        
        tag_epi = np.zeros(len(r), dtype=int)
        tag_epi[:] = left_atrial_wall_epi
        tag_endo = np.zeros(len(r), dtype=int)
        tag_endo[:] = left_atrial_wall_endo

    else: # Volume mesh
        tag = np.ones(len(r), dtype=int)*left_atrial_wall_epi

        epi = Method.vtk_thr(model,0,"CELLS","phie_phi",0.5)

        epi_ids = vtk.util.numpy_support.vtk_to_numpy(epi.GetCellData().GetArray('Global_ids'))

        endo_ids = np.arange(len(r)).astype(int)

        endo_ids = np.setdiff1d(endo_ids, epi_ids)

        tag[endo_ids] = left_atrial_wall_endo
        
    LAA_s = Method.vtk_thr(model,0,"CELLS","phie_r2",max_phie_r2_tau_lpv+0.01)
    
    LAA_s = Method.vtk_thr(LAA_s,0,"POINTS","phie_ab2",max_phie_ab_tau_lpv-0.03)
    
    ## Optimize shape of LAA solving a laplacian with 0 in LAA and 1 in the boundary of LAA_s
    
    LAA_bb = Method.vtk_thr(model,2,"POINTS","phie_ab2",max_phie_ab_tau_lpv-0.03, max_phie_ab_tau_lpv+0.01)
    
    LAA_bb_ids = vtk.util.numpy_support.vtk_to_numpy(LAA_bb.GetPointData().GetArray('Global_ids'))
    
    MV_ring_ids = np.loadtxt('{}_surf/ids_MV.vtx'.format(args.mesh), skiprows=2, dtype=int)

    LAA_bb_ids = np.append(LAA_bb_ids, MV_ring_ids)

    fname = '{}_surf/ids_LAA_bb.vtx'.format(args.mesh)
    f = open(fname, 'w')
    f.write('{}\n'.format(len(LAA_bb_ids)))
    f.write('extra\n')
    for i in LAA_bb_ids:
        f.write('{}\n'.format(i))
    f.close()
    
    LAA_s = laplace_0_1(args, job, model, "LAA", "LAA_bb", "phie_ab3")
    
    LAA_s = Method.vtk_thr(LAA_s,1,"POINTS","phie_ab3",0.95)
    
    ring_ids = np.loadtxt('{}_surf/'.format(args.mesh) + 'ids_MV.vtx', skiprows=2, dtype=int)
    
    rings_pts = vtk.util.numpy_support.vtk_to_numpy(model.GetPoints().GetData())[ring_ids,:]
    
    MV_ids = Method.get_element_ids_around_path_within_radius(model, rings_pts, 4*args.scale)
    
    LAA_ids = vtk.util.numpy_support.vtk_to_numpy(LAA_s.GetCellData().GetArray('Global_ids'))
    
    # tagging endo-layer
    if args.mesh_type == 'bilayer':
        tag_endo[MV_ids] = mitral_valve_endo
        ab_grad[MV_ids] = r_grad[MV_ids]
        
        tag_endo[LAA_ids] = left_atrial_appendage_endo
        
        tag_endo[PVs["RIPV"]] = inferior_right_pulmonary_vein_endo
        tag_endo[PVs["LIPV"]] = inferior_left_pulmonary_vein_endo
        
        tag_endo[PVs["RSPV"]] = superior_right_pulmonary_vein_endo
        tag_endo[PVs["LSPV"]] = superior_left_pulmonary_vein_endo
        
        ab_grad[PVs["RIPV"]] = v_grad[PVs["RIPV"]]
        ab_grad[PVs["LIPV"]] = v_grad[PVs["LIPV"]]
        ab_grad[PVs["RSPV"]] = v_grad[PVs["RSPV"]]
        ab_grad[PVs["LSPV"]] = v_grad[PVs["LSPV"]]
        
        # tagging epi-layer
        
        tag_epi[MV_ids] = mitral_valve_epi
        
        tag_epi[LAA_ids] = left_atrial_appendage_epi

        tag_epi[PVs["RIPV"]] = inferior_right_pulmonary_vein_epi
        tag_epi[PVs["LIPV"]] = inferior_left_pulmonary_vein_epi
        
        tag_epi[PVs["RSPV"]] = superior_right_pulmonary_vein_epi
        tag_epi[PVs["LSPV"]] = superior_left_pulmonary_vein_epi
    
    
        ab_grad_epi = np.copy(ab_grad)

    else:

        MV_ids_endo = np.intersect1d(MV_ids, endo_ids)
        tag[MV_ids_endo] = mitral_valve_endo
        ab_grad[MV_ids_endo] = r_grad[MV_ids_endo]
        
        LAA_ids_endo = np.intersect1d(LAA_ids, endo_ids)
        tag[LAA_ids_endo] = left_atrial_appendage_endo
        
        RIPV_ids_endo = np.intersect1d(PVs["RIPV"], endo_ids)
        tag[RIPV_ids_endo] = inferior_right_pulmonary_vein_endo
        LIPV_ids_endo = np.intersect1d(PVs["LIPV"], endo_ids)
        tag[LIPV_ids_endo] = inferior_left_pulmonary_vein_endo
        RSPV_ids_endo = np.intersect1d(PVs["RSPV"], endo_ids)
        tag[RSPV_ids_endo] = superior_right_pulmonary_vein_endo
        LSPV_ids_endo = np.intersect1d(PVs["LSPV"], endo_ids)
        tag[LSPV_ids_endo] = superior_left_pulmonary_vein_endo
        
        ab_grad[RIPV_ids_endo] = v_grad[RIPV_ids_endo]
        ab_grad[RSPV_ids_endo] = v_grad[RSPV_ids_endo]
        ab_grad[LIPV_ids_endo] = v_grad[LIPV_ids_endo]
        ab_grad[LSPV_ids_endo] = v_grad[LSPV_ids_endo]
        
        # tagging epi-layer
        
        MV_ids_epi = np.intersect1d(MV_ids, epi_ids)
        tag[MV_ids_epi] = mitral_valve_epi
        
        LAA_ids_epi = np.intersect1d(LAA_ids, epi_ids)
        tag[LAA_ids_epi] = left_atrial_appendage_epi

        RIPV_ids_epi = np.intersect1d(PVs["RIPV"], epi_ids)
        tag[RIPV_ids_epi] = inferior_right_pulmonary_vein_epi
        LIPV_ids_epi = np.intersect1d(PVs["LIPV"], epi_ids)
        tag[LIPV_ids_epi] = inferior_left_pulmonary_vein_epi
        RSPV_ids_epi = np.intersect1d(PVs["RSPV"], epi_ids)
        tag[RSPV_ids_epi] = superior_right_pulmonary_vein_epi
        LSPV_ids_epi = np.intersect1d(PVs["LSPV"], epi_ids)
        tag[LSPV_ids_epi] = superior_left_pulmonary_vein_epi
    
    # Get epi bundle band
    
    lpv_mean = np.mean([df["LIPV"].to_numpy(), df["LSPV"].to_numpy()], axis = 0)
    rpv_mean = np.mean([df["RIPV"].to_numpy(), df["RSPV"].to_numpy()], axis = 0)
    mv_mean = df["MV"].to_numpy()
    
    v1 = rpv_mean - mv_mean
    v2 = lpv_mean - mv_mean
    norm = np.cross(v2, v1)
    
    norm = norm / np.linalg.norm(norm)
    
    band_s = Method.vtk_thr(epi,0,"CELLS","phie_r2",max_phie_r2_tau_lpv)
    
    plane = vtk.vtkPlane()
    plane.SetNormal(norm[0], norm[1], norm[2])
    plane.SetOrigin(mv_mean[0], mv_mean[1], mv_mean[2])

    meshExtractFilter = vtk.vtkExtractGeometry()
    meshExtractFilter.SetInputData(band_s)
    meshExtractFilter.SetImplicitFunction(plane)
    meshExtractFilter.Update()
    
    band_cell_ids = vtk.util.numpy_support.vtk_to_numpy(meshExtractFilter.GetOutput().GetCellData().GetArray('Global_ids'))
    
    if args.mesh_type == "bilayer":
        ab_grad_epi[band_cell_ids] = r_grad[band_cell_ids]

        meshNew = dsa.WrapDataObject(model)
        meshNew.CellData.append(tag_endo, "elemTag")
        endo = meshNew.VTKObject
        
        meshNew = dsa.WrapDataObject(epi)
        meshNew.CellData.append(tag_epi, "elemTag")
        epi = meshNew.VTKObject

        # normalize the gradient phie
        phie_grad_norm = phie_grad

        ##### Local coordinate system #####
        # et
        et = phie_grad_norm
        print('############### et ###############')

        # k
        k_endo = ab_grad
        k_epi = ab_grad_epi
        print('############### k ###############')
        
        en_endo = k_endo - et*np.sum(k_endo*et,axis=1).reshape(len(et),1)
        en_epi = k_epi - et*np.sum(k_epi*et,axis=1).reshape(len(et),1)
        
        # normalize the en
        abs_en = np.linalg.norm(en_endo, axis=1, keepdims=True)
        abs_en = np.where(abs_en != 0, abs_en, 1)
        en_endo = en_endo / abs_en
        
        abs_en = np.linalg.norm(en_epi, axis=1, keepdims=True)
        abs_en = np.where(abs_en != 0, abs_en, 1)
        en_epi = en_epi / abs_en
        print('############### en ###############')

        # el
        el_endo = np.cross(en_endo, et)
        el_epi = np.cross(en_epi, et)
        print('############### el ###############')

        abs_v_grad = np.linalg.norm(v_grad, axis=1, keepdims=True)
        abs_v_grad = np.where(abs_v_grad != 0, abs_v_grad, 1)
        v_grad_norm = v_grad / abs_v_grad
        
        ### Subendo PVs bundle selection
        
        el_endo[PVs["LIPV"]] = v_grad_norm[PVs["LIPV"]]
        el_endo[PVs["LSPV"]] = v_grad_norm[PVs["LSPV"]]
        
        # for i in range(len(v_grad_norm)):
        #     if v[i] <= 0.2:
        #         el_endo[i] = v_grad_norm[i]

        end_time = datetime.datetime.now()
        
        el_endo = np.where(el_endo == [0,0,0], [1,0,0], el_endo).astype("float32")
        
        sheet_endo = np.cross(el_endo, et)
        sheet_endo = np.where(sheet_endo == [0,0,0], [1,0,0], sheet_endo).astype("float32")
        
        for i in range(endo.GetPointData().GetNumberOfArrays()-1, -1, -1):
            endo.GetPointData().RemoveArray(endo.GetPointData().GetArrayName(i))
        
        for i in range(endo.GetCellData().GetNumberOfArrays()-1, -1, -1):
            endo.GetCellData().RemoveArray(endo.GetCellData().GetArrayName(i))
        
        meshNew = dsa.WrapDataObject(endo)
        meshNew.CellData.append(tag_endo, "elemTag")
        meshNew.CellData.append(el_endo, "fiber")
        meshNew.CellData.append(sheet_endo, "sheet")
        if args.ofmt == 'vtk':
            writer = vtk.vtkUnstructuredGridWriter()
            writer.SetFileName(job.ID+"/result_LA/LA_endo_with_fiber.vtk")
            writer.SetFileTypeToBinary()
        else:
            writer = vtk.vtkXMLUnstructuredGridWriter()
            writer.SetFileName(job.ID+"/result_LA/LA_endo_with_fiber.vtu")
        writer.SetInputData(meshNew.VTKObject)
        writer.Write()
        
        pts = numpy_support.vtk_to_numpy(endo.GetPoints().GetData())
        with open(job.ID+'/result_LA/LA_endo_with_fiber.pts',"w") as f:
            f.write("{}\n".format(len(pts)))
            for i in range(len(pts)):
                f.write("{} {} {}\n".format(pts[i][0], pts[i][1], pts[i][2]))

        with open(job.ID+'/result_LA/LA_endo_with_fiber.elem',"w") as f:
            f.write("{}\n".format(endo.GetNumberOfCells()))
            for i in range(endo.GetNumberOfCells()):
                cell = endo.GetCell(i)
                if cell.GetNumberOfPoints() == 2:
                    f.write("Ln {} {} {}\n".format(cell.GetPointIds().GetId(0), cell.GetPointIds().GetId(1), tag_endo[i]))
                elif cell.GetNumberOfPoints() == 3:
                    f.write("Tr {} {} {} {}\n".format(cell.GetPointIds().GetId(0), cell.GetPointIds().GetId(1), cell.GetPointIds().GetId(2), tag_endo[i]))
                elif cell.GetNumberOfPoints() == 4:
                    f.write("Tt {} {} {} {} {}\n".format(cell.GetPointIds().GetId(0), cell.GetPointIds().GetId(1), cell.GetPointIds().GetId(2), cell.GetPointIds().GetId(3), tag_endo[i]))
        
        
        with open(job.ID+'/result_LA/LA_endo_with_fiber.lon',"w") as f:
            f.write("2\n")
            for i in range(len(el_endo)):
                f.write("{} {} {} {} {} {}\n".format(el_endo[i][0], el_endo[i][1], el_endo[i][2], sheet_endo[i][0], sheet_endo[i][1], sheet_endo[i][2]))
            
        running_time = end_time - start_time
        
        print('Calculating fibers... done! ' + str(end_time) + '\nIt takes: ' + str(running_time) + '\n')

    else:
        ab_grad[band_cell_ids] = r_grad[band_cell_ids]
    
        # normalize the gradient phie
        phie_grad_norm = phie_grad

        ##### Local coordinate system #####
        # et
        et = phie_grad_norm
        print('############### et ###############')

        # k
        k = ab_grad
        print('############### k ###############')
        
        en = k - et*np.sum(k*et,axis=1).reshape(len(et),1)
        
        # normalize the en
        abs_en = np.linalg.norm(en, axis=1, keepdims=True)
        abs_en = np.where(abs_en != 0, abs_en, 1)
        en = en / abs_en
        
        print('############### en ###############')

        # el
        el = np.cross(en, et)
        print('############### el ###############')

        abs_v_grad = np.linalg.norm(v_grad, axis=1, keepdims=True)
        abs_v_grad = np.where(abs_v_grad != 0, abs_v_grad, 1)
        v_grad_norm = v_grad / abs_v_grad
        
        ### Subendo PVs bundle selection
        
        el[LIPV_ids_endo] = v_grad_norm[LIPV_ids_endo]
        el[LSPV_ids_endo] = v_grad_norm[LSPV_ids_endo]
        
        # for i in range(len(v_grad_norm)):
        #     if v[i] <= 0.2:
        #         el_endo[i] = v_grad_norm[i]

        end_time = datetime.datetime.now()
        
        meshNew = dsa.WrapDataObject(model)
        meshNew.CellData.append(tag, "elemTag")
        meshNew.CellData.append(el, "fiber")
        model = meshNew.VTKObject

        # meshNew = dsa.WrapDataObject(endo)
        # meshNew.CellData.append(tag_endo, "elemTag")
        # meshNew.CellData.append(el_endo, "fiber")
        # meshNew.CellData.append(sheet_endo, "sheet")
        # if args.ofmt == 'vtk':
        #     writer = vtk.vtkUnstructuredGridWriter()
        #     writer.SetFileName(job.ID+"/result_LA/LA_endo_with_fiber.vtk")
        #     writer.SetFileTypeToBinary()
        # else:
        #     writer = vtk.vtkXMLUnstructuredGridWriter()
        #     writer.SetFileName(job.ID+"/result_LA/LA_endo_with_fiber.vtu")
        # writer.SetInputData(meshNew.VTKObject)
        # writer.Write()
        
        # pts = numpy_support.vtk_to_numpy(endo.GetPoints().GetData())
        # with open(job.ID+'/result_LA/LA_endo_with_fiber.pts',"w") as f:
        #     f.write("{}\n".format(len(pts)))
        #     for i in range(len(pts)):
        #         f.write("{} {} {}\n".format(pts[i][0], pts[i][1], pts[i][2]))

        # with open(job.ID+'/result_LA/LA_endo_with_fiber.elem',"w") as f:
        #     f.write("{}\n".format(endo.GetNumberOfCells()))
        #     for i in range(endo.GetNumberOfCells()):
        #         cell = endo.GetCell(i)
        #         if cell.GetNumberOfPoints() == 2:
        #             f.write("Ln {} {} {}\n".format(cell.GetPointIds().GetId(0), cell.GetPointIds().GetId(1), tag_endo[i]))
        #         elif cell.GetNumberOfPoints() == 3:
        #             f.write("Tr {} {} {} {}\n".format(cell.GetPointIds().GetId(0), cell.GetPointIds().GetId(1), cell.GetPointIds().GetId(2), tag_endo[i]))
        #         elif cell.GetNumberOfPoints() == 4:
        #             f.write("Tt {} {} {} {} {}\n".format(cell.GetPointIds().GetId(0), cell.GetPointIds().GetId(1), cell.GetPointIds().GetId(2), cell.GetPointIds().GetId(3), tag_endo[i]))
        
        
        # with open(job.ID+'/result_LA/LA_endo_with_fiber.lon',"w") as f:
        #     f.write("2\n")
        #     for i in range(len(el_endo)):
        #         f.write("{} {} {} {} {} {}\n".format(el_endo[i][0], el_endo[i][1], el_endo[i][2], sheet_endo[i][0], sheet_endo[i][1], sheet_endo[i][2]))
            
        running_time = end_time - start_time
        
        print('Calculating fibers... done! ' + str(end_time) + '\nIt takes: ' + str(running_time) + '\n')

    print("Creating bachmann bundles...")
    # Bachmann Bundle
    
    if args.mesh_type == "vol": # Extract epicardial surface
        
        # epi_surf = Method.smart_reader(args.mesh[:-4]+'_epi.obj')
        # epi_surf = Method.cell_array_mapper(model, epi_surf, "elemTag", "all")
        geo_filter = vtk.vtkGeometryFilter()
        geo_filter.SetInputData(model)
        geo_filter.Update()
        surf = geo_filter.GetOutput()
        epi_surf = Method.vtk_thr(surf,0,"CELLS","phie_phi",0.5)
        epi_surf_ids = vtk.util.numpy_support.vtk_to_numpy(epi_surf.GetCellData().GetArray('Global_ids'))

        # epi_surf_id_list = vtk.vtkIdList()
        # for i in epi_surf_ids:
        #     epi_surf_id_list.InsertNextId(i)
        # extract = vtk.vtkExtractCells()
        # extract.SetInputData(surf)
        # extract.SetCellList(epi_surf_id_list)
        # extract.Update()
        geo_filter = vtk.vtkGeometryFilter()
        geo_filter.SetInputData(epi_surf)
        #geo_filter.PassThroughCellIdsOn()
        geo_filter.Update()

        epi_surf = geo_filter.GetOutput()

    if args.mesh_type == "bilayer":
        bb_left, LAA_basis_inf, LAA_basis_sup, LAA_far_from_LIPV = Method.compute_wide_BB_path_left(epi, df, left_atrial_appendage_epi, mitral_valve_epi)
        tag_epi = Method.assign_element_tag_around_path_within_radius(epi, bb_left, w_bb, tag_epi, bachmann_bundel_left)
        el_epi = Method.assign_element_fiber_around_path_within_radius(epi, bb_left, w_bb, el_epi, smooth=True)
    else:
        bb_left, LAA_basis_inf, LAA_basis_sup, LAA_far_from_LIPV = Method.compute_wide_BB_path_left(epi_surf, df, left_atrial_appendage_epi, mitral_valve_epi)
        tag[epi_surf_ids] = Method.assign_element_tag_around_path_within_radius(epi_surf, bb_left, w_bb, vtk.util.numpy_support.vtk_to_numpy(epi_surf.GetCellData().GetArray('elemTag')), bachmann_bundel_left)
        el[epi_surf_ids] = Method.assign_element_fiber_around_path_within_radius(epi_surf, bb_left, w_bb, vtk.util.numpy_support.vtk_to_numpy(epi_surf.GetCellData().GetArray('fiber')), smooth=True)
    
    df["LAA_basis_inf"] = LAA_basis_inf
    df["LAA_basis_sup"] = LAA_basis_sup
    df["LAA_far_from_LIPV"] = LAA_far_from_LIPV
    
    df.to_csv(args.mesh+"_surf/rings_centroids.csv", float_format="%.2f", index = False)
    
    # # Bachmann_Bundle internal connection
    # la_connect_point, ra_connect_point = Method.get_connection_point_la_and_ra(la_appex_point)
    # la_connect_point = np.asarray(la_connect_point)
    #
    # path_start_end = np.vstack((appendage_basis, la_connect_point))
    # path_bb_ra = Method.creat_center_line(path_start_end)
    # tag = Method.assign_element_tag_around_path_within_radius(model, path_bb_ra, 2, tag, bachmann_bundel_left)
    # el = Method.assign_element_fiber_around_path_within_radius(model, path_bb_ra, 2, el, smooth=True)

    print("Creating bachmann bundles... done")
    if args.mesh_type == "bilayer":
        el_epi = np.where(el_epi == [0,0,0], [1,0,0], el_epi).astype("float32")
        #### save the result into vtk ####
        start_time = datetime.datetime.now()
        print('Writinga as LA_with_fiber... ' + str(start_time))
        
        sheet_epi = np.cross(el_epi, et)
        sheet_epi = np.where(sheet_epi == [0,0,0], [1,0,0], sheet_epi).astype("float32")
        
        for i in range(epi.GetPointData().GetNumberOfArrays()-1, -1, -1):
            epi.GetPointData().RemoveArray(epi.GetPointData().GetArrayName(i))
        
        for i in range(epi.GetCellData().GetNumberOfArrays()-1, -1, -1):
            epi.GetCellData().RemoveArray(epi.GetCellData().GetArrayName(i))

        meshNew = dsa.WrapDataObject(epi)
        meshNew.CellData.append(tag_epi, "elemTag")
        meshNew.CellData.append(el_epi, "fiber")
        meshNew.CellData.append(sheet_epi, "sheet")
        if args.ofmt == 'vtk':
            writer = vtk.vtkUnstructuredGridWriter()
            writer.SetFileName(job.ID+"/result_LA/LA_epi_with_fiber.vtk")
            writer.SetFileTypeToBinary()
        else:
            writer = vtk.vtkXMLUnstructuredGridWriter()
            writer.SetFileName(job.ID+"/result_LA/LA_epi_with_fiber.vtu")
        writer.SetInputData(meshNew.VTKObject)
        writer.Write()
        
        pts = numpy_support.vtk_to_numpy(epi.GetPoints().GetData())
        with open(job.ID+'/result_LA/LA_epi_with_fiber.pts',"w") as f:
            f.write("{}\n".format(len(pts)))
            for i in range(len(pts)):
                f.write("{} {} {}\n".format(pts[i][0], pts[i][1], pts[i][2]))

        with open(job.ID+'/result_LA/LA_epi_with_fiber.elem',"w") as f:
            f.write("{}\n".format(epi.GetNumberOfCells()))
            for i in range(epi.GetNumberOfCells()):
                cell = epi.GetCell(i)
                if cell.GetNumberOfPoints() == 2:
                    f.write("Ln {} {} {}\n".format(cell.GetPointIds().GetId(0), cell.GetPointIds().GetId(1), tag_epi[i]))
                elif cell.GetNumberOfPoints() == 3:
                    f.write("Tr {} {} {} {}\n".format(cell.GetPointIds().GetId(0), cell.GetPointIds().GetId(1), cell.GetPointIds().GetId(2), tag_epi[i]))
                elif cell.GetNumberOfPoints() == 4:
                    f.write("Tt {} {} {} {} {}\n".format(cell.GetPointIds().GetId(0), cell.GetPointIds().GetId(1), cell.GetPointIds().GetId(2), cell.GetPointIds().GetId(3), tag_epi[i]))
        
        with open(job.ID+'/result_LA/LA_epi_with_fiber.lon',"w") as f:
            f.write("2\n")
            for i in range(len(el_epi)):
                f.write("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(el_epi[i][0], el_epi[i][1], el_epi[i][2], sheet_epi[i][0], sheet_epi[i][1], sheet_epi[i][2]))
                
        end_time = datetime.datetime.now()
        running_time = end_time - start_time
        print('Writing as LA_with_fiber... done! ' + str(end_time) + '\nIt takes: ' + str(running_time) + '\n')
        
        endo = Method.move_surf_along_normals(endo, 0.1*args.scale, 1) #  Warning: set -1 if pts normals are pointing outside
        bilayer = Method.generate_bilayer(endo, epi)
        
        Method.write_bilayer(bilayer, args, job)

    else:
        el = np.where(el == [0,0,0], [1,0,0], el).astype("float32")

        #### save the result into vtk ####
        start_time = datetime.datetime.now()
        print('Writinga as LA_with_fiber... ' + str(start_time))
        
        sheet = np.cross(el, et)
        sheet = np.where(sheet == [0,0,0], [1,0,0], sheet).astype("float32")

        for i in range(model.GetPointData().GetNumberOfArrays()-1, -1, -1):
            model.GetPointData().RemoveArray(model.GetPointData().GetArrayName(i))
        
        for i in range(model.GetCellData().GetNumberOfArrays()-1, -1, -1):
            model.GetCellData().RemoveArray(model.GetCellData().GetArrayName(i))

        meshNew = dsa.WrapDataObject(model)
        meshNew.CellData.append(tag, "elemTag")
        meshNew.CellData.append(el, "fiber")
        meshNew.CellData.append(sheet, "sheet")
        if args.ofmt == 'vtk':
            writer = vtk.vtkUnstructuredGridWriter()
            writer.SetFileName(job.ID+"/result_LA/LA_vol_with_fiber.vtk")
            writer.SetFileTypeToBinary()
        else:
            writer = vtk.vtkXMLUnstructuredGridWriter()
            writer.SetFileName(job.ID+"/result_LA/LA_vol_with_fiber.vtu")
        writer.SetInputData(meshNew.VTKObject)
        writer.Write()
        
        pts = numpy_support.vtk_to_numpy(model.GetPoints().GetData())
        with open(job.ID+'/result_LA/LA_vol_with_fiber.pts',"w") as f:
            f.write("{}\n".format(len(pts)))
            for i in range(len(pts)):
                f.write("{} {} {}\n".format(pts[i][0], pts[i][1], pts[i][2]))

        with open(job.ID+'/result_LA/LA_vol_with_fiber.elem',"w") as f:
            f.write("{}\n".format(model.GetNumberOfCells()))
            for i in range(model.GetNumberOfCells()):
                cell = model.GetCell(i)
                if cell.GetNumberOfPoints() == 2:
                    f.write("Ln {} {} {}\n".format(cell.GetPointIds().GetId(0), cell.GetPointIds().GetId(1), tag[i]))
                elif cell.GetNumberOfPoints() == 3:
                    f.write("Tr {} {} {} {}\n".format(cell.GetPointIds().GetId(0), cell.GetPointIds().GetId(1), cell.GetPointIds().GetId(2), tag[i]))
                elif cell.GetNumberOfPoints() == 4:
                    f.write("Tt {} {} {} {} {}\n".format(cell.GetPointIds().GetId(0), cell.GetPointIds().GetId(1), cell.GetPointIds().GetId(2), cell.GetPointIds().GetId(3), tag[i]))
        
        with open(job.ID+'/result_LA/LA_vol_with_fiber.lon',"w") as f:
            f.write("2\n")
            for i in range(len(el)):
                f.write("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(el[i][0], el[i][1], el[i][2], sheet[i][0], sheet[i][1], sheet[i][2]))
                
        end_time = datetime.datetime.now()
        running_time = end_time - start_time
        print('Writing as LA_with_fiber... done! ' + str(end_time) + '\nIt takes: ' + str(running_time) + '\n')
