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
import pandas as pd
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.util.numpy_support import vtk_to_numpy
import datetime
import Methods_RA as Method
import csv
import pickle
import os
from scipy.spatial import cKDTree

EXAMPLE_DIR = os.path.dirname(os.path.realpath(__file__))

from create_bridges import add_free_bridge

vtk_version = vtk.vtkVersion.GetVTKSourceVersion().split()[-1].split('.')[0]

def ra_generate_fiber(model, args, job):
    
    simid = job.ID+"/result_RA"
    try:
        os.makedirs(simid)
    except OSError:
        print ("Creation of the directory %s failed" % simid)
    else:
        print ("Successfully created the directory %s " % simid)
        
    simid = job.ID+"/bridges"
    try:
        os.makedirs(simid)
    except OSError:
        print ("Creation of the directory %s failed" % simid)
    else:
        print ("Successfully created the directory %s " % simid)
        
    # Riunet
    tao_tv = 0.9
    tao_icv = 0.95 #0.05 # dk01 biatrial fit_both
    tao_scv = 0.10
    tao_ct_plus = -0.10
    tao_ct_minus = -0.13
    tao_ib = -0.06
    tao_ras = 0.13
    tao_raw = 0.55
    with open(os.path.join(EXAMPLE_DIR,'../../element_tag.csv')) as f:
        tag_dict = {}
        reader = csv.DictReader(f)
        for row in reader:
            tag_dict[row['name']] = row['tag']
    # load epi tags
    tricuspid_valve_epi = int(tag_dict['tricuspid_valve_epi'])
    superior_vena_cava_epi = int(tag_dict['superior_vena_cava_epi'])
    inferior_vena_cava_epi = int(tag_dict['inferior_vena_cava_epi'])
    crista_terminalis = int(tag_dict['crista_terminalis'])
    inter_caval_bundle_epi = int(tag_dict['inter_caval_bundle_epi'])
    right_atrial_lateral_wall_epi = int(tag_dict['right_atrial_wall_epi'])
    isthmus_epi = int(tag_dict['isthmus_epi'])
    right_atrial_septum_epi = int(tag_dict['right_atrial_septum_epi'])
    pectinate_muscle = int(tag_dict['pectinate_muscle'])
    right_atrial_appendage_epi = int(tag_dict['right_atrial_appendage_epi'])

    # load endo tags
    tricuspid_valve_endo = int(tag_dict['tricuspid_valve_endo'])
    superior_vena_cava_endo = int(tag_dict['superior_vena_cava_endo'])
    inferior_vena_cava_endo = int(tag_dict['inferior_vena_cava_endo'])
    inter_caval_bundle_endo = int(tag_dict['inter_caval_bundle_endo'])
    right_atrial_lateral_wall_endo = int(tag_dict['right_atrial_wall_endo'])
    isthmus_endo = int(tag_dict['isthmus_endo'])
    right_atrial_septum_endo = int(tag_dict['right_atrial_septum_endo'])
    coronary_sinus = int(tag_dict['coronary_sinus'])

    # load bridges tag
    bachmann_bundel_right = int(tag_dict['bachmann_bundel_right'])
    bachmann_bundel_internal = int(tag_dict['bachmann_bundel_internal'])
    
    # load left atrial wall epi
    left_atrial_wall_epi = int(tag_dict['left_atrial_wall_epi'])

    # load sinus node
    sinus_node = int(tag_dict['sinus_node'])

    # number of pectinate muscles
    pm_num = 15

    # size(Radius) of crista terminalis in mm
    w_ct = 4.62*args.scale

    # size(Radius) of pectinate muscle in mm
    w_pm = 0.66*args.scale

    # size(Radius) of Bachmann Bundle in mm
    w_bb = 2*args.scale
    
    # radius sinus node
    r_SN = 2.5*args.scale

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

    # w
    w = model.GetCellData().GetArray('phie_w')
    w_grad = model.GetCellData().GetArray('grad_w')
    w = vtk.util.numpy_support.vtk_to_numpy(w)
    w_grad = vtk.util.numpy_support.vtk_to_numpy(w_grad)

    # phie
    if args.mesh_type == "vol":
        phie = model.GetCellData().GetArray('phie_phi')
        phie = vtk.util.numpy_support.vtk_to_numpy(phie)
    phie_grad = model.GetCellData().GetArray('grad_phi')
    phie_grad = vtk.util.numpy_support.vtk_to_numpy(phie_grad)

    start_time = datetime.datetime.now()
    print('Calculating fibers... ' + str(start_time))
    
    cellid = vtk.vtkIdFilter()
    cellid.CellIdsOn()
    cellid.SetInputData(model) # vtkPolyData()
    cellid.PointIdsOn()
    if int(vtk_version) >= 9:
        cellid.SetPointIdsArrayName('Global_ids')
        cellid.SetCellIdsArrayName('Global_ids')
    else:
        cellid.SetIdsArrayName('Global_ids')
    cellid.Update()
    
    model = cellid.GetOutput()
    
    # TV
    
    tag = np.zeros((len(ab),), dtype = int)
    
    k = np.copy(ab_grad)
    
    # # # Get valve using Laplacian solutions

    # TV_s = Method.vtk_thr(model,0,"CELLS","phie_r",tao_tv) # grad_r
    
    # TV_ids = vtk.util.numpy_support.vtk_to_numpy(TV_s.GetCellData().GetArray('Global_ids'))

    # no_TV_s = Method.vtk_thr(model, 1,"CELLS","phie_r",tao_tv)
    
    # Use fixed thickness

    ring_ids = np.loadtxt('{}_surf/'.format(args.mesh) + 'ids_TV.vtx', skiprows=2, dtype=int)

    rings_pts = vtk.util.numpy_support.vtk_to_numpy(model.GetPoints().GetData())[ring_ids,:]

    if args.debug:
        Method.create_pts(rings_pts,'TV_ring','{}_surf/'.format(args.mesh))

    TV_ids = Method.get_element_ids_around_path_within_radius(model, rings_pts, 4*args.scale)

    ra_TV = vtk.vtkIdList()
    for var in TV_ids:
        ra_TV.InsertNextId(var)

    extract = vtk.vtkExtractCells()
    extract.SetInputData(model)
    extract.SetCellList(ra_TV)
    extract.Update()

    TV_s = extract.GetOutput()

    ra_diff = list(set(list(vtk.util.numpy_support.vtk_to_numpy(model.GetCellData().GetArray('Global_ids')))).difference(set(TV_ids)))
    ra_no_TV = vtk.vtkIdList()
    for var in ra_diff:
        ra_no_TV.InsertNextId(var)

    extract = vtk.vtkExtractCells()
    extract.SetInputData(model)
    extract.SetCellList(ra_no_TV)
    extract.Update()

    no_TV_s = extract.GetOutput()

    # To check if TV was correctly identified
    if args.debug:
        Method.writer_vtk(TV_s, '{}_surf/'.format(args.mesh) + "tv_s.vtk")
        Method.writer_vtk(no_TV_s, '{}_surf/'.format(args.mesh) + "no_tv_s.vtk")

    # del ra_TV, ra_diff, ra_no_TV
    
    tag[TV_ids] = tricuspid_valve_epi
    
    k[TV_ids] = r_grad[TV_ids]


    IVC_s = Method.vtk_thr(no_TV_s, 0,"CELLS","phie_v",tao_icv) # Changed 0-1 because ICV and SVC are inverted
    no_IVC_s = Method.vtk_thr(no_TV_s, 1,"CELLS","phie_v",tao_icv) #Changed 1-0

    IVC_s = Method.extract_largest_region(IVC_s) # Added


    max_phie_r_ivc = np.max(vtk.util.numpy_support.vtk_to_numpy(IVC_s.GetCellData().GetArray('phie_r')))

    RAW_s = Method.vtk_thr(no_TV_s, 1,"CELLS","phie_r", max_phie_r_ivc) # Added +0.03 fro dk01

    SVC_s = Method.vtk_thr(RAW_s, 1,"CELLS","phie_v",tao_scv) # Changed 1->0
    no_SVC_s = Method.vtk_thr(RAW_s, 0,"CELLS","phie_v",tao_scv) #Changed 0->1
    
    SVC_s = Method.extract_largest_region(SVC_s)

    if args.debug:  # CHECK
        Method.writer_vtk(IVC_s, '{}_surf/'.format(args.mesh) + "ivc_s.vtk")
        Method.writer_vtk(no_IVC_s, '{}_surf/'.format(args.mesh) + "no_ivc_s.vtk")
        Method.writer_vtk(SVC_s, '{}_surf/'.format(args.mesh) + "svc_s.vtk")
        Method.writer_vtk(no_SVC_s, '{}_surf/'.format(args.mesh) + "no_svc_s.vtk")
    
    tao_ct_plus = np.min(vtk.util.numpy_support.vtk_to_numpy(SVC_s.GetCellData().GetArray('phie_w')))
    
    SVC_CT_pt = SVC_s.GetPoint(np.argmin(vtk.util.numpy_support.vtk_to_numpy(SVC_s.GetPointData().GetArray('phie_w'))))
    
    tao_ct_minus = np.min(vtk.util.numpy_support.vtk_to_numpy(IVC_s.GetCellData().GetArray('phie_w')))
    
    IVC_CT_pt = IVC_s.GetPoint(np.argmin(vtk.util.numpy_support.vtk_to_numpy(IVC_s.GetPointData().GetArray('phie_w'))))
    
    IVC_SEPT_CT_pt = IVC_s.GetPoint(np.argmax(vtk.util.numpy_support.vtk_to_numpy(IVC_s.GetPointData().GetArray('phie_w'))))
    
    IVC_max_r_CT_pt = IVC_s.GetPoint(np.argmax(vtk.util.numpy_support.vtk_to_numpy(IVC_s.GetPointData().GetArray('phie_r')))) # not always the best choice for pm1

    #thr_min = 0.150038
    #thr_max = 0.38518

    #CT_band = Method.vtk_thr(RAW_s, 2,"CELLS","phie_w", thr_min, thr_max) # dk01 fit_both
    CT_band = Method.vtk_thr(RAW_s, 2,"CELLS","phie_w", tao_ct_minus-0.01, tao_ct_plus) # grad_w

    CT_ub = Method.vtk_thr(RAW_s, 2,"CELLS","phie_w", tao_ct_plus-0.02, tao_ct_plus) # grad_w

    CT_ub = Method.extract_largest_region(CT_ub)

    if args.debug:
        Method.writer_vtk(RAW_s, '{}_surf/'.format(args.mesh) + "raw_s.vtk")
        Method.writer_vtk(CT_band, '{}_surf/'.format(args.mesh) + "ct_band.vtk")
        Method.writer_vtk(CT_ub, '{}_surf/'.format(args.mesh) + "ct_ub.vtk")
    
    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(CT_ub)
    geo_filter.Update()
    mesh_surf = geo_filter.GetOutput()

    loc = vtk.vtkPointLocator()
    loc.SetDataSet(mesh_surf)
    loc.BuildLocator()

    IVC_CT_pt_id = loc.FindClosestPoint(np.array(IVC_CT_pt))

    SVC_CT_pt_id = loc.FindClosestPoint(np.array(SVC_CT_pt))

    CT_ub_pts = Method.dijkstra_path(mesh_surf, IVC_CT_pt_id, SVC_CT_pt_id)
    
    filter_cell_centers = vtk.vtkCellCenters()
    filter_cell_centers.SetInputData(CT_band)
    filter_cell_centers.Update()
    centroids = filter_cell_centers.GetOutput().GetPoints()
    centroids_array = vtk.util.numpy_support.vtk_to_numpy(centroids.GetData())

    tree = cKDTree(centroids_array)

    ii = tree.query_ball_point(CT_ub_pts, r = 7*args.scale, n_jobs=-1)
    
    ii = set([item for sublist in ii for item in sublist])

    cell_ids = vtk.vtkIdList()
    for i in ii:
        cell_ids.InsertNextId(i)
    extract = vtk.vtkExtractCells()
    extract.SetInputData(CT_band)
    extract.SetCellList(cell_ids)
    extract.Update()
    
    CT_band = extract.GetOutput()

    #IVC_max_r_CT_pt = CT_band.GetPoint(np.argmax(vtk.util.numpy_support.vtk_to_numpy(CT_band.GetPointData().GetArray('phie_r'))))  # optional choice for pm, be careful as it overwrites

    if args.debug:
        Method.writer_vtk(CT_band, '{}_surf/'.format(args.mesh) + "ct_band_2.vtk")
    
    CT_band_ids = vtk.util.numpy_support.vtk_to_numpy(CT_band.GetCellData().GetArray('Global_ids'))

    tao_RAA = np.max(vtk.util.numpy_support.vtk_to_numpy(CT_band.GetCellData().GetArray('phie_v2')))
    
    # CT_ids = vtk.util.numpy_support.vtk_to_numpy(CT_band.GetCellData().GetArray('Global_ids'))
    
    # tag[CT_ids] = crista_terminalis

    # CT part from IVC to septum
    
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(CT_band)
    loc.BuildLocator()

    IVC_CT_pt_id = loc.FindClosestPoint(np.array(IVC_CT_pt))
    
    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(no_IVC_s)
    geo_filter.Update()
    no_IVC_s = geo_filter.GetOutput()
    
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(no_IVC_s)
    loc.BuildLocator()

    IVC_CT_pt_id = loc.FindClosestPoint(np.array(CT_band.GetPoint(IVC_CT_pt_id)))
    IVC_max_r_CT_pt_id = loc.FindClosestPoint(np.array(IVC_max_r_CT_pt))
    IVC_SEPT_CT_pt_id = loc.FindClosestPoint(np.array(IVC_SEPT_CT_pt))
    
    CT_SEPT_path = np.concatenate((Method.dijkstra_path(no_IVC_s, IVC_CT_pt_id, IVC_max_r_CT_pt_id), Method.dijkstra_path(no_IVC_s, IVC_max_r_CT_pt_id, IVC_SEPT_CT_pt_id)), axis=0)
    
    CT_SEPT_ids = Method.get_element_ids_around_path_within_radius(no_IVC_s, CT_SEPT_path, w_ct)

    # SVC_CT_pt_id = loc.FindClosestPoint(SVC_CT_pt)
    
    CT_minus = Method.vtk_thr(RAW_s, 1,"CELLS","phie_w", tao_ct_plus) # grad_ab
    
    RAW_I_ids = vtk.util.numpy_support.vtk_to_numpy(CT_minus.GetCellData().GetArray('Global_ids'))
    
    ii = set(RAW_I_ids) - set(CT_SEPT_ids) - set(CT_band_ids)
    cell_ids = vtk.vtkIdList()
    for i in ii:
        cell_ids.InsertNextId(i)
    extract = vtk.vtkExtractCells()
    extract.SetInputData(CT_minus)
    extract.SetCellList(cell_ids)
    extract.Update()
    
    CT_minus = extract.GetOutput()
    
    RAW_I_ids = vtk.util.numpy_support.vtk_to_numpy(CT_minus.GetCellData().GetArray('Global_ids'))
    
    tag[RAW_I_ids] = right_atrial_lateral_wall_epi
    
    k[TV_ids] = r_grad[TV_ids]
    
    CT_plus = Method.vtk_thr(RAW_s, 0,"CELLS","phie_w", tao_ct_plus)
    
    RAW_S = Method.vtk_thr(CT_plus, 2,"CELLS","phie_v",  tao_scv, tao_icv) # IB_S grad_v Changed order tao_scv, tao_icv
    
    RAW_S_ids = vtk.util.numpy_support.vtk_to_numpy(RAW_S.GetCellData().GetArray('Global_ids'))
    
    tag[RAW_S_ids] = right_atrial_lateral_wall_epi
    
    k[RAW_S_ids] = ab_grad[RAW_S_ids]
    
    IB = Method.vtk_thr(RAW_S, 1,"CELLS","phie_r", 0.05)  # grad_r or w
    
    IB_ids = vtk.util.numpy_support.vtk_to_numpy(IB.GetCellData().GetArray('Global_ids'))
    
    tag[IB_ids] = inter_caval_bundle_epi

    if args.debug:
        Method.writer_vtk(IB, '{}_surf/'.format(args.mesh) + "ib.vtk")
        Method.writer_vtk(RAW_S, '{}_surf/'.format(args.mesh) + "raw_s.vtk")
        Method.writer_vtk(CT_plus, '{}_surf/'.format(args.mesh) + "ct_plus.vtk")

    
    k[IB_ids] = v_grad[IB_ids]
    
    df = pd.read_csv(args.mesh+"_surf/rings_centroids.csv")
    
    # calculate the norm vector
    v1 = np.array(df["IVC"]) - np.array(df["SVC"])
    v2 = np.array(df["TV"]) - np.array(df["IVC"])
    norm = np.cross(v1, v2)
    
    #normalize norm
    n = np.linalg.norm(norm)
    norm_1 = norm/n # Changed sign

    plane = vtk.vtkPlane()
    plane.SetNormal(norm_1[0], norm_1[1], norm_1[2]) # changed
    plane.SetOrigin(df["TV"][0], df["TV"][1], df["TV"][2])
    
    meshExtractFilter = vtk.vtkExtractGeometry()
    meshExtractFilter.SetInputData(Method.to_polydata(RAW_S))
    meshExtractFilter.SetImplicitFunction(plane)
    meshExtractFilter.Update()
    septal_surf = meshExtractFilter.GetOutput()


    RAS_S = Method.vtk_thr(septal_surf, 0,"CELLS","phie_w", tao_ct_plus)
    RAS_S = Method.vtk_thr(RAS_S, 0,"CELLS","phie_r", 0.05)  # grad_r or w

    if args.debug:
        Method.writer_vtk(septal_surf, '{}_surf/'.format(args.mesh) + "septal_surf.vtk")
        Method.writer_vtk(RAS_S, '{}_surf/'.format(args.mesh) + "ras_s.vtk")
    
    RAS_S_ids = vtk.util.numpy_support.vtk_to_numpy(RAS_S.GetCellData().GetArray('Global_ids'))
    
    tag[RAS_S_ids] = right_atrial_septum_epi
    
    k[RAS_S_ids] = r_grad[RAS_S_ids]
    
    RAW_low = Method.vtk_thr(no_TV_s, 0,"CELLS","phie_r", max_phie_r_ivc)
    
    meshExtractFilter = vtk.vtkExtractGeometry()
    meshExtractFilter.SetInputData(RAW_low)
    meshExtractFilter.SetImplicitFunction(plane)
    meshExtractFilter.Update()
    RAS_low = meshExtractFilter.GetOutput()
    
    RAS_low = Method.vtk_thr(RAS_low, 0,"CELLS","phie_w", 0) # grad_r overwrites the previous
    
    RAS_low_ids = vtk.util.numpy_support.vtk_to_numpy(RAS_low.GetCellData().GetArray('Global_ids'))
    
    tag[RAS_low_ids] = right_atrial_septum_epi
    
    k[RAS_low_ids] = r_grad[RAS_low_ids]
    
    RAW_low = Method.vtk_thr(RAW_low, 1,"CELLS","phie_w", 0) # grad_ab
    
    RAW_low_ids = vtk.util.numpy_support.vtk_to_numpy(RAW_low.GetCellData().GetArray('Global_ids'))
    
    tag[RAW_low_ids] = right_atrial_lateral_wall_epi
    
    k[RAW_low_ids] = ab_grad[RAW_low_ids]
    
    # calculate the norm vector
    #v1 = np.array(IVC_SEPT_CT_pt) - np.array(IVC_CT_pt)
    #v2 = np.array(df["TV"]) - np.array(df["IVC"])
    #norm = np.cross(v1, v2)
    norm = np.array(df["SVC"]) - np.array(df["IVC"])
    #normalize norm
    n = np.linalg.norm(norm)
    norm_1 = norm/n

    plane = vtk.vtkPlane()
    plane.SetNormal(norm_1[0], norm_1[1], norm_1[2])
    plane.SetOrigin(IVC_SEPT_CT_pt[0], IVC_SEPT_CT_pt[1], IVC_SEPT_CT_pt[2])
    
    meshExtractFilter = vtk.vtkExtractGeometry()
    meshExtractFilter.SetInputData(no_TV_s)
    meshExtractFilter.SetImplicitFunction(plane)
    meshExtractFilter.Update()
    septal_surf = meshExtractFilter.GetOutput()

    if args.debug:
        Method.writer_vtk(septal_surf, '{}_surf/'.format(args.mesh) + "septal_surf_2.vtk")
    
    CS_ids = vtk.util.numpy_support.vtk_to_numpy(septal_surf.GetCellData().GetArray('Global_ids'))

    #if len(CS_ids) == 0:
    ring_ids = np.loadtxt('{}_surf/'.format(args.mesh) + 'ids_CS.vtx', skiprows=2, dtype=int)
    
    rings_pts = vtk.util.numpy_support.vtk_to_numpy(model.GetPoints().GetData())[ring_ids,:]
    
    CS_ids = Method.get_element_ids_around_path_within_radius(model, rings_pts, 4*args.scale)
    
    tag[CS_ids] = coronary_sinus
    k[CS_ids] = ab_grad[CS_ids]
    
    #tag = Method.assign_ra_appendage(model, SVC_s, np.array(df["RAA"]), tag, right_atrial_appendage_epi)
    RAA_s = Method.vtk_thr(no_TV_s, 0,"CELLS","phie_v2", tao_RAA)

    if args.debug:
        Method.writer_vtk(RAS_low, '{}_surf/'.format(args.mesh) + "ras_low.vtk")
        Method.writer_vtk(RAA_s, '{}_surf/'.format(args.mesh) + "raa_s.vtk") # Check here if RAA is correctly tagged
        Method.writer_vtk(RAW_low, '{}_surf/'.format(args.mesh) + "raw_low.vtk")

    
    RAA_ids = vtk.util.numpy_support.vtk_to_numpy(RAA_s.GetCellData().GetArray('Global_ids'))
    
    tag[RAA_ids] = right_atrial_appendage_epi
    
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(CT_band)
    loc.BuildLocator()

    RAA_CT_pt = CT_band.GetPoint(loc.FindClosestPoint(np.array(df["RAA"])))

    # # calculate the norm vector
    # v1 = np.array(SVC_CT_pt) - np.array(RAA_CT_pt)
    # v2 = np.array(SVC_CT_pt) - np.array(df["TV"])
    # norm = np.cross(v1, v2)
    
    # #normalize norm
    # n = np.linalg.norm(norm)
    # norm_1 = norm/n

    # plane = vtk.vtkPlane()
    # plane.SetNormal(norm_1[0], norm_1[1], norm_1[2])
    # plane.SetOrigin(SVC_CT_pt[0], SVC_CT_pt[1], SVC_CT_pt[2])
    
    # meshExtractFilter = vtk.vtkExtractGeometry()
    # meshExtractFilter.SetInputData(CT_band)
    # meshExtractFilter.SetImplicitFunction(plane)
    # meshExtractFilter.Update()
    # CT = meshExtractFilter.GetOutput()
    
    # CT = Method.extract_largest_region(CT)
    CT = CT_band
    
    CT_ids = vtk.util.numpy_support.vtk_to_numpy(CT.GetCellData().GetArray('Global_ids'))
    
    CT_ids = np.setdiff1d(CT_ids, RAA_ids, assume_unique = True)
    
    tag[CT_ids] = crista_terminalis
    
    k[CT_ids] = w_grad[CT_ids]
    
    tag[CT_SEPT_ids] = crista_terminalis
    
    SVC_ids = vtk.util.numpy_support.vtk_to_numpy(SVC_s.GetCellData().GetArray('Global_ids'))
    
    tag[SVC_ids] = superior_vena_cava_epi
    
    k[SVC_ids] = v_grad[SVC_ids]
    
    IVC_ids = vtk.util.numpy_support.vtk_to_numpy(IVC_s.GetCellData().GetArray('Global_ids'))
    
    tag[IVC_ids] = inferior_vena_cava_epi
    
    k[IVC_ids] = v_grad[IVC_ids]
    
    tag = np.where(tag == 0, right_atrial_lateral_wall_epi, tag)
    
    SN_ids = Method.get_element_ids_around_path_within_radius(no_SVC_s, np.asarray([SVC_CT_pt]), r_SN)
    
    tag[SN_ids] = sinus_node
    
    # meshNew = dsa.WrapDataObject(model)
    # meshNew.CellData.append(tag, "elemTag")
    # writer = vtk.vtkUnstructuredGridWriter()
    # writer.SetFileName(job.ID+"/result_RA/RA_epi_with_fiber.vtk")
    # writer.SetInputData(meshNew.VTKObject)
    # writer.Write()
    
    # print('Region growing...to get the tao_ct_minus...')
    # # extract septum
    # thresh = vtk.vtkThreshold()
    # thresh.SetInputData(model)
    # thresh.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_CELLS", "phie_r")
    # thresh.ThresholdByLower(0.6)
    # thresh.Update()
    # septum = thresh.GetOutput()
    # points_data = septum.GetPoints().GetData()
    # septum_points = vtk.util.numpy_support.vtk_to_numpy(points_data)

    # # extract ICV form septum
    # thresh = vtk.vtkThreshold()
    # thresh.SetInputData(septum)
    # thresh.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_CELLS", "phie_v")
    # thresh.ThresholdByUpper(tao_icv)
    # thresh.Update()
    # ICV = thresh.GetOutput()
    # ICV = Method.extract_largest_region(ICV)
    # points_data = ICV.GetPoints().GetData()
    # ICV_points = vtk.util.numpy_support.vtk_to_numpy(points_data)

    # # extract SCV form septum
    # thresh = vtk.vtkThreshold()
    # thresh.SetInputData(septum)
    # thresh.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_CELLS", "phie_v")
    # thresh.ThresholdByLower(tao_scv)
    # thresh.Update()
    # SCV = thresh.GetOutput()
    # SCV = Method.extract_largest_region(SCV)
    # points_data = SCV.GetPoints().GetData()
    # SCV_points = vtk.util.numpy_support.vtk_to_numpy(points_data)

    
    # """
    # region growing to get tao_ct_minus
    # """
    # # tao_ct_minus and tao_ct_plus
    # value = -0.25
    # step = 0.005
    # touch_icv = 0
    # touch_scv = 0
    # k = 1

    # while touch_icv == 0 or touch_scv == 0:
    #     thresh = vtk.vtkThreshold()
    #     thresh.SetInputData(septum)
    #     thresh.ThresholdByLower(value)
    #     thresh.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_CELLS", "phie_w")
    #     thresh.Update()
    #     temp = thresh.GetOutput()
    #     points_data = temp.GetPoints().GetData()
    #     temp = vtk.util.numpy_support.vtk_to_numpy(points_data)

    #     touch_icv = Method.multidim_intersect_bool(ICV_points, temp)
    #     touch_scv = Method.multidim_intersect_bool(SCV_points, temp)
    #     print("touch_icv: ", touch_icv)
    #     print("touch_scv: ", touch_scv)
    #     if touch_icv == 0 or touch_scv == 0:
    #         value += step
    #     print("Iteration: ", k)
    #     print("Value of tao_ct_minus: ", value)
    #     k += 1
    # tao_ct_minus = value + 0.001
    # print('Region growing...to get the tao_ct_minus...done')
    # print("Final tao_ct_minus: ", tao_ct_minus)
    # tao_ct_plus = tao_ct_minus + 0.02
    # print("Final tao_ct_plus: ", tao_ct_plus)
    # tag = np.zeros(len(w))
    # print('Bundles selection...')
    # #### Bundles selection ####
    # for i in range(len(ab_grad)):
    #     if r[i] >= tao_tv:
    #         ab_grad[i] = r_grad[i]
    #         tag[i] = tricuspid_valve_epi
    #     else:
    #         if r[i] < tao_raw:
    #             if w[i] >= tao_ct_minus and w[i] <= tao_ct_plus:
    #                 # ab_grad[i] = w_grad[i]
    #                 tag[i] = crista_terminalis
    #             elif w[i] <= tao_ct_minus:
    #                 if v[i] >= tao_icv or v[i] <= tao_scv:
    #                     ab_grad[i] = v_grad[i]
    #                     if v[i] >= tao_icv:
    #                         tag[i] = inferior_vena_cava_epi
    #                     if v[i] <= tao_scv:
    #                         tag[i] = superior_vena_cava_epi
    #                 else:
    #                     tag[i] = right_atrial_lateral_wall_epi
    #             else:
    #                 if v[i] >= tao_icv or v[i] <= tao_scv:
    #                     ab_grad[i] = v_grad[i]
    #                     if v[i] >= tao_icv:
    #                         tag[i] = inferior_vena_cava_epi
    #                     if v[i] <= tao_scv:
    #                         tag[i] = superior_vena_cava_epi
    #                 else:
    #                     if w[i] < tao_ib:
    #                         ab_grad[i] = v_grad[i]
    #                         tag[i] = inter_caval_bundle_epi
    #                     elif w[i] > tao_ras:
    #                         ab_grad[i] = r_grad[i] #right_atrial_septum_lower_epi
    #                         tag[i] = right_atrial_septum_epi
    #                         # tag[i] =120
    #                     else:
    #                         ab_grad[i] = r_grad[i] #right_atrial_septum_upper_epi
    #                         tag[i] = right_atrial_septum_epi
    #                         # tag[i] = 130
    #         else:
    #             if v[i] >= tao_icv or v[i] <= tao_scv:
    #                 ab_grad[i] = v_grad[i]
    #                 if v[i] >= tao_icv:
    #                     tag[i] = inferior_vena_cava_epi
    #                 if v[i] <= tao_scv:
    #                     tag[i] = superior_vena_cava_epi
    #             else:
    #                 if w[i] >= 0:
    #                     ab_grad[i] = r_grad[i] #right_atrial_septum_lower_epi
    #                     tag[i] = right_atrial_septum_epi
    #                     # tag[i] = 140
    #                 else:
    #                     tag[i] = right_atrial_lateral_wall_epi
    #     if v[i] >= tao_icv or v[i] <= tao_scv:
    #         ab_grad[i] = v_grad[i]
    #         if v[i] >= tao_icv:
    #             tag[i] = inferior_vena_cava_epi
    #         if v[i] <= tao_scv:
    #             tag[i] = superior_vena_cava_epi
    # # tag = Method.assign_ra_appendage(model, SCV, ra_appex_point, tag, right_atrial_appendage_epi)
    
    # meshNew = dsa.WrapDataObject(model)
    # meshNew.CellData.append(tag, "elemTag")
    # writer = vtk.vtkUnstructuredGridWriter()
    # writer.SetFileName(job.ID+"/result_RA/RA_Tianbao_epi_with_fiber.vtk")
    # writer.SetInputData(meshNew.VTKObject)
    # writer.Write()
    
    print('Bundles selection...done')
    # normalize the gradient phie
    abs_phie_grad = np.linalg.norm(phie_grad, axis=1, keepdims=True)
    abs_phie_grad = np.where(abs_phie_grad != 0, abs_phie_grad, 1)
    phie_grad_norm = phie_grad / abs_phie_grad

    ##### Local coordinate system #####
    # et
    et = phie_grad_norm
    print('############### et ###############')
    # print(et)

    # k
    #k = ab_grad
    print('############### k ###############')
    # print(k)

    # en
    #en = ab_grad
    # for i in range(len(k)):
    #     en[i] = k[i] - np.dot(k[i], et[i]) * et[i]
        
    en = k - et*np.sum(k*et,axis=1).reshape(len(et),1)
    # normlize the en
    # abs_en = np.linalg.norm(en, axis=1, keepdims=True)
    # for i in range(len(abs_en)):
    #     if abs_en[i] == 0:
    #         abs_en[i] =1
    # en = en/abs_en
    abs_en = np.linalg.norm(en, axis=1, keepdims=True)
    abs_en = np.where(abs_en != 0, abs_en, 1)
    en = en / abs_en
    print('############### en ###############')
    # print(en)

    # el
    el = np.cross(en, et)
    
    el = Method.assign_element_fiber_around_path_within_radius(model, CT_SEPT_path, w_ct, el, smooth=True)
    
    el = np.where(el == [0,0,0], [1,0,0], el).astype("float32")

    print('############### el ###############')
    # print(el)
    end_time = datetime.datetime.now()
    running_time = end_time - start_time
    print('Calculating epicardial fibers... done! ' + str(end_time) + '\nIt takes: ' + str(running_time) + '\n')
    
    if args.mesh_type == "bilayer":
        sheet = np.cross(el, et)
        
        for i in range(model.GetPointData().GetNumberOfArrays()-1, -1, -1):
            model.GetPointData().RemoveArray(model.GetPointData().GetArrayName(i))
        
        for i in range(model.GetCellData().GetNumberOfArrays()-1, -1, -1):
            model.GetCellData().RemoveArray(model.GetCellData().GetArrayName(i))
                
        meshNew = dsa.WrapDataObject(model)
        meshNew.CellData.append(tag, "elemTag")
        meshNew.CellData.append(el, "fiber")
        meshNew.CellData.append(sheet, "sheet")

        writer = vtk.vtkUnstructuredGridWriter()
        if args.ofmt == 'vtk':
            writer = vtk.vtkUnstructuredGridWriter()
            writer.SetFileName(job.ID+"/result_RA/RA_epi_with_fiber.vtk")
            writer.SetFileTypeToBinary()
        else:
            writer = vtk.vtkXMLUnstructuredGridWriter()
            writer.SetFileName(job.ID+"/result_RA/RA_epi_with_fiber.vtu")
        writer.SetInputData(meshNew.VTKObject)
        writer.Write()
        
        """
        PM and CT
        """

        cellid = vtk.vtkIdFilter()
        cellid.CellIdsOn()
        cellid.SetInputData(meshNew.VTKObject) # vtkPolyData()
        cellid.PointIdsOn()
        if int(vtk_version) >= 9:
            cellid.SetPointIdsArrayName('Global_ids')
            cellid.SetCellIdsArrayName('Global_ids')
        else:
            cellid.SetIdsArrayName('Global_ids')
        cellid.Update()
        
        model = cellid.GetOutput()
        
        endo = vtk.vtkUnstructuredGrid()
        endo.DeepCopy(model)
        
        CT = Method.vtk_thr(model, 2,"CELLS","elemTag", crista_terminalis, crista_terminalis)
        
        CT_ids = vtk.util.numpy_support.vtk_to_numpy(CT.GetCellData().GetArray('Global_ids'))

    elif args.mesh_type == "vol":

        CT_id_list = vtk.vtkIdList()
        for var in CT_ids:
            CT_id_list.InsertNextId(var)
        for var in CT_SEPT_ids:
            CT_id_list.InsertNextId(var)

        extract = vtk.vtkExtractCells()
        extract.SetInputData(model)
        extract.SetCellList(CT_id_list)
        extract.Update()
        CT = extract.GetOutput()

        CT_ids = vtk.util.numpy_support.vtk_to_numpy(CT.GetCellData().GetArray('Global_ids'))

        if args.debug:
            
            meshNew = dsa.WrapDataObject(model)
            meshNew.CellData.append(tag, "elemTag")
            meshNew.CellData.append(el, "fiber")

            writer = vtk.vtkUnstructuredGridWriter()
            if args.ofmt == 'vtk':
                writer = vtk.vtkUnstructuredGridWriter()
                writer.SetFileName(job.ID+"/result_RA/RA_epi_with_fiber.vtk")
                writer.SetFileTypeToBinary()
            else:
                writer = vtk.vtkXMLUnstructuredGridWriter()
                writer.SetFileName(job.ID+"/result_RA/RA_epi_with_fiber.vtu")
            writer.SetInputData(meshNew.VTKObject)
            writer.Write()
    
    center = np.asarray((np.array(df["SVC"])+np.array(df["IVC"]))/2)
    
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(CT)
    loc.BuildLocator()
    
    point1_id = loc.FindClosestPoint(IVC_max_r_CT_pt)
    point2_id = loc.FindClosestPoint(SVC_CT_pt)
    
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(TV_s)
    loc.BuildLocator()
    
    point3_id = loc.FindClosestPoint(IVC_max_r_CT_pt)
    point4_id = loc.FindClosestPoint(np.array(df["RAA"]))  # this is also the id for Bachmann-Bundle on the right atrium
    
    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(CT)
    geo_filter.Update()
    CT = geo_filter.GetOutput()
    
    # calculate the norm vector
    v1 = np.array(df["IVC"]) - np.array(df["SVC"])
    v2 = np.array(df["TV"]) - np.array(df["IVC"])
    norm = np.cross(v2, v1)
    
    #normalize norm
    n = np.linalg.norm(norm)
    norm_1 = norm/n

    plane = vtk.vtkPlane()
    plane.SetNormal(norm_1[0], norm_1[1], norm_1[2])
    plane.SetOrigin(df["TV"][0], df["TV"][1], df["TV"][2])
    
    meshExtractFilter = vtk.vtkExtractGeometry()
    meshExtractFilter.SetInputData(TV_s)
    meshExtractFilter.SetImplicitFunction(plane)
    meshExtractFilter.Update()
    TV_lat = meshExtractFilter.GetOutput()
    
    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(TV_lat)
    geo_filter.Update()
    TV_lat = geo_filter.GetOutput()
    
    cln = vtk.vtkCleanPolyData()
    cln.SetInputData(TV_lat)
    cln.Update()
    TV_lat = cln.GetOutput()

    if args.debug:
        Method.writer_vtk(TV_lat, '{}_surf/'.format(args.mesh) + "TV_lat.vtk")
    
    # writer = vtk.vtkPolyDataWriter()
    # writer.SetFileName("TV_lat.vtk")
    # writer.SetInputData(TV_lat)
    # writer.Write()
    
    ct_points_data = Method.dijkstra_path(CT, point1_id, point2_id)
    
    # np.savetxt("ct_points_data.txt", ct_points_data, fmt='%.4f')

    if args.debug:
        Method.create_pts(ct_points_data, 'ct_points_data', '{}_surf/'.format(args.mesh))
    
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(TV_lat)
    loc.BuildLocator()
    
    point3_id = loc.FindClosestPoint(TV_s.GetPoint(point3_id))
    point4_id = loc.FindClosestPoint(TV_s.GetPoint(point4_id))  # this is also the id for Bachmann-Bundle on the right atrium
    
    tv_points_data = Method.dijkstra_path(TV_lat, point3_id, point4_id)

    if args.debug:
        Method.create_pts(tv_points_data, 'tv_points_data', '{}_surf/'.format(args.mesh))
    # np.savetxt("tv_points_data.txt", tv_points_data, fmt='%.4f')
    
    # np.savetxt("point3_id_new.txt", TV_lat.GetPoint(point3_id), fmt='%.4f')
    # np.savetxt("point4_id_new.txt", TV_lat.GetPoint(point4_id), fmt='%.4f')
    
    print("Creating Pectinate muscle...")

    if args.mesh_type == "vol":

        # Copy current tags and fibers, they will be overwritten by the PMs
        tag_old = np.array(tag, dtype=int)
        el_old = np.array(el)

        geo_filter = vtk.vtkGeometryFilter()
        geo_filter.SetInputData(model)
        geo_filter.Update()
        surface = geo_filter.GetOutput()

        epi = Method.vtk_thr(surface, 0,"POINTS","phie_phi", 0.5)

        endo = Method.vtk_thr(surface, 1,"POINTS","phie_phi", 0.5)

        geo_filter = vtk.vtkGeometryFilter()
        geo_filter.SetInputData(endo)
        geo_filter.Update()
        surface = geo_filter.GetOutput()

    elif args.mesh_type == "bilayer":

        fiber_endo = el.copy()
        tag_endo = np.copy(tag)
        
        geo_filter = vtk.vtkGeometryFilter()
        geo_filter.SetInputData(endo)
        geo_filter.Update()
        surface = geo_filter.GetOutput()

    loc = vtk.vtkPointLocator()
    loc.SetDataSet(surface)
    loc.BuildLocator()
    point_apx_id = loc.FindClosestPoint(np.array(df["RAA"]))
    pm_ct_id_list = []
    for i in range(len(ct_points_data)):
        pm_ct_id_list.append(loc.FindClosestPoint(ct_points_data[i]))

    pm_tv_id_list = []
    for i in range(len(tv_points_data)):
        pm_tv_id_list.append(loc.FindClosestPoint(tv_points_data[i]))

    pm_ct_dis = int(len(ct_points_data) / pm_num)
    pm_tv_dis = int(len(tv_points_data) / pm_num)
    
        # the first PM is the one to the appendage
        
        # pm = Method.dijkstra_path(surface, pm_ct_id_list[-1], point_apx_id)
        
        # tag = Method.assign_element_tag_around_path_within_radius(mesh, pm, w_pm, tag, pectinate_muscle)
        # fiber_endo = Method.assign_element_fiber_around_path_within_radius(mesh, pm, w_pm, fiber_endo, smooth=True)
        
        # for i in range(pm_num):
        #     pm_point_1 = pm_ct_id_list[(i + 1) * pm_ct_dis]
        #     pm_point_2 = pm_tv_id_list[(i + 1) * pm_tv_dis]
        #     pm = Method.dijkstra_path_on_a_plane(surface, args, pm_point_1, pm_point_2, center)
        #     # pm = Method.dijkstra_path(surface, pm_point_1, pm_point_2)
        #     # if i == 0:
        #     #     pm = Method.dijkstra_path(surface, pm_point_1, pm_point_2)
        #     # else:
        #     #     pm = Method.dijkstra_path_on_a_plane(surface, pm_point_1, pm_point_2, center)
        #     print("The ", i + 1, "th pm done")
        #     tag = Method.assign_element_tag_around_path_within_radius(mesh, pm, w_pm, tag, pectinate_muscle)
        #     print("The ", i + 1, "th pm's tag is done")
        #     fiber_endo = Method.assign_element_fiber_around_path_within_radius(mesh, pm, w_pm, fiber_endo, smooth=True)
        #     print("The ", i + 1, "th pm's fiber is done")
        
        # print("Creating Pectinate muscle... done!")
        
        # # Crista Terminalis
        # print("Creating Crista Terminalis...")
        # tag = Method.assign_element_tag_around_path_within_radius(mesh, ct_points_data, w_ct, tag, crista_terminalis)
        # fiber_endo = Method.assign_element_fiber_around_path_within_radius(mesh, ct_points_data, w_ct, fiber_endo, smooth=True)
        # print("Creating Crista Terminalis... done!")
        
        # """
        # over write the pm on the TV
        # """
        # for i in range(len(ab_grad)):
        #     if r[i] >= tao_tv:
        #         fiber_endo[i] = el[i]
        #         if phie[i] <= 0.5:
        #             tag[i] = tricuspid_valve_endo
        #         else:
        #             tag[i] = tricuspid_valve_epi
         
        # # Bachmann-Bundle
    
        # loc = vtk.vtkPointLocator()
        # loc.SetDataSet(surface)
        # loc.BuildLocator()
        
        # # Bachmann-Bundle starting point
        # bb_1_id = loc.FindClosestPoint(SVC_CT_pt)
        # bb_2_id = loc.FindClosestPoint(TV_s.GetPoint(point4_id))
        
        # bachmann_bundle_points_data = Method.dijkstra_path(surface, bb_1_id, bb_2_id)
        # bb_step = 10
        # bb_path = np.asarray([bachmann_bundle_points_data[i] for i in range(len(bachmann_bundle_points_data)) if i % bb_step == 0 or i == len(bachmann_bundle_points_data)-1])
        # spline_points = vtk.vtkPoints()
        # for i in range(len(bb_path)):
        #     spline_points.InsertPoint(i, bb_path[i][0], bb_path[i][1], bb_path[i][2])
        
        # # Fit a spline to the points
        # spline = vtk.vtkParametricSpline()
        # spline.SetPoints(spline_points)
        # functionSource = vtk.vtkParametricFunctionSource()
        # functionSource.SetParametricFunction(spline)
        # functionSource.SetUResolution(30 * spline_points.GetNumberOfPoints())
        # functionSource.Update()
        
        # bb_points = vtk.util.numpy_support.vtk_to_numpy(functionSource.GetOutput().GetPoints().GetData())
        
        # tag = Method.assign_element_tag_around_path_within_radius(model, bb_points, w_bb, tag, bachmann_bundel_right)
        # el = Method.assign_element_fiber_around_path_within_radius(model, bb_points, w_bb, el, smooth=True)
        
        # tag[SN_ids] = sinus_node
        
        # for i in range(model.GetPointData().GetNumberOfArrays()-1, -1, -1):
        #     model.GetPointData().RemoveArray(model.GetPointData().GetArrayName(i))
        
        # for i in range(model.GetCellData().GetNumberOfArrays()-1, -1, -1):
        #     model.GetCellData().RemoveArray(model.GetCellData().GetArrayName(i))
                
        # meshNew = dsa.WrapDataObject(model)
        # meshNew.CellData.append(tag, "elemTag")
        # meshNew.CellData.append(el, "fiber")
        # meshNew.CellData.append(sheet, "sheet")
        # writer = vtk.vtkUnstructuredGridWriter()
        # writer.SetFileName(job.ID+"/result_RA/RA_epi_with_fiber.vtk")
        # writer.SetInputData(meshNew.VTKObject)
        # writer.Write()
        
        # # Bachmann_Bundle internal connection
        # la_connect_point, ra_connect_point = Method.get_connection_point_la_and_ra(df["LAA"])
        # la_connect_point = np.asarray(la_connect_point)
        # ra_connect_point = np.asarray(ra_connect_point)
        
        # la_epi = Method.smart_reader("/home/luca/IBT/AtrialLDRBM_la816/LDRBM/Fiber_LA/result_RA/LA_epi_with_fiber.vtk")
        # la_appendage_basis_point = np.asarray(df["LAA_basis_inf"])
        # length = len(bachmann_bundle_points_data)
        # ra_bb_center = bachmann_bundle_points_data[int(length * 0.45)]
        # # TODO PLAN D
        # geo_filter_la_epi = vtk.vtkGeometryFilter()
        # geo_filter_la_epi.SetInputData(la_epi)
        # geo_filter_la_epi.Update()
        # la_epi = geo_filter_la_epi.GetOutput()
    
        # geo_filter_ra_epi = vtk.vtkGeometryFilter()
        # geo_filter_ra_epi.SetInputData(model)
        # geo_filter_ra_epi.Update()
        # ra_epi = geo_filter_ra_epi.GetOutput()
    
        # loc_la_epi = vtk.vtkPointLocator()
        # loc_la_epi.SetDataSet(la_epi)
        # loc_la_epi.BuildLocator()
    
        # loc_ra_epi = vtk.vtkPointLocator()
        # loc_ra_epi.SetDataSet(ra_epi)
        # loc_ra_epi.BuildLocator()
    
        # ra_a_id = loc_ra_epi.FindClosestPoint(ra_bb_center)
        # ra_b_id = loc_ra_epi.FindClosestPoint(ra_connect_point)
        # la_c_id = loc_la_epi.FindClosestPoint(la_connect_point)
        # la_d_id = loc_la_epi.FindClosestPoint(la_appendage_basis_point)
        # path_1 = Method.dijkstra_path(ra_epi, ra_a_id, ra_b_id)
        # path_2 = Method.dijkstra_path(la_epi, la_c_id, la_d_id)
        # path_all_temp = np.vstack((path_1, path_2))
        # # down sampling to smooth the path
        # step = 10
        # path_all = np.asarray([path_all_temp[i] for i in range(len(path_all_temp)) if i % step == 0 or i == len(path_all_temp)-1])
        
        # # save points for bb fiber
        # filename = job.ID+'/bridges/bb_fiber.dat'
        # f = open(filename, 'wb')
        # pickle.dump(path_all, f)
        # f.close()
        
        
        # # BB tube
        
        # bb_tube = Method.creat_tube_around_spline(path_all, 2)
        # sphere_a = Method.creat_sphere(la_appendage_basis_point, 2 * 1.02)
        # sphere_b = Method.creat_sphere(ra_bb_center, 2 * 1.02)
        # Method.smart_bridge_writer(bb_tube, sphere_a, sphere_b, "BB_intern_bridges")
    
    
        # # tag = Method.assign_element_tag_around_path_within_radius(mesh, path_bb_ra, w_bb, tag, bachmann_bundel_right)
        # # fiber_endo = Method.assign_element_fiber_around_path_within_radius(mesh, path_bb_ra, w_bb, fiber_endo,
        # #                                                                  smooth=True)
    
        # tag_data = vtk.util.numpy_support.numpy_to_vtk(tag, deep=True, array_type=vtk.VTK_INT)
        # tag_data.SetNumberOfComponents(1)
        # tag_data.SetName("elemTag")
        # model.GetCellData().RemoveArray("elemTag")
        # model.GetCellData().SetScalars(tag_data)
    
        # #model.GetCellData().AddArray(tag_data)
        # abs_el = np.linalg.norm(fiber_endo, axis=1, keepdims=True)
        # interpolate_arr = np.asarray([0, 0, 1])
        # index = np.argwhere(abs_el == 0)
        # print('There is',len(index),'zero vector(s).')
        # for var in index:
        #     fiber_endo[var[0]] = interpolate_arr
    
        # fiber_data = vtk.util.numpy_support.numpy_to_vtk(fiber_endo, deep=True, array_type=vtk.VTK_DOUBLE)
        # fiber_data.SetNumberOfComponents(3)
        # fiber_data.SetName("fiber")
        # model.GetCellData().SetVectors(fiber_data)
    
    
        # start_time = datetime.datetime.now()
        # print('Writing as RA_with_fiber... ' + str(start_time))
        # meshNew = dsa.WrapDataObject(mesh)
        # writer = vtk.vtkUnstructuredGridWriter()
        # writer.SetFileName(job.ID+"/result_RA/RA_with_fiber.vtk")
        # writer.SetInputData(meshNew.VTKObject)
        # writer.Write()
        # end_time = datetime.datetime.now()
        # running_time = end_time - start_time
        # print('Writing as RA_with_fiber... done! ' + str(end_time) + '\nIt takes: ' + str(running_time) + '\n')
        
    # Pectinate muscle
    print("Creating Pectinate muscle 1")
    # the first PM is the one to the appendage
    #pm = Method.dijkstra_path(surface, pm_ct_id_list[-1], point_apx_id)
    loc = vtk.vtkPointLocator()

    loc.SetDataSet(surface)
    loc.BuildLocator()
    RAA_CT_id = loc.FindClosestPoint(RAA_CT_pt)

    # Pectinate muscle going from the septum spurius to the RAA apex
    pm = Method.dijkstra_path(surface, RAA_CT_id, point_apx_id)

    pm = Method.downsample_path(pm, int(len(pm)*0.1))
    if args.debug:
        Method.create_pts(pm, 'pm_0_downsampled', '{}_surf/'.format(args.mesh))


    if args.mesh_type == "bilayer":
    
        tag_endo = Method.assign_element_tag_around_path_within_radius(endo, pm, w_pm, tag_endo, pectinate_muscle)
        fiber_endo = Method.assign_element_fiber_around_path_within_radius(endo, pm, w_pm, fiber_endo, smooth=False)
    
    elif args.mesh_type == "vol":

        filter_cell_centers = vtk.vtkCellCenters()
        filter_cell_centers.SetInputData(model)
        filter_cell_centers.Update()
        centroids_array = vtk.util.numpy_support.vtk_to_numpy(filter_cell_centers.GetOutput().GetPoints().GetData())

        tree = cKDTree(centroids_array)

        ii = tree.query_ball_point(pm, r = w_pm, n_jobs=-1)

        ii = list(set([item for sublist in ii for item in sublist]))

        tag[ii] = pectinate_muscle

        el = Method.assign_element_fiber_around_path_within_radius(model, pm, w_pm, el, smooth=False)

    for i in range(4,pm_num-1): # skip the first 3 pm as they will be in the IVC side
        pm_point_1 = pm_ct_id_list[(i + 1) * pm_ct_dis]
        pm_point_2 = pm_tv_id_list[(i + 1) * pm_tv_dis]
        pm = Method.dijkstra_path_on_a_plane(surface, args, pm_point_1, pm_point_2, center)

        #skip first 3% of the points since they will be on the roof of the RA
        pm = pm[int(len(pm)*0.03):,:] #


        pm = Method.downsample_path(pm, int(len(pm)*0.065))

        if args.debug:
            Method.create_pts(pm, 'pm_'+ str(i+1) +'_downsampled', '{}_surf/'.format(args.mesh))


        print("The ", i + 1, "th pm done")
        if args.mesh_type == "bilayer":

            tag_endo = Method.assign_element_tag_around_path_within_radius(endo, pm, w_pm, tag_endo, pectinate_muscle)
            print("The ", i + 1, "th pm's tag is done")
            fiber_endo = Method.assign_element_fiber_around_path_within_radius(endo, pm, w_pm, fiber_endo, smooth=False)
            print("The ", i + 1, "th pm's fiber is done")

        elif args.mesh_type == "vol":

            ii = tree.query_ball_point(pm, r = w_pm, n_jobs=-1)

            ii = list(set([item for sublist in ii for item in sublist]))

            tag[ii] = pectinate_muscle

            el = Method.assign_element_fiber_around_path_within_radius(model, pm, w_pm, el, smooth=False)
        
        # pm = Method.dijkstra_path(surface, pm_ct_id_list[-1], point_apx_id)
        
        # tag_endo = Method.assign_element_tag_around_path_within_radius(endo, pm, w_pm, tag_endo, pectinate_muscle)
        # fiber_endo = Method.assign_element_fiber_around_path_within_radius(endo, pm, w_pm, fiber_endo, smooth=True)
        
        # for i in range(pm_num):
        #     pm_point_1 = pm_ct_id_list[(i + 1) * pm_ct_dis]
        #     pm_point_2 = pm_tv_id_list[(i + 1) * pm_tv_dis]
            
        #     pm = Method.dijkstra_path_on_a_plane(surface, args, pm_point_1, pm_point_2, center)
        #     pm = pm[int(len(pm)*0.05):,:] 
        #     print("The ", i + 1, "th pm done")
        #     tag_endo = Method.assign_element_tag_around_path_within_radius(endo, pm, w_pm, tag_endo, pectinate_muscle)
        #     fiber_endo = Method.assign_element_fiber_around_path_within_radius(endo, pm, w_pm, fiber_endo, smooth=True)
    
    if args.mesh_type == "bilayer":

        print("Creating Pectinate muscle... done!")
        
        # Crista Terminalis
        print("Creating Crista Terminalis...")
        tag_endo[CT_ids] = tag[CT_ids]
        fiber_endo[CT_ids] = el[CT_ids]
        print("Creating Crista Terminalis... done!")
        
        """
        overwrite the pm on the TV
        """
        tag_endo[TV_ids] = tag[TV_ids]
        fiber_endo[TV_ids] = el[TV_ids]
        
        tag_endo[IVC_ids] = tag[IVC_ids]
        fiber_endo[IVC_ids] = el[IVC_ids]
        
        tag_endo[SN_ids] = sinus_node

        tag_endo[IVC_ids] = tag[IVC_ids]
        
        for i in range(endo.GetPointData().GetNumberOfArrays()-1, -1, -1):
            endo.GetPointData().RemoveArray(endo.GetPointData().GetArrayName(i))
        
        for i in range(endo.GetCellData().GetNumberOfArrays()-1, -1, -1):
            endo.GetCellData().RemoveArray(endo.GetCellData().GetArrayName(i))
        
        fiber_endo = np.where(fiber_endo == [0,0,0], [1,0,0], fiber_endo).astype("float32")
        sheet = np.cross(fiber_endo, et)
        sheet = np.where(sheet == [0,0,0], [1,0,0], sheet).astype("float32")
        meshNew = dsa.WrapDataObject(endo)
        meshNew.CellData.append(tag_endo, "elemTag")
        meshNew.CellData.append(fiber_endo, "fiber")
        meshNew.CellData.append(sheet, "sheet")
        
        endo = meshNew.VTKObject
        
        writer = vtk.vtkUnstructuredGridWriter()

        if args.ofmt == 'vtk':
            writer = vtk.vtkUnstructuredGridWriter()
            writer.SetFileName(job.ID+"/result_RA/RA_endo_with_fiber.vtk")
            writer.SetFileTypeToBinary()
        else:
            writer = vtk.vtkXMLUnstructuredGridWriter()
            writer.SetFileName(job.ID+"/result_RA/RA_endo_with_fiber.vtu")
        writer.SetInputData(endo)
        writer.Write()
        
        CT_PMs = Method.vtk_thr(endo,2,"CELLS","elemTag",pectinate_muscle, crista_terminalis)
        
        writer = vtk.vtkUnstructuredGridWriter()
        if args.ofmt == 'vtk':
            writer = vtk.vtkUnstructuredGridWriter()
            writer.SetFileName(job.ID+"/result_RA/RA_CT_PMs.vtk")
            writer.SetFileTypeToBinary()
        else:
            writer = vtk.vtkXMLUnstructuredGridWriter()
            writer.SetFileName(job.ID+"/result_RA/RA_CT_PMs.vtu")
        writer.SetInputData(CT_PMs)
        writer.Write()

    elif args.mesh_type == "vol":
        
        print("Creating Pectinate muscle... done!")
        
        # Crista Terminalis
        print("Creating Crista Terminalis...")
        tag[CT_ids] = tag_old[CT_ids]
        el[CT_ids] = el_old[CT_ids]
        print("Creating Crista Terminalis... done!")
        
        """
        overwrite the pm on the TV
        """
        tag[TV_ids] = tag_old[TV_ids]
        el[TV_ids] = el_old[TV_ids]
        
        tag[IVC_ids] = tag_old[IVC_ids]
        el[IVC_ids] = el_old[IVC_ids]
        
        tag[SN_ids] = sinus_node

        tag[IVC_ids] = tag_old[IVC_ids]
        
        if args.debug:

            for i in range(model.GetPointData().GetNumberOfArrays()-1, -1, -1):
                model.GetPointData().RemoveArray(model.GetPointData().GetArrayName(i))
            
            for i in range(model.GetCellData().GetNumberOfArrays()-1, -1, -1):
                model.GetCellData().RemoveArray(model.GetCellData().GetArrayName(i))
            
            el = np.where(el == [0,0,0], [1,0,0], el).astype("float32")
            sheet = np.cross(el, et)
            sheet = np.where(sheet == [0,0,0], [1,0,0], sheet).astype("float32")
            meshNew = dsa.WrapDataObject(model)
            meshNew.CellData.append(tag, "elemTag")
            meshNew.CellData.append(el, "fiber")
            meshNew.CellData.append(sheet, "sheet")
            
            writer = vtk.vtkUnstructuredGridWriter()
            if args.ofmt == 'vtk':
                writer = vtk.vtkUnstructuredGridWriter()
                writer.SetFileName(job.ID+"/result_RA/RA_endo_with_fiber.vtk")
                writer.SetFileTypeToBinary()
            else:
                writer = vtk.vtkXMLUnstructuredGridWriter()
                writer.SetFileName(job.ID+"/result_RA/RA_endo_with_fiber.vtu")
            writer.SetInputData(meshNew.VTKObject)
            writer.Write()
        
    # Bachmann-Bundle
    if args.mesh_type =="vol":

        geo_filter = vtk.vtkGeometryFilter()
        geo_filter.SetInputData(epi)
        geo_filter.Update()
        surface = geo_filter.GetOutput()

    loc = vtk.vtkPointLocator()
    loc.SetDataSet(RAS_S)
    loc.BuildLocator()
        
    bb_c_id = loc.FindClosestPoint((np.array(df["TV"])+np.array(df["SVC"]))/2)
        
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(surface)
    loc.BuildLocator()
        
    # Bachmann-Bundle starting point
    bb_1_id = loc.FindClosestPoint(SVC_CT_pt)
    bb_2_id = loc.FindClosestPoint(TV_lat.GetPoint(point4_id))
    bb_c_id = loc.FindClosestPoint(RAS_S.GetPoint(bb_c_id))

    bachmann_bundle_points_data_1 = Method.dijkstra_path(surface, bb_1_id, bb_c_id)
    
    bachmann_bundle_points_data_2 = Method.dijkstra_path(surface, bb_c_id, bb_2_id)
    
    bachmann_bundle_points_data = np.concatenate((bachmann_bundle_points_data_1, bachmann_bundle_points_data_2), axis=0)
    
    np.savetxt('bb.txt',bachmann_bundle_points_data,fmt='%.5f') # Change directory

    bb_step = int(len(bachmann_bundle_points_data)*0.1)
    bb_path = np.asarray([bachmann_bundle_points_data[i] for i in range(len(bachmann_bundle_points_data)) if i % bb_step == 0 or i == len(bachmann_bundle_points_data)-1])
    spline_points = vtk.vtkPoints()
    for i in range(len(bb_path)):
        spline_points.InsertPoint(i, bb_path[i][0], bb_path[i][1], bb_path[i][2])
    
    # Fit a spline to the points
    spline = vtk.vtkParametricSpline()
    spline.SetPoints(spline_points)
    functionSource = vtk.vtkParametricFunctionSource()
    functionSource.SetParametricFunction(spline)
    functionSource.SetUResolution(30 * spline_points.GetNumberOfPoints())
    functionSource.Update()
        
    bb_points = vtk.util.numpy_support.vtk_to_numpy(functionSource.GetOutput().GetPoints().GetData())
        
    tag = Method.assign_element_tag_around_path_within_radius(model, bb_points, w_bb, tag, bachmann_bundel_right)
    el = Method.assign_element_fiber_around_path_within_radius(model, bb_points, w_bb, el, smooth=True)
        
    tag[SN_ids] = sinus_node
        
    for i in range(model.GetPointData().GetNumberOfArrays()-1, -1, -1):
        model.GetPointData().RemoveArray(model.GetPointData().GetArrayName(i))
        
    for i in range(model.GetCellData().GetNumberOfArrays()-1, -1, -1):
        model.GetCellData().RemoveArray(model.GetCellData().GetArrayName(i))
        
    el = np.where(el == [0,0,0], [1,0,0], el).astype("float32")
    sheet = np.cross(el, et)
    sheet = np.where(sheet == [0,0,0], [1,0,0], sheet).astype("float32")
    meshNew = dsa.WrapDataObject(model)
    meshNew.CellData.append(tag, "elemTag")
    meshNew.CellData.append(el, "fiber")
    meshNew.CellData.append(sheet, "sheet")
    writer = vtk.vtkUnstructuredGridWriter()
    
    if args.mesh_type == "bilayer":
        if args.ofmt == 'vtk':
            writer = vtk.vtkUnstructuredGridWriter()
            writer.SetFileName(job.ID+"/result_RA/RA_epi_with_fiber.vtk")
            writer.SetFileTypeToBinary()
        else:
            writer = vtk.vtkXMLUnstructuredGridWriter()
            writer.SetFileName(job.ID+"/result_RA/RA_epi_with_fiber.vtu")
    else:
        if args.ofmt == 'vtk':
            writer = vtk.vtkUnstructuredGridWriter()
            writer.SetFileName(job.ID+"/result_RA/RA_vol_with_fiber.vtk")
            writer.SetFileTypeToBinary()
        else:
            writer = vtk.vtkXMLUnstructuredGridWriter()
            writer.SetFileName(job.ID+"/result_RA/RA_vol_with_fiber.vtu")
    writer.SetInputData(meshNew.VTKObject)
    writer.Write()
    
    model = meshNew.VTKObject

    if args.add_bridges:
        # Bachmann_Bundle internal connection

        if args.mesh_type == "bilayer":
            if args.ofmt == 'vtk':
                la_epi = Method.smart_reader(job.ID+"/result_LA/LA_epi_with_fiber.vtk")
            else:
                la_epi = Method.smart_reader(job.ID+"/result_LA/LA_epi_with_fiber.vtu")

        elif args.mesh_type == "vol":
            # extension = args.mesh.split('_RA_vol')[-1]
            meshname = args.mesh[:-7]

            if args.ofmt == 'vtk':
                la = Method.smart_reader(meshname+"_LA_vol_fibers/result_LA/LA_vol_with_fiber.vtk")
            else:
                la = Method.smart_reader(meshname+"_LA_vol_fibers/result_LA/LA_vol_with_fiber.vtu")

            geo_filter = vtk.vtkGeometryFilter()
            geo_filter.SetInputData(la)
            geo_filter.Update()
            la_surf = geo_filter.GetOutput()

            la_epi = Method.vtk_thr(la_surf,2,"CELLS","elemTag",left_atrial_wall_epi, 99)
        
            df = pd.read_csv(meshname+"_LA_vol_surf/rings_centroids.csv")

        la_appendage_basis_point = np.asarray(df["LAA_basis_inf"])
        length = len(bachmann_bundle_points_data)
        ra_bb_center = bachmann_bundle_points_data[int(length * 0.45)]
            
        geo_filter_la_epi = vtk.vtkGeometryFilter()
        geo_filter_la_epi.SetInputData(la_epi)
        geo_filter_la_epi.Update()
        la_epi = geo_filter_la_epi.GetOutput()
            
        if args.mesh_type == "bilayer":
            geo_filter_ra_epi = vtk.vtkGeometryFilter()
            geo_filter_ra_epi.SetInputData(model)
            geo_filter_ra_epi.Update()
            ra_epi = geo_filter_ra_epi.GetOutput()
        else:
            ra_epi = surface
        
        loc_la_epi = vtk.vtkPointLocator()
        loc_la_epi.SetDataSet(la_epi)
        loc_la_epi.BuildLocator()
        
        loc_ra_epi = vtk.vtkPointLocator()
        loc_ra_epi.SetDataSet(ra_epi)
        loc_ra_epi.BuildLocator()
            
        ra_a_id = loc_ra_epi.FindClosestPoint(ra_bb_center)
        la_c_id = loc_la_epi.FindClosestPoint(ra_bb_center)
        ra_b_id = loc_ra_epi.FindClosestPoint(la_epi.GetPoint(la_c_id))
        la_d_id = loc_la_epi.FindClosestPoint(la_appendage_basis_point)
        
        
        path_1 = Method.dijkstra_path(ra_epi, ra_a_id, ra_b_id)
        path_2 = Method.dijkstra_path(la_epi, la_c_id, la_d_id)
        path_all_temp = np.vstack((path_1, path_2))
        # down sampling to smooth the path
        step = 20
        #step = int(len(path_all_temp)*0.1)
        path_all = np.asarray([path_all_temp[i] for i in range(len(path_all_temp)) if i % step == 0 or i == len(path_all_temp)-1])
        
        # save points for bb fiber
        filename = job.ID+'/bridges/bb_fiber.dat'
        f = open(filename, 'wb')
        pickle.dump(path_all, f)
        f.close()
            
        # BB tube
            
        bb_tube = Method.creat_tube_around_spline(path_all, 2*args.scale)
        sphere_a = Method.creat_sphere(la_appendage_basis_point, 2 * 1.02*args.scale)
        sphere_b = Method.creat_sphere(ra_bb_center, 2 * 1.02*args.scale)
        Method.smart_bridge_writer(bb_tube, sphere_a, sphere_b, "BB_intern_bridges", job)
        
        df = pd.read_csv(args.mesh+"_surf/rings_centroids.csv")

        try:
            CS_p = np.array(df["CS"])
        except KeyError:
            CS_p = IVC_SEPT_CT_pt
            print("No CS found, use last CT point instead")
            
        if args.mesh_type == "bilayer":    
            add_free_bridge(args, la_epi, model, CS_p, df, job)
        elif args.mesh_type == "vol":    
            add_free_bridge(args, la, model, CS_p, df, job)
