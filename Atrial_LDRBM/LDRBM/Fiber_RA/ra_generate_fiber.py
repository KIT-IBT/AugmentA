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
import csv
import datetime
import os
import pickle

import numpy as np
import pandas as pd
import vtk
from scipy.spatial import cKDTree
from vtk.numpy_interface import dataset_adapter as dsa

import Atrial_LDRBM.LDRBM.Fiber_RA.Methods_RA as Method
from vtk_opencarp_helper_methods.AugmentA_methods.vtk_operations import vtk_thr
from vtk_opencarp_helper_methods.vtk_methods.converters import vtk_to_numpy
from vtk_opencarp_helper_methods.vtk_methods.exporting import vtk_unstructured_grid_writer, \
    vtk_xml_unstructured_grid_writer
from vtk_opencarp_helper_methods.vtk_methods.filters import apply_vtk_geom_filter, clean_polydata, generate_ids, \
    get_cells_with_ids, apply_extract_cell_filter, get_elements_above_plane
from vtk_opencarp_helper_methods.vtk_methods.finder import find_closest_point
from vtk_opencarp_helper_methods.vtk_methods.init_objects import initialize_plane

EXAMPLE_DIR = os.path.dirname(os.path.realpath(__file__))

from Atrial_LDRBM.LDRBM.Fiber_RA.create_bridges import add_free_bridge

vtk_version = vtk.vtkVersion.GetVTKSourceVersion().split()[-1].split('.')[0]


def ra_generate_fiber(model, args, job):
    simid = job.ID + "/result_RA"
    try:
        os.makedirs(simid)
    except OSError:
        print(f"Creation of the directory {simid} failed")
    else:
        print(f"Successfully created the directory {simid} ")

    simid = job.ID + "/bridges"
    try:
        os.makedirs(simid)
    except OSError:
        print(f"Creation of the directory {simid} failed")
    else:
        print(f"Successfully created the directory {simid} ")

    # Riunet
    tao_tv = 0.9
    tao_icv = 0.95  # 0.05 # dk01 biatrial fit_both
    tao_scv = 0.10
    tao_ct_plus = -0.10
    tao_ct_minus = -0.13
    tao_ib = -0.06
    tao_ras = 0.13
    tao_raw = 0.55
    with open(os.path.join(EXAMPLE_DIR, '../../element_tag.csv')) as f:
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
    w_ct = 4.62 * args.scale

    # size(Radius) of pectinate muscle in mm
    w_pm = 0.66 * args.scale

    # size(Radius) of Bachmann Bundle in mm
    w_bb = 2 * args.scale

    # radius sinus node
    r_SN = 2.5 * args.scale

    # ab
    ab = model.GetCellData().GetArray('phie_ab')
    ab_grad = model.GetCellData().GetArray('grad_ab')
    ab = vtk_to_numpy(ab)
    ab_grad = vtk_to_numpy(ab_grad)

    # v
    v = model.GetCellData().GetArray('phie_v')
    v_grad = model.GetCellData().GetArray('grad_v')
    v = vtk_to_numpy(v)
    v_grad = vtk_to_numpy(v_grad)

    # r
    r = model.GetCellData().GetArray('phie_r')
    r_grad = model.GetCellData().GetArray('grad_r')
    r = vtk_to_numpy(r)
    r_grad = vtk_to_numpy(r_grad)

    # w
    w = model.GetCellData().GetArray('phie_w')
    w_grad = model.GetCellData().GetArray('grad_w')
    w = vtk_to_numpy(w)
    w_grad = vtk_to_numpy(w_grad)

    # phie
    if args.mesh_type == "vol":
        phie = model.GetCellData().GetArray('phie_phi')
        phie = vtk_to_numpy(phie)
    phie_grad = model.GetCellData().GetArray('grad_phi')
    phie_grad = vtk_to_numpy(phie_grad)

    start_time = datetime.datetime.now()
    print('Calculating fibers... ' + str(start_time))

    model = generate_ids(model, "Global_ids", "Global_ids")

    # TV

    tag = np.zeros((len(ab),), dtype=int)

    k = np.copy(ab_grad)

    # # # Get valve using Laplacian solutions
    # Use fixed thickness

    ring_ids = np.loadtxt(f'{args.mesh}_surf/' + 'ids_TV.vtx', skiprows=2, dtype=int)

    rings_pts = vtk_to_numpy(model.GetPoints().GetData())[ring_ids, :]

    if args.debug:
        Method.create_pts(rings_pts, 'TV_ring', f'{args.mesh}_surf/')

    TV_ids = Method.get_element_ids_around_path_within_radius(model, rings_pts, 4 * args.scale)

    TV_s = get_cells_with_ids(model, TV_ids)

    ra_diff = list(
        set(list(vtk_to_numpy(model.GetCellData().GetArray('Global_ids')))).difference(
            set(TV_ids)))

    no_TV_s = get_cells_with_ids(model, ra_diff)

    # To check if TV was correctly identified
    if args.debug:
        Method.writer_vtk(TV_s, f'{args.mesh}_surf/' + "tv_s.vtk")
        Method.writer_vtk(no_TV_s, f'{args.mesh}_surf/' + "no_tv_s.vtk")

    # del ra_TV, ra_diff, ra_no_TV

    tag[TV_ids] = tricuspid_valve_epi

    k[TV_ids] = r_grad[TV_ids]

    IVC_s = vtk_thr(no_TV_s, 0, "CELLS", "phie_v", tao_icv)  # Changed 0-1 because ICV and SVC are inverted
    no_IVC_s = vtk_thr(no_TV_s, 1, "CELLS", "phie_v", tao_icv)  # Changed 1-0

    IVC_s = Method.extract_largest_region(IVC_s)  # Added

    max_phie_r_ivc = np.max(vtk_to_numpy(IVC_s.GetCellData().GetArray('phie_r'))) + 0.2

    RAW_s = vtk_thr(no_TV_s, 1, "CELLS", "phie_r", max_phie_r_ivc)  # Added +0.03 fro dk01

    SVC_s = vtk_thr(RAW_s, 1, "CELLS", "phie_v", tao_scv)  # Changed 1->0
    no_SVC_s = vtk_thr(RAW_s, 0, "CELLS", "phie_v", tao_scv)  # Changed 0->1

    SVC_s = Method.extract_largest_region(SVC_s)

    if args.debug:  # CHECK
        Method.writer_vtk(IVC_s, f'{args.mesh}_surf/' + "ivc_s.vtk")
        Method.writer_vtk(no_IVC_s, f'{args.mesh}_surf/' + "no_ivc_s.vtk")
        Method.writer_vtk(SVC_s, f'{args.mesh}_surf/' + "svc_s.vtk")
        Method.writer_vtk(no_SVC_s, f'{args.mesh}_surf/' + "no_svc_s.vtk")
        Method.writer_vtk(RAW_s, f'{args.mesh}_surf/' + "raw_s.vtk")

    tao_ct_plus = np.min(vtk_to_numpy(SVC_s.GetCellData().GetArray('phie_w')))

    SVC_CT_pt = SVC_s.GetPoint(np.argmin(vtk_to_numpy(SVC_s.GetPointData().GetArray('phie_w'))))

    tao_ct_minus = np.min(vtk_to_numpy(IVC_s.GetCellData().GetArray('phie_w')))

    IVC_CT_pt = IVC_s.GetPoint(np.argmin(vtk_to_numpy(IVC_s.GetPointData().GetArray('phie_w'))))

    IVC_SEPT_CT_pt = IVC_s.GetPoint(
        np.argmax(vtk_to_numpy(IVC_s.GetPointData().GetArray('phie_w'))))

    IVC_max_r_CT_pt = IVC_s.GetPoint(np.argmax(vtk_to_numpy(
        IVC_s.GetPointData().GetArray('phie_r'))))  # not always the best choice for pm1

    CT_band = vtk_thr(RAW_s, 2, "CELLS", "phie_w", 0.1, tao_ct_plus)  # grad_w
    CT_band = Method.extract_largest_region(CT_band)
    CT_ub = vtk_thr(RAW_s, 2, "CELLS", "phie_w", tao_ct_plus - 0.02, tao_ct_plus)  # grad_w

    CT_ub = Method.extract_largest_region(CT_ub)

    if args.debug:
        Method.writer_vtk(CT_band, f'{args.mesh}_surf/' + "ct_band.vtk")
        Method.writer_vtk(CT_ub, f'{args.mesh}_surf/' + "ct_ub.vtk")

    mesh_surf = apply_vtk_geom_filter(CT_band)

    IVC_CT_pt_id = find_closest_point(mesh_surf, np.array(IVC_CT_pt))

    SVC_CT_pt_id = find_closest_point(mesh_surf, np.array(SVC_CT_pt))

    CT_ub_pts = Method.dijkstra_path(mesh_surf, IVC_CT_pt_id, SVC_CT_pt_id)

    filter_cell_centers = vtk.vtkCellCenters()
    filter_cell_centers.SetInputData(CT_band)
    filter_cell_centers.Update()
    centroids = filter_cell_centers.GetOutput().GetPoints()
    centroids_array = vtk_to_numpy(centroids.GetData())

    tree = cKDTree(centroids_array)

    ii = tree.query_ball_point(CT_ub_pts, r=7 * args.scale)  # , n_jobs=-1)

    ii = set([item for sublist in ii for item in sublist])

    CT_band = get_cells_with_ids(CT_band, ii)

    if args.debug:
        Method.writer_vtk(CT_band, f'{args.mesh}_surf/' + "ct_band_2.vtk")

    CT_band_ids = vtk_to_numpy(CT_band.GetCellData().GetArray('Global_ids'))

    tao_RAA = np.max(vtk_to_numpy(CT_band.GetCellData().GetArray('phie_v2')))

    IVC_CT_pt_id = find_closest_point(CT_band, np.array(IVC_CT_pt))

    no_IVC_s = apply_vtk_geom_filter(no_IVC_s)

    IVC_CT_pt_id = find_closest_point(no_IVC_s, np.array(CT_band.GetPoint(IVC_CT_pt_id)))
    IVC_max_r_CT_pt_id = find_closest_point(no_IVC_s, np.array(IVC_max_r_CT_pt))
    IVC_SEPT_CT_pt_id = find_closest_point(no_IVC_s, np.array(IVC_SEPT_CT_pt))

    CT_SEPT_path = np.concatenate((Method.dijkstra_path(no_IVC_s, IVC_CT_pt_id, IVC_max_r_CT_pt_id),
                                   Method.dijkstra_path(no_IVC_s, IVC_max_r_CT_pt_id, IVC_SEPT_CT_pt_id)), axis=0)

    CT_SEPT_ids = Method.get_element_ids_around_path_within_radius(no_IVC_s, CT_SEPT_path, w_ct)

    # SVC_CT_pt_id = loc.FindClosestPoint(SVC_CT_pt)

    CT_minus = vtk_thr(RAW_s, 1, "CELLS", "phie_w", tao_ct_plus)  # grad_ab

    RAW_I_ids = vtk_to_numpy(CT_minus.GetCellData().GetArray('Global_ids'))

    ii = set(RAW_I_ids) - set(CT_SEPT_ids) - set(CT_band_ids)

    CT_minus = get_cells_with_ids(CT_minus, ii)

    RAW_I_ids = vtk_to_numpy(CT_minus.GetCellData().GetArray('Global_ids'))

    tag[RAW_I_ids] = right_atrial_lateral_wall_epi

    k[TV_ids] = r_grad[TV_ids]

    CT_plus = vtk_thr(RAW_s, 0, "CELLS", "phie_w", tao_ct_plus)

    RAW_S = vtk_thr(CT_plus, 2, "CELLS", "phie_v", tao_scv,
                    tao_icv)  # IB_S grad_v Changed order tao_scv, tao_icv

    RAW_S_ids = vtk_to_numpy(RAW_S.GetCellData().GetArray('Global_ids'))

    tag[RAW_S_ids] = right_atrial_lateral_wall_epi

    k[RAW_S_ids] = ab_grad[RAW_S_ids]

    IB = vtk_thr(RAW_S, 1, "CELLS", "phie_r", 0.05)  # grad_r or w

    IB_ids = vtk_to_numpy(IB.GetCellData().GetArray('Global_ids'))

    tag[IB_ids] = inter_caval_bundle_epi  # Change to 68

    if args.debug:
        Method.writer_vtk(IB, f'{args.mesh}_surf/' + "ib.vtk")
        Method.writer_vtk(RAW_S, f'{args.mesh}_surf/' + "raw_s.vtk")
        Method.writer_vtk(CT_plus, f'{args.mesh}_surf/' + "ct_plus.vtk")

    k[IB_ids] = v_grad[IB_ids]

    df = pd.read_csv(args.mesh + "_surf/rings_centroids.csv")

    # calculate the norm vector
    v1 = np.array(df["IVC"]) - np.array(df["SVC"])
    v2 = np.array(df["TV"]) - np.array(df["IVC"])
    norm = np.cross(v1, v2)

    # normalize norm
    n = np.linalg.norm(norm)
    norm_1 = norm / n  # Changed sign

    plane = initialize_plane(norm_1, df["TV"])

    septal_surf = get_elements_above_plane(apply_vtk_geom_filter(RAW_S), plane)

    RAS_S = vtk_thr(septal_surf, 0, "CELLS", "phie_w", tao_ct_plus)
    RAS_S = vtk_thr(RAS_S, 0, "CELLS", "phie_r", 0.05)  # grad_r or w

    if args.debug:
        Method.writer_vtk(septal_surf, f'{args.mesh}_surf/' + "septal_surf.vtk")
        Method.writer_vtk(RAS_S, f'{args.mesh}_surf/' + "ras_s.vtk")

    RAS_S_ids = vtk_to_numpy(RAS_S.GetCellData().GetArray('Global_ids'))

    tag[RAS_S_ids] = right_atrial_septum_epi

    k[RAS_S_ids] = r_grad[RAS_S_ids]

    RAW_low = vtk_thr(no_TV_s, 0, "CELLS", "phie_r", max_phie_r_ivc)

    RAS_low = get_elements_above_plane(RAW_low, plane)

    RAS_low = vtk_thr(RAS_low, 0, "CELLS", "phie_w", 0)  # grad_r overwrites the previous

    RAS_low_ids = vtk_to_numpy(RAS_low.GetCellData().GetArray('Global_ids'))

    tag[RAS_low_ids] = right_atrial_septum_epi

    k[RAS_low_ids] = r_grad[RAS_low_ids]

    RAW_low = vtk_thr(RAW_low, 1, "CELLS", "phie_w", 0)  # grad_ab

    RAW_low_ids = vtk_to_numpy(RAW_low.GetCellData().GetArray('Global_ids'))

    tag[RAW_low_ids] = right_atrial_lateral_wall_epi

    k[RAW_low_ids] = ab_grad[RAW_low_ids]

    # calculate the norm vector
    norm = np.array(df["SVC"]) - np.array(df["IVC"])
    # normalize norm
    n = np.linalg.norm(norm)
    norm_1 = norm / n

    plane = initialize_plane(norm_1, IVC_SEPT_CT_pt)

    septal_surf = get_elements_above_plane(no_TV_s, plane)

    if args.debug:
        Method.writer_vtk(septal_surf, f'{args.mesh}_surf/' + "septal_surf_2.vtk")

    CS_ids = vtk_to_numpy(septal_surf.GetCellData().GetArray('Global_ids'))

    # if len(CS_ids) == 0:
    ring_ids = np.loadtxt(f'{args.mesh}_surf/' + 'ids_CS.vtx', skiprows=2, dtype=int)

    rings_pts = vtk_to_numpy(model.GetPoints().GetData())[ring_ids, :]

    CS_ids = Method.get_element_ids_around_path_within_radius(model, rings_pts, 4 * args.scale)

    tag[CS_ids] = coronary_sinus
    k[CS_ids] = ab_grad[CS_ids]

    # tag = Method.assign_ra_appendage(model, SVC_s, np.array(df["RAA"]), tag, right_atrial_appendage_epi)
    RAA_s = vtk_thr(no_TV_s, 0, "CELLS", "phie_v2", tao_RAA)

    if args.debug:
        Method.writer_vtk(RAS_low, f'{args.mesh}_surf/' + "ras_low.vtk")
        Method.writer_vtk(RAA_s, f'{args.mesh}_surf/' + "raa_s.vtk")  # Check here if RAA is correctly tagged
        Method.writer_vtk(RAW_low, f'{args.mesh}_surf/' + "raw_low.vtk")

    RAA_ids = vtk_to_numpy(RAA_s.GetCellData().GetArray('Global_ids'))

    tag[RAA_ids] = right_atrial_appendage_epi

    RAA_CT_pt = CT_band.GetPoint(find_closest_point(CT_band, np.array(df["RAA"])))

    CT = CT_band

    CT_ids = vtk_to_numpy(CT.GetCellData().GetArray('Global_ids'))

    CT_ids = np.setdiff1d(CT_ids, RAA_ids, assume_unique=True)

    tag[CT_ids] = crista_terminalis

    k[CT_ids] = w_grad[CT_ids]

    tag[CT_SEPT_ids] = crista_terminalis

    SVC_ids = vtk_to_numpy(SVC_s.GetCellData().GetArray('Global_ids'))

    tag[SVC_ids] = superior_vena_cava_epi

    k[SVC_ids] = v_grad[SVC_ids]

    IVC_ids = vtk_to_numpy(IVC_s.GetCellData().GetArray('Global_ids'))

    tag[IVC_ids] = inferior_vena_cava_epi

    k[IVC_ids] = v_grad[IVC_ids]

    tag = np.where(tag == 0, right_atrial_lateral_wall_epi, tag)

    SN_ids = Method.get_element_ids_around_path_within_radius(no_SVC_s, np.asarray([SVC_CT_pt]), r_SN)

    tag[SN_ids] = sinus_node

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
    # k = ab_grad
    print('############### k ###############')

    en = k - et * np.sum(k * et, axis=1).reshape(len(et), 1)

    abs_en = np.linalg.norm(en, axis=1, keepdims=True)
    abs_en = np.where(abs_en != 0, abs_en, 1)
    en = en / abs_en
    print('############### en ###############')
    # el
    el = np.cross(en, et)

    el = Method.assign_element_fiber_around_path_within_radius(model, CT_SEPT_path, w_ct, el, smooth=True)

    el = np.where(el == [0, 0, 0], [1, 0, 0], el).astype("float32")

    print('############### el ###############')
    # print(el)
    end_time = datetime.datetime.now()
    running_time = end_time - start_time
    print('Calculating epicardial fibers... done! ' + str(end_time) + '\nIt takes: ' + str(running_time) + '\n')

    if args.mesh_type == "bilayer":
        sheet = np.cross(el, et)

        for i in range(model.GetPointData().GetNumberOfArrays() - 1, -1, -1):
            model.GetPointData().RemoveArray(model.GetPointData().GetArrayName(i))

        for i in range(model.GetCellData().GetNumberOfArrays() - 1, -1, -1):
            model.GetCellData().RemoveArray(model.GetCellData().GetArrayName(i))

        meshNew = dsa.WrapDataObject(model)
        meshNew.CellData.append(tag, "elemTag")
        meshNew.CellData.append(el, "fiber")
        meshNew.CellData.append(sheet, "sheet")
        if args.ofmt == 'vtk':
            vtk_unstructured_grid_writer(job.ID + "/result_RA/RA_epi_with_fiber.vtk", meshNew.VTKObject,
                                         store_binary=True)
        else:
            vtk_xml_unstructured_grid_writer(job.ID + "/result_RA/RA_epi_with_fiber.vtu", meshNew.VTKObject)
        """
        PM and CT
        """

        model = generate_ids(meshNew.VTKObject, "Global_ids", "Global_ids")

        endo = vtk.vtkUnstructuredGrid()
        endo.DeepCopy(model)

        CT = vtk_thr(model, 2, "CELLS", "elemTag", crista_terminalis, crista_terminalis)

        CT_ids = vtk_to_numpy(CT.GetCellData().GetArray('Global_ids'))

    elif args.mesh_type == "vol":

        CT_id_list = vtk.vtkIdList()
        for var in CT_ids:
            CT_id_list.InsertNextId(var)
        for var in CT_SEPT_ids:
            CT_id_list.InsertNextId(var)

        CT = apply_extract_cell_filter(model, CT_id_list)

        CT_ids = vtk_to_numpy(CT.GetCellData().GetArray('Global_ids'))

        if args.debug:

            meshNew = dsa.WrapDataObject(model)
            meshNew.CellData.append(tag, "elemTag")
            meshNew.CellData.append(el, "fiber")

            writer = vtk.vtkUnstructuredGridWriter()
            if args.ofmt == 'vtk':
                vtk_unstructured_grid_writer(job.ID + "/result_RA/RA_epi_with_fiber.vtk", meshNew.VTKObject,
                                             store_binary=True)
            else:
                vtk_xml_unstructured_grid_writer(job.ID + "/result_RA/RA_epi_with_fiber.vtu", meshNew.VTKObject)
    center = np.asarray((np.array(df["SVC"]) + np.array(df["IVC"])) / 2)

    point1_id = find_closest_point(CT, IVC_max_r_CT_pt)
    point2_id = find_closest_point(CT, SVC_CT_pt)

    point3_id = find_closest_point(TV_s, IVC_max_r_CT_pt)
    point4_id = find_closest_point(TV_s,
                                   np.array(df["RAA"]))  # this is also the id for Bachmann-Bundle on the right atrium

    CT = apply_vtk_geom_filter(CT)

    # calculate the norm vector
    v1 = np.array(df["IVC"]) - np.array(df["SVC"])
    v2 = np.array(df["TV"]) - np.array(df["IVC"])
    norm = np.cross(v2, v1)

    # normalize norm
    n = np.linalg.norm(norm)
    norm_1 = norm / n

    initialize_plane(norm_1, df["TV"])

    TV_lat = get_elements_above_plane(TV_s, plane)

    TV_lat = apply_vtk_geom_filter(TV_lat)

    TV_lat = clean_polydata(TV_lat)

    if args.debug:
        Method.writer_vtk(TV_lat, f'{args.mesh}_surf/' + "TV_lat.vtk")

    ct_points_data = Method.dijkstra_path(CT, point1_id, point2_id)

    if args.debug:
        Method.create_pts(ct_points_data, 'ct_points_data', f'{args.mesh}_surf/')

    point3_id = find_closest_point(TV_lat, TV_s.GetPoint(point3_id))

    # this is also the id for Bachmann-Bundle on the right atrium
    point4_id = find_closest_point(TV_lat, TV_s.GetPoint(point4_id))

    tv_points_data = Method.dijkstra_path(TV_lat, point3_id, point4_id)

    if args.debug:
        Method.create_pts(tv_points_data, 'tv_points_data', f'{args.mesh}_surf/')

    print("Creating Pectinate muscle...")

    if args.mesh_type == "vol":

        # Copy current tags and fibers, they will be overwritten by the PMs
        tag_old = np.array(tag, dtype=int)
        el_old = np.array(el)

        surface = apply_vtk_geom_filter(model)

        epi = vtk_thr(surface, 0, "POINTS", "phie_phi", 0.5)

        endo = vtk_thr(surface, 1, "POINTS", "phie_phi", 0.5)

        surface = apply_vtk_geom_filter(endo)


    elif args.mesh_type == "bilayer":
        fiber_endo = el.copy()
        tag_endo = np.copy(tag)
        surface = apply_vtk_geom_filter(endo)

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

    # Pectinate muscle
    print("Creating Pectinate muscle 1")
    # the first PM is the one to the appendage
    # pm = Method.dijkstra_path(surface, pm_ct_id_list[-1], point_apx_id)

    RAA_CT_id = find_closest_point(surface, RAA_CT_pt)

    # Pectinate muscle going from the septum spurius to the RAA apex
    pm = Method.dijkstra_path(surface, RAA_CT_id, point_apx_id)

    pm = Method.downsample_path(pm, int(len(pm) * 0.1))
    if args.debug:
        Method.create_pts(pm, 'pm_0_downsampled', f'{args.mesh}_surf/')

    if args.mesh_type == "bilayer":

        tag_endo = Method.assign_element_tag_around_path_within_radius(endo, pm, w_pm, tag_endo, pectinate_muscle)
        fiber_endo = Method.assign_element_fiber_around_path_within_radius(endo, pm, w_pm, fiber_endo, smooth=False)

    elif args.mesh_type == "vol":

        filter_cell_centers = vtk.vtkCellCenters()
        filter_cell_centers.SetInputData(model)
        filter_cell_centers.Update()
        centroids_array = vtk_to_numpy(filter_cell_centers.GetOutput().GetPoints().GetData())

        tree = cKDTree(centroids_array)

        ii = tree.query_ball_point(pm, r=w_pm, n_jobs=-1)

        ii = list(set([item for sublist in ii for item in sublist]))

        tag[ii] = pectinate_muscle

        el = Method.assign_element_fiber_around_path_within_radius(model, pm, w_pm, el, smooth=False)

    for i in range(3, pm_num - 1):  # skip the first 3 pm as they will be in the IVC side
        pm_point_1 = pm_ct_id_list[(i + 1) * pm_ct_dis]
        pm_point_2 = pm_tv_id_list[(i + 1) * pm_tv_dis]
        pm = Method.dijkstra_path_on_a_plane(surface, args, pm_point_1, pm_point_2, center)

        # skip first 3% of the points since they will be on the roof of the RA
        pm = pm[int(len(pm) * 0.03):, :]  #

        pm = Method.downsample_path(pm, int(len(pm) * 0.065))

        if args.debug:
            Method.create_pts(pm, 'pm_' + str(i + 1) + '_downsampled', f'{args.mesh}_surf/')

        print("The ", i + 1, "th pm done")
        if args.mesh_type == "bilayer":

            tag_endo = Method.assign_element_tag_around_path_within_radius(endo, pm, w_pm, tag_endo, pectinate_muscle)
            print("The ", i + 1, "th pm's tag is done")
            fiber_endo = Method.assign_element_fiber_around_path_within_radius(endo, pm, w_pm, fiber_endo, smooth=False)
            print("The ", i + 1, "th pm's fiber is done")

        elif args.mesh_type == "vol":

            ii = tree.query_ball_point(pm, r=w_pm, n_jobs=-1)

            ii = list(set([item for sublist in ii for item in sublist]))

            tag[ii] = pectinate_muscle

            el = Method.assign_element_fiber_around_path_within_radius(model, pm, w_pm, el, smooth=False)

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

        for i in range(endo.GetPointData().GetNumberOfArrays() - 1, -1, -1):
            endo.GetPointData().RemoveArray(endo.GetPointData().GetArrayName(i))

        for i in range(endo.GetCellData().GetNumberOfArrays() - 1, -1, -1):
            endo.GetCellData().RemoveArray(endo.GetCellData().GetArrayName(i))

        fiber_endo = np.where(fiber_endo == [0, 0, 0], [1, 0, 0], fiber_endo).astype("float32")
        sheet = np.cross(fiber_endo, et)
        sheet = np.where(sheet == [0, 0, 0], [1, 0, 0], sheet).astype("float32")
        meshNew = dsa.WrapDataObject(endo)
        meshNew.CellData.append(tag_endo, "elemTag")
        meshNew.CellData.append(fiber_endo, "fiber")
        meshNew.CellData.append(sheet, "sheet")

        endo = meshNew.VTKObject

        if args.ofmt == 'vtk':

            vtk_unstructured_grid_writer(job.ID + "/result_RA/RA_endo_with_fiber.vtk", endo, store_binary=True)

        else:

            vtk_xml_unstructured_grid_writer(job.ID + "/result_RA/RA_endo_with_fiber.vtu", endo)

        CT_PMs = vtk_thr(endo, 2, "CELLS", "elemTag", pectinate_muscle, crista_terminalis)

        if args.ofmt == 'vtk':
            vtk_unstructured_grid_writer(job.ID + "/result_RA/RA_CT_PMs.vtk", CT_PMs, store_binary=True)
        else:
            vtk_xml_unstructured_grid_writer(job.ID + "/result_RA/RA_CT_PMs.vtu", CT_PMs)

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

            for i in range(model.GetPointData().GetNumberOfArrays() - 1, -1, -1):
                model.GetPointData().RemoveArray(model.GetPointData().GetArrayName(i))

            for i in range(model.GetCellData().GetNumberOfArrays() - 1, -1, -1):
                model.GetCellData().RemoveArray(model.GetCellData().GetArrayName(i))

            el = np.where(el == [0, 0, 0], [1, 0, 0], el).astype("float32")
            sheet = np.cross(el, et)
            sheet = np.where(sheet == [0, 0, 0], [1, 0, 0], sheet).astype("float32")
            meshNew = dsa.WrapDataObject(model)
            meshNew.CellData.append(tag, "elemTag")
            meshNew.CellData.append(el, "fiber")
            meshNew.CellData.append(sheet, "sheet")

            writer = vtk.vtkUnstructuredGridWriter()
            if args.ofmt == 'vtk':
                vtk_unstructured_grid_writer(job.ID + "/result_RA/RA_endo_with_fiber.vtk", meshNew.VTKObject,
                                             store_binary=True)
            else:
                vtk_xml_unstructured_grid_writer(job.ID + "/result_RA/RA_endo_with_fiber.vtu", meshNew.VTKObject)

    # Bachmann-Bundle
    if args.mesh_type == "vol":
        surface = apply_vtk_geom_filter(epi)

    bb_c_id = find_closest_point(RAS_S, (np.array(df["TV"]) + np.array(df["SVC"])) / 2)

    # Bachmann-Bundle starting point
    bb_1_id = find_closest_point(surface, SVC_CT_pt)
    bb_2_id = find_closest_point(surface, TV_lat.GetPoint(point4_id))
    bb_c_id = find_closest_point(surface, RAS_S.GetPoint(bb_c_id))

    bachmann_bundle_points_data_1 = Method.dijkstra_path(surface, bb_1_id, bb_c_id)

    bachmann_bundle_points_data_2 = Method.dijkstra_path(surface, bb_c_id, bb_2_id)

    bachmann_bundle_points_data = np.concatenate((bachmann_bundle_points_data_1, bachmann_bundle_points_data_2), axis=0)

    np.savetxt(job.ID + '/bb.txt', bachmann_bundle_points_data, fmt='%.5f')  # Change directory

    bb_step = int(len(bachmann_bundle_points_data) * 0.1)
    bb_path = np.asarray([bachmann_bundle_points_data[i] for i in range(len(bachmann_bundle_points_data)) if
                          i % bb_step == 0 or i == len(bachmann_bundle_points_data) - 1])
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

    bb_points = vtk_to_numpy(functionSource.GetOutput().GetPoints().GetData())

    tag = Method.assign_element_tag_around_path_within_radius(model, bb_points, w_bb, tag, bachmann_bundel_right)
    el = Method.assign_element_fiber_around_path_within_radius(model, bb_points, w_bb, el, smooth=True)

    tag[SN_ids] = sinus_node

    for i in range(model.GetPointData().GetNumberOfArrays() - 1, -1, -1):
        model.GetPointData().RemoveArray(model.GetPointData().GetArrayName(i))

    for i in range(model.GetCellData().GetNumberOfArrays() - 1, -1, -1):
        model.GetCellData().RemoveArray(model.GetCellData().GetArrayName(i))

    el = np.where(el == [0, 0, 0], [1, 0, 0], el).astype("float32")
    sheet = np.cross(el, et)
    sheet = np.where(sheet == [0, 0, 0], [1, 0, 0], sheet).astype("float32")
    meshNew = dsa.WrapDataObject(model)
    meshNew.CellData.append(tag, "elemTag")
    meshNew.CellData.append(el, "fiber")
    meshNew.CellData.append(sheet, "sheet")

    if args.mesh_type == "bilayer":
        if args.ofmt == 'vtk':
            vtk_unstructured_grid_writer(job.ID + "/result_RA/RA_epi_with_fiber.vtk", meshNew.VTKObject, True)
        else:
            vtk_xml_unstructured_grid_writer(job.ID + "/result_RA/RA_epi_with_fiber.vtu", meshNew.VTKObject)
    else:
        if args.ofmt == 'vtk':
            vtk_unstructured_grid_writer(job.ID + "/result_RA/RA_vol_with_fiber.vtk", meshNew.VTKObject,
                                         store_binary=True)
        else:
            vtk_xml_unstructured_grid_writer(job.ID + "/result_RA/RA_vol_with_fiber.vtu", meshNew.VTKObject)

    model = meshNew.VTKObject

    if args.add_bridges:
        # Bachmann_Bundle internal connection

        if args.mesh_type == "bilayer":
            if args.ofmt == 'vtk':
                la_epi = Method.smart_reader(job.ID + "/result_LA/LA_epi_with_fiber.vtk")
            else:
                la_epi = Method.smart_reader(job.ID + "/result_LA/LA_epi_with_fiber.vtu")

        elif args.mesh_type == "vol":
            # extension = args.mesh.split('_RA_vol')[-1]
            meshname = args.mesh[:-7]

            if args.ofmt == 'vtk':
                la = Method.smart_reader(meshname + "_LA_vol_fibers/result_LA/LA_vol_with_fiber.vtk")
            else:
                la = Method.smart_reader(meshname + "_LA_vol_fibers/result_LA/LA_vol_with_fiber.vtu")

            la_surf = apply_vtk_geom_filter(la)

            la_epi = vtk_thr(la_surf, 2, "CELLS", "elemTag", left_atrial_wall_epi, 99)

            df = pd.read_csv(meshname + "_LA_vol_surf/rings_centroids.csv")

        la_appendage_basis_point = np.asarray(df["LAA_basis_inf"])
        length = len(bachmann_bundle_points_data)
        ra_bb_center = bachmann_bundle_points_data[int(length * 0.45)]

        la_epi = apply_vtk_geom_filter(la_epi)

        if args.mesh_type == "bilayer":
            ra_epi = apply_vtk_geom_filter(model)

        else:
            ra_epi = surface

        ra_a_id = find_closest_point(ra_epi, ra_bb_center)
        la_c_id = find_closest_point(la_epi, ra_bb_center)
        ra_b_id = find_closest_point(ra_epi, la_epi.GetPoint(la_c_id))
        la_d_id = find_closest_point(la_epi, la_appendage_basis_point)

        path_1 = Method.dijkstra_path(ra_epi, ra_a_id, ra_b_id)
        path_2 = Method.dijkstra_path(la_epi, la_c_id, la_d_id)
        path_all_temp = np.vstack((path_1, path_2))
        # down sampling to smooth the path
        step = 20
        # step = int(len(path_all_temp)*0.1)
        path_all = np.asarray(
            [path_all_temp[i] for i in range(len(path_all_temp)) if i % step == 0 or i == len(path_all_temp) - 1])

        # save points for bb fiber
        filename = job.ID + '/bridges/bb_fiber.dat'
        f = open(filename, 'wb')
        pickle.dump(path_all, f)
        f.close()

        # BB tube

        bb_tube = Method.creat_tube_around_spline(path_all, 2 * args.scale)
        sphere_a = Method.creat_sphere(la_appendage_basis_point, 2 * 1.02 * args.scale)
        sphere_b = Method.creat_sphere(ra_bb_center, 2 * 1.02 * args.scale)
        Method.smart_bridge_writer(bb_tube, sphere_a, sphere_b, "BB_intern_bridges", job)

        df = pd.read_csv(args.mesh + "_surf/rings_centroids.csv")

        try:
            CS_p = np.array(df["CS"])
        except KeyError:
            CS_p = IVC_SEPT_CT_pt
            print("No CS found, use last CT point instead")

        if args.mesh_type == "bilayer":
            add_free_bridge(args, la_epi, model, CS_p, df, job)
        elif args.mesh_type == "vol":
            add_free_bridge(args, la, model, CS_p, df, job)
