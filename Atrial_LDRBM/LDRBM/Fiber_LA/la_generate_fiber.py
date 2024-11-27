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

import numpy as np
import pandas as pd
import vtk
from vtk.numpy_interface import dataset_adapter as dsa

import Atrial_LDRBM.LDRBM.Fiber_LA.Methods_LA as Method
from Atrial_LDRBM.LDRBM.Fiber_LA.la_laplace import laplace_0_1
from vtk_opencarp_helper_methods.AugmentA_methods.vtk_operations import vtk_thr
from vtk_opencarp_helper_methods.openCARP.exporting import write_to_pts, write_to_elem, write_to_lon
from vtk_opencarp_helper_methods.vtk_methods.converters import vtk_to_numpy
from vtk_opencarp_helper_methods.vtk_methods.exporting import vtk_unstructured_grid_writer, \
    vtk_xml_unstructured_grid_writer
from vtk_opencarp_helper_methods.vtk_methods.filters import apply_vtk_geom_filter, generate_ids, \
    get_elements_above_plane
from vtk_opencarp_helper_methods.vtk_methods.init_objects import initialize_plane_with_points

EXAMPLE_DIR = os.path.dirname(os.path.realpath(__file__))

vtk_version = vtk.vtkVersion.GetVTKSourceVersion().split()[-1].split('.')[0]


def la_generate_fiber(model, args, job):
    # size(Radius) of Bachmann Bundle in mm
    w_bb = 2 * args.scale

    simid = job.ID + "/result_LA"
    try:
        os.makedirs(simid)
    except OSError:
        print(f"Creation of the directory {simid} failed")
    else:
        print(f"Successfully created the directory {simid} ")

    with open(os.path.join(EXAMPLE_DIR, '../../element_tag.csv')) as f:
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

    # r2
    r2 = model.GetCellData().GetArray('phie_r2')
    r2 = vtk_to_numpy(r2)

    # phie
    if args.mesh_type == "vol":
        phie = model.GetCellData().GetArray('phie_phi')
        phie = vtk_to_numpy(phie)

    phie_grad = model.GetCellData().GetArray('grad_phi')
    phie_grad = vtk_to_numpy(phie_grad)

    model = generate_ids(model, "Global_ids", "Global_ids")

    df = pd.read_csv(args.mesh + "_surf/rings_centroids.csv")

    # LPV
    lb = 0
    ub = 0.4
    tao_lpv = Method.find_tau(model, ub, lb, "low", "phie_v")
    print('Calculating tao_lpv done! tap_lpv = ', tao_lpv)

    thr = vtk_thr(model, 1, "CELLS", "phie_v", tao_lpv)
    connect = vtk.vtkConnectivityFilter()
    connect.SetInputData(thr)
    connect.SetExtractionModeToAllRegions()
    connect.Update()

    PVs = dict()
    # Distinguish between LIPV and LSPV
    PVs = Method.distinguish_PVs(connect, PVs, df, "LIPV", "LSPV")

    model = laplace_0_1(args, job, model, "RPV", "LAA", "phie_ab2")

    thr = vtk_thr(model, 1, "CELLS", "phie_v", tao_lpv)

    phie_r2_tau_lpv = vtk_to_numpy(thr.GetCellData().GetArray('phie_r2'))
    max_phie_r2_tau_lpv = np.max(phie_r2_tau_lpv)

    phie_ab_tau_lpv = vtk_to_numpy(thr.GetPointData().GetArray('phie_ab2'))
    max_phie_ab_tau_lpv = np.max(phie_ab_tau_lpv)

    print("max_phie_r2_tau_lpv ", max_phie_r2_tau_lpv)
    print("max_phie_ab_tau_lpv ", max_phie_ab_tau_lpv)

    # RPV
    lb = 0.6
    ub = 1
    tao_rpv = Method.find_tau(model, ub, lb, "up", "phie_v")
    print('Calculating tao_rpv done! tap_rpv = ', tao_rpv)

    thr = vtk_thr(model, 0, "CELLS", "phie_v", tao_rpv)

    connect = vtk.vtkConnectivityFilter()
    connect.SetInputData(thr)
    connect.SetExtractionModeToAllRegions()
    connect.Update()

    # Distinguish between RIPV and RSPV
    PVs = Method.distinguish_PVs(connect, PVs, df, "RIPV", "RSPV")

    start_time = datetime.datetime.now()
    print('Calculating fibers... ' + str(start_time))

    # Bilayer mesh
    if args.mesh_type == 'bilayer':
        epi = vtk.vtkUnstructuredGrid()
        epi.DeepCopy(model)

        tag_epi = np.zeros(len(r), dtype=int)
        tag_epi[:] = left_atrial_wall_epi
        tag_endo = np.zeros(len(r), dtype=int)
        tag_endo[:] = left_atrial_wall_endo

    else:  # Volume mesh
        tag = np.ones(len(r), dtype=int) * left_atrial_wall_epi

        epi = vtk_thr(model, 0, "CELLS", "phie_phi", 0.5)

        epi_ids = vtk_to_numpy(epi.GetCellData().GetArray('Global_ids'))

        endo_ids = np.arange(len(r)).astype(int)

        endo_ids = np.setdiff1d(endo_ids, epi_ids)

        tag[endo_ids] = left_atrial_wall_endo

    LAA_s = vtk_thr(model, 0, "CELLS", "phie_r2", max_phie_r2_tau_lpv + 0.01)

    LAA_s = vtk_thr(LAA_s, 0, "POINTS", "phie_ab2", max_phie_ab_tau_lpv - 0.03)

    ## Optimize shape of LAA solving a laplacian with 0 in LAA and 1 in the boundary of LAA_s

    LAA_bb = vtk_thr(model, 2, "POINTS", "phie_ab2", max_phie_ab_tau_lpv - 0.03, max_phie_ab_tau_lpv + 0.01)

    LAA_bb_ids = vtk_to_numpy(LAA_bb.GetPointData().GetArray('Global_ids'))

    MV_ring_ids = np.loadtxt(f'{args.mesh}_surf/ids_MV.vtx', skiprows=2, dtype=int)

    LAA_bb_ids = np.append(LAA_bb_ids, MV_ring_ids)

    fname = f'{args.mesh}_surf/ids_LAA_bb.vtx'
    f = open(fname, 'w')
    f.write(f'{len(LAA_bb_ids)}\n')
    f.write('extra\n')
    for i in LAA_bb_ids:
        f.write(f'{i}\n')
    f.close()

    LAA_s = laplace_0_1(args, job, model, "LAA", "LAA_bb", "phie_ab3")

    LAA_s = vtk_thr(LAA_s, 1, "POINTS", "phie_ab3", 0.95)

    ring_ids = np.loadtxt(f'{args.mesh}_surf/' + 'ids_MV.vtx', skiprows=2, dtype=int)

    rings_pts = vtk_to_numpy(model.GetPoints().GetData())[ring_ids, :]

    MV_ids = Method.get_element_ids_around_path_within_radius(model, rings_pts, 4 * args.scale)

    LAA_ids = vtk_to_numpy(LAA_s.GetCellData().GetArray('Global_ids'))

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

    lpv_mean = np.mean([df["LIPV"].to_numpy(), df["LSPV"].to_numpy()], axis=0)
    rpv_mean = np.mean([df["RIPV"].to_numpy(), df["RSPV"].to_numpy()], axis=0)
    mv_mean = df["MV"].to_numpy()

    plane = initialize_plane_with_points(mv_mean, rpv_mean, lpv_mean, mv_mean)

    band_s = vtk_thr(epi, 0, "CELLS", "phie_r2", max_phie_r2_tau_lpv)

    extracted_mesh = get_elements_above_plane(band_s, plane)

    band_cell_ids = vtk_to_numpy(extracted_mesh.GetCellData().GetArray('Global_ids'))

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

        en_endo = k_endo - et * np.sum(k_endo * et, axis=1).reshape(len(et), 1)
        en_epi = k_epi - et * np.sum(k_epi * et, axis=1).reshape(len(et), 1)

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

        end_time = datetime.datetime.now()

        el_endo = np.where(el_endo == [0, 0, 0], [1, 0, 0], el_endo).astype("float32")

        sheet_endo = np.cross(el_endo, et)
        sheet_endo = np.where(sheet_endo == [0, 0, 0], [1, 0, 0], sheet_endo).astype("float32")

        for i in range(endo.GetPointData().GetNumberOfArrays() - 1, -1, -1):
            endo.GetPointData().RemoveArray(endo.GetPointData().GetArrayName(i))

        for i in range(endo.GetCellData().GetNumberOfArrays() - 1, -1, -1):
            endo.GetCellData().RemoveArray(endo.GetCellData().GetArrayName(i))

        meshNew = dsa.WrapDataObject(endo)
        meshNew.CellData.append(tag_endo, "elemTag")
        meshNew.CellData.append(el_endo, "fiber")
        meshNew.CellData.append(sheet_endo, "sheet")
        if args.ofmt == 'vtk':
            vtk_unstructured_grid_writer(job.ID + "/result_LA/LA_endo_with_fiber.vtk", meshNew.VTKObject,
                                         store_binary=True)
        else:
            vtk_xml_unstructured_grid_writer(job.ID + "/result_LA/LA_endo_with_fiber.vtu", meshNew.VTKObject)
        pts = vtk_to_numpy(endo.GetPoints().GetData())
        write_to_pts(job.ID + '/result_LA/LA_endo_with_fiber.pts', pts)

        write_to_elem(job.ID + '/result_LA/LA_endo_with_fiber.elem', endo, tag_endo)

        write_to_lon(job.ID + '/result_LA/LA_endo_with_fiber.lon', el_endo, sheet_endo)

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

        en = k - et * np.sum(k * et, axis=1).reshape(len(et), 1)

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

        end_time = datetime.datetime.now()

        meshNew = dsa.WrapDataObject(model)
        meshNew.CellData.append(tag, "elemTag")
        meshNew.CellData.append(el, "fiber")
        model = meshNew.VTKObject

        running_time = end_time - start_time

        print('Calculating fibers... done! ' + str(end_time) + '\nIt takes: ' + str(running_time) + '\n')

    print("Creating bachmann bundles...")
    # Bachmann Bundle

    if args.mesh_type == "vol":  # Extract epicardial surface
        surf = apply_vtk_geom_filter(model)

        epi_surf = vtk_thr(surf, 0, "CELLS", "phie_phi", 0.5)
        epi_surf_ids = vtk_to_numpy(epi_surf.GetCellData().GetArray('Global_ids'))

        epi_surf = apply_vtk_geom_filter(epi_surf)

    if args.mesh_type == "bilayer":
        bb_left, LAA_basis_inf, LAA_basis_sup, LAA_far_from_LIPV = Method.compute_wide_BB_path_left(epi, df,
                                                                                                    left_atrial_appendage_epi,
                                                                                                    mitral_valve_epi)
        tag_epi = Method.assign_element_tag_around_path_within_radius(epi, bb_left, w_bb, tag_epi, bachmann_bundel_left)
        el_epi = Method.assign_element_fiber_around_path_within_radius(epi, bb_left, w_bb, el_epi, smooth=True)
    else:
        bb_left, LAA_basis_inf, LAA_basis_sup, LAA_far_from_LIPV = Method.compute_wide_BB_path_left(epi_surf, df,
                                                                                                    left_atrial_appendage_epi,
                                                                                                    mitral_valve_epi)
        tag[epi_surf_ids] = Method.assign_element_tag_around_path_within_radius(epi_surf, bb_left, w_bb,
                                                                                vtk_to_numpy(
                                                                                    epi_surf.GetCellData().GetArray(
                                                                                        'elemTag')),
                                                                                bachmann_bundel_left)
        el[epi_surf_ids] = Method.assign_element_fiber_around_path_within_radius(epi_surf, bb_left, w_bb,
                                                                                 vtk_to_numpy(
                                                                                     epi_surf.GetCellData().GetArray(
                                                                                         'fiber')), smooth=True)

    df["LAA_basis_inf"] = LAA_basis_inf
    df["LAA_basis_sup"] = LAA_basis_sup
    df["LAA_far_from_LIPV"] = LAA_far_from_LIPV

    df.to_csv(args.mesh + "_surf/rings_centroids.csv", float_format="%.2f", index=False)
    print("Creating bachmann bundles... done")
    if args.mesh_type == "bilayer":
        el_epi = np.where(el_epi == [0, 0, 0], [1, 0, 0], el_epi).astype("float32")
        #### save the result into vtk ####
        start_time = datetime.datetime.now()
        print('Writinga as LA_with_fiber... ' + str(start_time))

        sheet_epi = np.cross(el_epi, et)
        sheet_epi = np.where(sheet_epi == [0, 0, 0], [1, 0, 0], sheet_epi).astype("float32")

        for i in range(epi.GetPointData().GetNumberOfArrays() - 1, -1, -1):
            epi.GetPointData().RemoveArray(epi.GetPointData().GetArrayName(i))

        for i in range(epi.GetCellData().GetNumberOfArrays() - 1, -1, -1):
            epi.GetCellData().RemoveArray(epi.GetCellData().GetArrayName(i))

        meshNew = dsa.WrapDataObject(epi)
        meshNew.CellData.append(tag_epi, "elemTag")
        meshNew.CellData.append(el_epi, "fiber")
        meshNew.CellData.append(sheet_epi, "sheet")
        if args.ofmt == 'vtk':
            vtk_unstructured_grid_writer(job.ID + "/result_LA/LA_epi_with_fiber.vtk", meshNew.VTKObject,
                                         store_binary=True)
        else:
            vtk_xml_unstructured_grid_writer(job.ID + "/result_LA/LA_epi_with_fiber.vtu", meshNew.VTKObject)
        pts = vtk_to_numpy(epi.GetPoints().GetData())

        write_to_pts(job.ID + '/result_LA/LA_epi_with_fiber.pts', pts)

        write_to_elem(job.ID + '/result_LA/LA_epi_with_fiber.elem', epi, tag_epi)

        write_to_lon(job.ID + '/result_LA/LA_epi_with_fiber.lon', el_epi, sheet_epi)

        end_time = datetime.datetime.now()
        running_time = end_time - start_time
        print('Writing as LA_with_fiber... done! ' + str(end_time) + '\nIt takes: ' + str(running_time) + '\n')

        endo = Method.move_surf_along_normals(endo, 0.1 * args.scale,
                                              1)  # Warning: set -1 if pts normals are pointing outside
        bilayer = Method.generate_bilayer(endo, epi)

        Method.write_bilayer(bilayer, args, job)

    else:
        el = np.where(el == [0, 0, 0], [1, 0, 0], el).astype("float32")

        #### save the result into vtk ####
        start_time = datetime.datetime.now()
        print('Writinga as LA_with_fiber... ' + str(start_time))

        sheet = np.cross(el, et)
        sheet = np.where(sheet == [0, 0, 0], [1, 0, 0], sheet).astype("float32")

        for i in range(model.GetPointData().GetNumberOfArrays() - 1, -1, -1):
            model.GetPointData().RemoveArray(model.GetPointData().GetArrayName(i))

        for i in range(model.GetCellData().GetNumberOfArrays() - 1, -1, -1):
            model.GetCellData().RemoveArray(model.GetCellData().GetArrayName(i))

        meshNew = dsa.WrapDataObject(model)
        meshNew.CellData.append(tag, "elemTag")
        meshNew.CellData.append(el, "fiber")
        meshNew.CellData.append(sheet, "sheet")
        filename = job.ID + "/result_LA/LA_vol_with_fiber"
        if args.ofmt == 'vtk':
            vtk_unstructured_grid_writer(filename + ".vtk", meshNew.VTKObject,
                                         store_binary=True)
        else:
            vtk_xml_unstructured_grid_writer(filename + ".vtu", meshNew.VTKObject)
        pts = vtk_to_numpy(model.GetPoints().GetData())

        write_to_pts(filename + '.pts', pts)
        write_to_elem(filename + '.elem', model, tag)
        write_to_lon(filename + '.lon', el, sheet)

        end_time = datetime.datetime.now()
        running_time = end_time - start_time
        print('Writing as LA_with_fiber... done! ' + str(end_time) + '\nIt takes: ' + str(running_time) + '\n')
