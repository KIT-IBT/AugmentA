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
from vtk_opencarp_helper_methods.openCARP.exporting import write_to_pts, write_to_elem, write_to_lon
from vtk_opencarp_helper_methods.vtk_methods.converters import vtk_to_numpy
from vtk_opencarp_helper_methods.vtk_methods.exporting import vtk_xml_unstructured_grid_writer

EXAMPLE_DESCRIPTIVE_NAME = 'Tune conductivities to fit clinical LAT map'
EXAMPLE_AUTHOR = 'Luca Azzolin <luca.azzolin@kit.edu>'

import os
import vtk

from datetime import date
from carputils import settings
from carputils import tools

import numpy as np
import csv
from vtk.numpy_interface import dataset_adapter as dsa
import Methods_fit_to_clinical_LAT

from sklearn.metrics import mean_squared_error

vtk_version = vtk.vtkVersion.GetVTKSourceVersion().split()[-1].split('.')[0]


def parser():
    # Generate the standard command line parser
    parser = tools.standard_parser()
    # Add arguments
    parser.add_argument('--giL', type=float, default=0.4,
                        help='intracellular longitudinal conductivity to fit CV=0.714 m/s with dx=0.3 mm and dt=20')
    parser.add_argument('--geL', type=float, default=1.1,
                        help='extracellular longitudinal conductivity to fit CV=0.714 m/s with dx=0.3 mm and dt=20')
    parser.add_argument('--model',
                        type=str,
                        default='COURTEMANCHE',
                        help='input ionic model')
    parser.add_argument('--low_vol_thr',
                        type=float,
                        default=0.5,
                        help='bipolar voltage threshold to define low voltage region')
    parser.add_argument('--low_CV_thr',
                        type=float,
                        default=300,
                        help='CV threshold to define low CV region in mm/s')
    parser.add_argument('--LaAT',
                        type=float,
                        default=134,
                        help='wanted last activation')
    parser.add_argument('--max_LAT_id',
                        type=int,
                        default=0,
                        help='wanted last activation')
    parser.add_argument('--fib_perc',
                        type=float,
                        default=0.3,
                        help='bipolar voltage threshold to define low voltage region')
    parser.add_argument('--tol',
                        type=float, default=1,
                        help='tolerance to optimize RMSE in [ms]')
    parser.add_argument('--max_LAT_pt',
                        type=str, default='max',
                        help='meshname')
    parser.add_argument('--step',
                        type=float,
                        default=20,
                        help='LAT band steps in ms')
    parser.add_argument('--thr',
                        type=int,
                        default=4,
                        help='LAT band steps in ms')
    parser.add_argument('--mesh',
                        type=str, default='',
                        help='meshname directory. Example: ../meshes/mesh_name/mesh_name')
    parser.add_argument('--results_dir',
                        type=str,
                        default='../results',
                        help='path to results folder')
    parser.add_argument('--init_state_dir',
                        type=str,
                        default='../data',
                        help='path to initialization state folder')
    parser.add_argument('--fibrotic_tissue',
                        type=int,
                        default=1,
                        help='set 1 for mesh with fibrotic tissue, 0 otherwise')
    parser.add_argument('--M_lump',
                        type=int,
                        default='1',
                        help='set 1 for mass lumping, 0 otherwise')
    parser.add_argument('--meth',
                        type=int,
                        default=0,
                        help='0 only low voltage, 1 scale vec, 2 low CV as 0.3 m/s')
    parser.add_argument('--SSM_fitting',
                        type=int,
                        default=0,
                        help='set to 1 to proceed with the fitting of a given SSM, 0 otherwise')
    parser.add_argument('--SSM_basename',
                        type=str,
                        default="mesh/meanshape",
                        help='statistical shape model basename')
    # ----------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--dt',
                        type=float, default=20.0,
                        help='[microsec]')

    parser.add_argument('--bcl',
                        type=float, default=500.0,
                        help='initial basic cycle lenght in [ms]')
    parser.add_argument('--stim_duration',
                        type=float, default=3.0,
                        help='stimulation duration in ms')
    parser.add_argument('--strength',
                        type=float, default=30,
                        help='stimulation transmembrane current in uA/cm^2 (2Dcurrent) or uA/cm^3 (3Dcurrent)')
    # Single cell prepace
    parser.add_argument('--cell_bcl',
                        type=float,
                        default=1000.0,
                        help='Specify the basic cycle length (ms) to initialize cells')
    parser.add_argument('--numstim',
                        type=int,
                        default=50,
                        help='Specify the number of single cell stimuli with the given bcl')

    parser.add_argument('--prebeats',
                        type=int,
                        default=4,
                        help='Number of beats to prepace the tissue')
    parser.add_argument('--debug',
                        type=int,
                        default=1,
                        help='set to 1 to debug step by step, 0 otherwise')
    # ----------------------------------------------------------------------------------------------------------------------------------------

    return parser


def jobID(args):
    today = date.today()
    mesh = args.mesh.split('/')[-1]
    ID = '{}/{}_converge_band_{}_prebeats_{}_bcl_{}_fib_{}_max_LAT_pt_{}_voltage_{}_CV_{}_meth_{}_fib_p_{}_step_{}_thr_{}'.format(
        args.results_dir, today.isoformat(), mesh,
        args.prebeats, args.bcl, args.fibrotic_tissue, args.max_LAT_pt, args.low_vol_thr, args.low_CV_thr, args.meth,
        args.fib_perc, args.step, args.thr)
    return ID


def single_cell_initialization(args, job, steady_state_dir, to_do):
    g_CaL_reg = [0.45, 0.7515, 0.7515, 0.3015, 0.3015, 0.4770, 0.4770, 0.45, 0.3375]
    g_K1_reg = [2, 2, 2, 2, 2, 2, 2, 2, 1.34]
    blf_g_Kur_reg = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    g_to_reg = [0.35, 0.35, 0.35, 0.35, 0.35, 0.2380, 0.2380, 0.35, 0.2625]
    g_Ks_reg = [2, 2, 2, 2, 2, 2, 2, 2, 3.74]
    maxI_pCa_reg = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
    maxI_NaCa_reg = [1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6]
    g_Kr_reg = [1, 1, 1, 1.53, 2.44, 1, 1.6, 1.6, 2.4]

    n_regions = len(g_CaL_reg)

    duration = args.numstim * args.cell_bcl
    for k in range(n_regions):
        init_file = steady_state_dir + '/init_values_stab_bcl_{}_reg_{}_numstim_{}.sv'.format(args.cell_bcl, k,
                                                                                              args.numstim)
        cmd = [settings.execs.BENCH,
               '--imp', args.model,
               '--imp-par',
               'g_CaL*{},g_K1*{},blf_i_Kur*{},g_to*{},g_Ks*{},maxI_pCa*{},maxI_NaCa*{},g_Kr*{}'.format(g_CaL_reg[k],
                                                                                                       g_K1_reg[k],
                                                                                                       blf_g_Kur_reg[k],
                                                                                                       g_to_reg[k],
                                                                                                       g_Ks_reg[k],
                                                                                                       maxI_pCa_reg[k],
                                                                                                       maxI_NaCa_reg[k],
                                                                                                       g_Kr_reg[k]),
               '--bcl', args.cell_bcl,
               '--dt-out', 1,  # Temporal granularity in ms (if  --dt-out= duration it saves only last value)
               '--stim-curr', 9.5,
               '--stim-dur', args.stim_duration,
               '--numstim', args.numstim,
               '--duration', duration,
               '--stim-start', 0,
               '--dt', args.dt / 1000,
               '--fout=' + steady_state_dir + f'/{args.numstim}_numstim_{args.cell_bcl}_bcl_ms_{k}',
               '-S', duration,
               '-F', init_file,
               '--trace-no', k]
        if to_do:
            job.bash(cmd)

    if args.fibrotic_tissue == 1:

        g_CaL_fib = [0.225]
        g_Na_fib = [0.6]
        blf_g_Kur_fib = [0.5]
        g_to_fib = [0.35]
        g_Ks_fib = [2]
        maxI_pCa_fib = [1.5]
        maxI_NaCa_fib = [1.6]

        n_regions += len(g_CaL_fib)

        for kk in range(len(g_CaL_fib)):
            init_file = steady_state_dir + '/init_values_stab_bcl_{}_reg_{}_numstim_{}.sv'.format(args.cell_bcl,
                                                                                                  k + 1 + kk,
                                                                                                  args.numstim)
            cmd = [settings.execs.BENCH,
                   '--imp', args.model,
                   '--imp-par',
                   'g_CaL*{},g_Na*{},blf_i_Kur*{},g_to*{},g_Ks*{},maxI_pCa*{},maxI_NaCa*{}'.format(g_CaL_fib[kk],
                                                                                                   g_Na_fib[kk],
                                                                                                   blf_g_Kur_fib[kk],
                                                                                                   g_to_fib[kk],
                                                                                                   g_Ks_fib[kk],
                                                                                                   maxI_pCa_fib[kk],
                                                                                                   maxI_NaCa_fib[kk]),
                   '--bcl', args.cell_bcl,
                   '--dt-out', 1,  # Temporal granularity in ms (if  --dt-out= duration it saves only last value)
                   '--stim-curr', 9.5,
                   '--stim-dur', args.stim_duration,
                   '--numstim', args.numstim,
                   '--duration', duration,
                   '--stim-start', 0,
                   '--dt', args.dt / 1000,
                   '--fout=' + steady_state_dir + '/{}_numstim_{}_bcl_ms_{}'.format(args.numstim, args.cell_bcl,
                                                                                    k + 1 + kk),
                   '-S', duration,
                   '-F', init_file,
                   '--trace-no', k + 1 + kk]
            if to_do:
                job.bash(cmd)

    tissue_init = []
    for k in range(n_regions):
        init_file = steady_state_dir + '/init_values_stab_bcl_{}_reg_{}_numstim_{}.sv'.format(args.cell_bcl, k,
                                                                                              args.numstim)
        tissue_init += [f'-imp_region[{k}].im_sv_init', init_file]

    if to_do:  # Move trace files to steady_state_dir
        for file in os.listdir(os.getcwd()):
            if file.startswith("Trace") or file.startswith(f"{args.model}_trace"):
                old_file = os.getcwd() + '/' + file
                new_file = steady_state_dir + '/' + file
                os.rename(old_file, new_file)

    return tissue_init


def remove_trash2(simid):
    for f in os.listdir(simid):
        if f.startswith("init_") or f.startswith("Trace_"):
            if os.path.isfile(os.path.join(simid, f)):
                os.remove(os.path.join(simid, f))


def tagregopt(reg, field, val):
    return ['-tagreg[' + str(reg) + '].' + field, val]


def tri_centroid(nodes, element):
    x1 = nodes[element[0], 0]
    x2 = nodes[element[1], 0]
    x3 = nodes[element[2], 0]

    y1 = nodes[element[0], 1]
    y2 = nodes[element[1], 1]
    y3 = nodes[element[2], 1]

    z1 = nodes[element[0], 2]
    z2 = nodes[element[1], 2]
    z3 = nodes[element[2], 2]

    return [(x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3, (z1 + z2 + z3) / 3]


@tools.carpexample(parser, jobID)
def run(args, job):
    # Polyfit of the CVs

    # p = np.poly1d([0.67278584, 0.17556362, 0.01718574])

    meshname = f'{args.mesh}_fibers/result_LA/LA_bilayer_with_fiber'  # um
    meshbasename = meshname.split('/')[-4]
    meshfold = f'{args.init_state_dir}/{meshbasename}'

    steady_state_dir = f'{args.init_state_dir}/{meshbasename}/cell_state'

    try:
        os.makedirs(steady_state_dir)
    except OSError:
        print(f"Creation of the directory {steady_state_dir} failed")
    else:
        print(f"Successfully created the directory {steady_state_dir} ")

    if not os.path.isfile(
            steady_state_dir + f'/init_values_stab_bcl_{args.cell_bcl}_reg_{0}_numstim_{args.numstim}.sv'):
        tissue_init = single_cell_initialization(args, job, steady_state_dir, 1)
    else:
        tissue_init = single_cell_initialization(args, job, steady_state_dir, 0)

    simid = job.ID

    try:
        os.makedirs(simid)
    except OSError:
        print(f"Creation of the directory {simid} failed")
    else:
        print(f"Successfully created the directory {simid} ")

    bilayer_n_cells, elements_in_fibrotic_reg, endo, endo_ids, centroids, LAT_map, min_LAT, el_to_clean, el_border, stim_pt, fit_LAT, healthy_endo = Methods_fit_to_clinical_LAT.low_vol_LAT(
        args, meshname + '_with_data_um.vtk')

    with open(f"{args.init_state_dir}/{meshbasename}/clinical_stim_pt.txt", "w") as f:
        f.write(f"{stim_pt[0]} {stim_pt[1]} {stim_pt[2]}")

    # Set to 1 every LAT <= 1
    LAT_map = np.where(LAT_map <= 1, 1, LAT_map)

    args.LaAT = args.LaAT - min_LAT

    # Set to max_LAT every LAT >= max_LAT
    LAT_map = np.where(LAT_map > args.LaAT, args.LaAT, LAT_map)

    print(f"Wanted LAT: {args.LaAT}")
    print(f"Max LAT point id: {args.max_LAT_id}")
    print(fit_LAT)

    # Find all not conductive elements belonging to the fibrotic tissue and not use them in the fitting
    tag = {}
    if not os.path.isfile(f'{meshfold}/elems_slow_conductive.regele'):
        Methods_fit_to_clinical_LAT.create_regele(endo, args)

    print('Reading regele file ...')

    elems_not_conductive = np.loadtxt(f'{meshfold}/elems_slow_conductive.regele', skiprows=1, dtype=int)

    endo_etag = vtk_to_numpy(endo.GetCellData().GetArray('elemTag'))

    elems_not_conductive = elems_not_conductive[np.where(elems_not_conductive < len(endo_etag))]

    endo_etag[elems_not_conductive] = 103

    # Save endocardium mesh in carp format

    meshNew = dsa.WrapDataObject(endo)
    meshNew.CellData.append(endo_etag, "elemTag")

    vtk_xml_unstructured_grid_writer(f"{meshfold}/LA_endo_with_fiber_30_um.vtu", meshNew.VTKObject)
    pts = vtk_to_numpy(meshNew.VTKObject.GetPoints().GetData())

    el_epi = vtk_to_numpy(meshNew.VTKObject.GetCellData().GetArray('fiber'))
    sheet_epi = vtk_to_numpy(meshNew.VTKObject.GetCellData().GetArray('sheet'))

    el_epi = el_epi / 1000
    sheet_epi = sheet_epi / 1000
    file_name = meshfold + "/LA_endo_with_fiber_30_um"
    write_to_pts(f"{file_name}.pts", pts)
    write_to_elem(f"{file_name}.elem", meshNew.VTKObject, endo_etag)
    write_to_lon(f"{file_name}.lon", el_epi, sheet_epi)

    meshname_e = file_name

    new_endo = Methods_fit_to_clinical_LAT.smart_reader(meshname_e + '.vtu')
    cellid = vtk.vtkIdFilter()
    cellid.CellIdsOn()
    cellid.SetInputData(new_endo)
    cellid.PointIdsOn()
    cellid.FieldDataOn()
    if int(vtk_version) >= 9:
        cellid.SetPointIdsArrayName('Global_ids')
        cellid.SetCellIdsArrayName('Global_ids')
    else:
        cellid.SetIdsArrayName('Global_ids')
    cellid.Update()
    new_endo = cellid.GetOutput()

    with open('element_tag.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag[row['name']] = int(row['tag'])

    reg_0 = [tag['sinus_node'], 54, 65, 70]  # SN 69, ICB_endo 54, ICB_epi 65,  CS 70
    reg_1 = [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 51, 52, 53, 54, 55, 56, 57, 58, 61, 62, 63, 66, 67, 68,
             84, 86,
             88]  # LA endo 1-7, LA epi 11-17, RA endo 51-58 , RA epi 61-68, 84 MPB, 86 UPB, CSB 88 RA endo 51-58 ->bridges
    reg_2 = tag['crista_terminalis']  # CT 60
    reg_3 = tag['pectinate_muscle']  # PM 59
    reg_4 = [tag['bachmann_bundel_left'], tag['bachmann_bundel_right'], tag['bachmann_bundel_internal']]  # BB 81,82,83

    if args.fibrotic_tissue == 1:  # append regions 101,102,103
        reg_1 = [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 17, 51, 52, 55, 56, 58, 61, 62, 63, 66, 68, 84, 86, 88, 101,
                 102, 103]

    lat = ['-num_LATs', 1,
           '-lats[0].ID', 'ACTs',
           '-lats[0].all', 0,
           '-lats[0].measurand', 0,
           '-lats[0].mode', 0,
           '-lats[0].threshold', -50]

    slow_CV = np.ones((len(endo_ids),))
    slow_CV_old = np.ones((len(endo_ids),))

    # Create the conductivity scale map with ones factors as initial condition
    f = open(simid + '/low_CV.dat', 'w')
    for i in slow_CV:
        f.write(f"{i:.4f}\n")
    f.close()

    f = open(simid + '/low_CV_old.dat', 'w')
    for i in slow_CV:
        f.write(f"{i:.4f}\n")
    f.close()

    final_diff = []
    old_cells = np.array([], dtype=int)
    lats_to_fit = np.array([])

    active_cells_old = np.array([], dtype=int)
    active_cells_band = np.array([], dtype=int)

    for l in range(len(fit_LAT)):

        RMSE = 500
        err = RMSE

        it = 1

        while RMSE > args.tol:

            cmd = tools.carp_cmd('stimulation.par')

            tissue_0 = ['-num_gregions', 6,
                        '-gregion[0].num_IDs', len(reg_0),
                        '-gregion[0].g_il', args.giL,
                        '-gregion[0].g_it', args.giL,
                        '-gregion[0].g_in', args.giL,
                        '-gregion[0].g_el', args.geL,
                        '-gregion[0].g_et ', args.geL,
                        '-gregion[0].g_en ', args.geL]

            tissue_1 = ['-gregion[1].num_IDs ', len(reg_1),
                        '-gregion[1].g_il ', args.giL,
                        '-gregion[1].g_it ', args.giL / (3.75 * 0.62),
                        '-gregion[1].g_in ', args.giL / (3.75 * 0.62),
                        '-gregion[1].g_el ', args.geL,
                        '-gregion[1].g_et ', args.geL / (3.75 * 0.62),
                        '-gregion[1].g_en ', args.geL / (3.75 * 0.62)]
            if args.fibrotic_tissue == 1:
                tissue_0 = ['-num_gregions', 8,
                            '-gregion[0].num_IDs', len(reg_0),
                            '-gregion[0].g_il', args.giL,
                            '-gregion[0].g_it', args.giL,
                            '-gregion[0].g_in', args.giL,
                            '-gregion[0].g_el', args.geL,
                            '-gregion[0].g_et ', args.geL,
                            '-gregion[0].g_en ', args.geL]

                tissue_1 = ['-gregion[1].num_IDs ', len(reg_1),
                            '-gregion[1].g_il ', args.giL,
                            '-gregion[1].g_it ', args.giL / (3.75 * 0.62),
                            '-gregion[1].g_in ', args.giL / (3.75 * 0.62),
                            '-gregion[1].g_el ', args.geL,
                            '-gregion[1].g_et ', args.geL / (3.75 * 0.62),
                            '-gregion[1].g_en ', args.geL / (3.75 * 0.62)]
            for i in range(len(reg_0)):
                tissue_0 += ['-gregion[0].ID[' + str(i) + ']', reg_0[i], ]
            for i in range(len(reg_1)):
                tissue_1 += ['-gregion[1].ID[' + str(i) + ']', reg_1[i], ]

            CT_gi = args.giL * 2
            CT_ge = args.geL * 2
            tissue_2 = ['-gregion[2].num_IDs ', 1,
                        '-gregion[2].ID ', reg_2,
                        '-gregion[2].g_il ', CT_gi,
                        '-gregion[2].g_it ', CT_gi / (6.562 * 0.62),
                        '-gregion[2].g_in ', CT_gi / (6.562 * 0.62),
                        '-gregion[2].g_el ', CT_ge,
                        '-gregion[2].g_et ', CT_ge / (6.562 * 0.62),
                        '-gregion[2].g_en ', CT_ge / (6.562 * 0.62)]
            PM_gi = args.giL * 2
            PM_ge = args.geL * 2
            tissue_3 = ['-gregion[3].num_IDs ', 1,
                        '-gregion[3].ID ', reg_3,
                        '-gregion[3].g_il ', PM_gi,
                        '-gregion[3].g_it ', PM_gi / (10.52 * 0.62),
                        '-gregion[3].g_in ', PM_gi / (10.52 * 0.62),
                        '-gregion[3].g_el ', PM_ge,
                        '-gregion[3].g_et ', PM_ge / (10.52 * 0.62),
                        '-gregion[3].g_en ', PM_ge / (10.52 * 0.62)]
            BB_gi = args.giL * 3
            BB_ge = args.geL * 3
            tissue_4 = ['-gregion[4].num_IDs ', 3,
                        '-gregion[4].g_il ', BB_gi,
                        '-gregion[4].g_it ', BB_gi / (9 * 0.62),
                        '-gregion[4].g_in ', BB_gi / (9 * 0.62),
                        '-gregion[4].g_el ', BB_ge,
                        '-gregion[4].g_et ', BB_ge / (9 * 0.62),
                        '-gregion[4].g_en ', BB_ge / (9 * 0.62)]
            for i in range(len(reg_4)):
                tissue_4 += ['-gregion[4].ID[' + str(i) + ']', reg_4[i], ]

            sigma_L = 2.1
            bilayer = ['-gregion[5].num_IDs ', 1,
                       '-gregion[5].ID ', 100,
                       '-gregion[5].g_il ', sigma_L,
                       '-gregion[5].g_it ', sigma_L,
                       '-gregion[5].g_in ', sigma_L,
                       '-gregion[5].g_el ', sigma_L,
                       '-gregion[5].g_et ', sigma_L,
                       '-gregion[5].g_en ', sigma_L]

            g_scale = ['-ge_scale_vec', simid + '/low_CV.dat',
                       '-gi_scale_vec', simid + '/low_CV.dat']

            # Set different tissue properties
            cmd += tissue_0
            cmd += tissue_1
            cmd += tissue_2
            cmd += tissue_3
            cmd += tissue_4 + bilayer
            cmd += g_scale

            writestatef = 'state'
            tsav_state = fit_LAT[l]
            # Setting the stimulus at the sinus node
            prepace = ['-num_stim', 1,
                       '-write_statef', writestatef,
                       '-num_tsav', 1,
                       '-tsav[0]', tsav_state,
                       '-stimulus[0].stimtype', 0,
                       '-stimulus[0].strength', 30.0,
                       '-stimulus[0].duration', args.stim_duration,
                       '-stimulus[0].npls', 1,
                       '-stimulus[0].ctr_def', 1,
                       '-stimulus[0].x0', stim_pt[0],
                       '-stimulus[0].xd', 3000,
                       '-stimulus[0].y0', stim_pt[1],
                       '-stimulus[0].yd', 3000,
                       '-stimulus[0].z0', stim_pt[2],
                       '-stimulus[0].zd', 3000]

            cmd += lat

            cmd += tissue_init + prepace

            cmd += ['-simID', simid,
                    '-dt', 20,
                    '-spacedt', 1,
                    '-mass_lumping', args.M_lump,
                    '-timedt', 10,
                    '-tend', tsav_state + 0.1,
                    '-meshname', meshname_e]
            # Run simulation
            remove_trash2(simid)
            job.carp(cmd)

            # Read simulated LAT map
            lats = np.loadtxt(simid + '/init_acts_ACTs-thresh.dat')
            meshNew = dsa.WrapDataObject(new_endo)
            # Convert point to cell data
            meshNew.PointData.append(lats, "lat_s")
            pt_cell = vtk.vtkPointDataToCellData()
            pt_cell.SetInputData(meshNew.VTKObject)
            pt_cell.AddPointDataArray("lat_s")
            pt_cell.PassPointDataOn()
            pt_cell.CategoricalDataOff()
            pt_cell.ProcessAllArraysOff()
            pt_cell.Update()

            model = pt_cell.GetOutput()
            meshNew = dsa.WrapDataObject(model)
            # Extract all not fibrotic tissue (103 is not conductive)
            healthy_endo = Methods_fit_to_clinical_LAT.vtk_thr(model, 1, "CELLS", "elemTag", 102)
            # Extract all cells which are activated
            active = Methods_fit_to_clinical_LAT.vtk_thr(healthy_endo, 0, "POINTS", "lat_s", 0)

            active_cells = vtk_to_numpy(active.GetCellData().GetArray('Global_ids')).astype(int)
            print(f"active_cells: {len(active_cells)}")
            act_cls_old = np.zeros((model.GetNumberOfCells(),))
            act_cls = np.zeros((model.GetNumberOfCells(),))
            meshNew.CellData.append(act_cls, "act_cls")
            meshNew.CellData.append(act_cls_old, "act_cls_old")
            active_cells_old = np.array(active_cells_band, dtype=int)

            # Remove from fitting all the cells which were fitted in the previous step
            active_cells_band = np.setdiff1d(active_cells, old_cells)
            act_cls_old[active_cells_old] = 1
            act_cls[active_cells] = 1

            lats_to_fit_old = np.array(lats_to_fit)
            lats_to_fit = vtk_to_numpy(model.GetCellData().GetArray('lat_s'))

            if len(lats_to_fit_old) > 0:
                meshNew.CellData.append(lats_to_fit_old, "LATs_old")
            meshNew.CellData.append(LAT_map, "LAT_to_clean")

            # Find all active areas (border = 2 and core = 1) marked as wrong annotation, we give to the core the mean of the active border
            active_to_interpolate = []
            active_border = []
            idss = np.zeros((model.GetNumberOfCells(),))
            l_idss = np.zeros((model.GetNumberOfCells(),))
            for k in range(len(el_to_clean)):
                idss[el_to_clean[k]] = 1
                idss[el_border[k]] = 2
                current_active_to_interp = np.setdiff1d(np.intersect1d(el_to_clean[k], active_cells_band), old_cells)
                if len(current_active_to_interp > 0):
                    active_to_interpolate.append(current_active_to_interp)
                    active_border.append(np.setdiff1d(np.intersect1d(el_border[k], active_cells_band), old_cells))
                    l_idss[current_active_to_interp] = 1
                    l_idss[np.setdiff1d(np.intersect1d(el_border[k], active_cells_band), old_cells)] = 2

            meshNew.CellData.append(idss, "idss")
            meshNew.CellData.append(l_idss, "l_idss")

            last_ACT = np.mean(lats_to_fit[active_cells_band])

            print(f"ACT to fit: {fit_LAT[l]}")
            print(f"last ACT: {last_ACT}")
            print(f"old_cells: {len(old_cells)}")

            # Compute RMSE between simulated and clinical LAT excluding elements to clean (marked as wrong annotation)
            if len(lats_to_fit[active_cells_band]) > 0:
                if len(active_border) > 0:
                    print("Active border")
                    current_active_to_interp = np.array([], dtype=int)
                    for k in range(len(active_to_interpolate)):
                        current_active_to_interp = np.union1d(current_active_to_interp, active_to_interpolate[k])
                    active_cleaned_cells = np.setdiff1d(active_cells_band, current_active_to_interp)
                    RMSE = mean_squared_error(LAT_map[active_cleaned_cells], lats_to_fit[active_cleaned_cells],
                                              squared=False)
                else:
                    RMSE = mean_squared_error(LAT_map[active_cells_band], lats_to_fit[active_cells_band], squared=False)
                print("RMSE: ", RMSE)
                print("err: ", err)

            if RMSE > args.tol and RMSE + args.tol * 0.25 < err:  # Stopping criteria: RMSE< tol or new RMSE + 0.25*tol > old RMSE

                meshNew.CellData.append(slow_CV_old, "slow_CV_old")
                slow_CV_old[:] = slow_CV[:]
                active_cells_old_old = np.array(active_cells_old, dtype=int)
                if len(active_border) > 0:  # Elements to clean
                    # For each area to clean, give to the active core the mean of conductivity of the active border
                    slow_CV[active_cleaned_cells] = slow_CV[active_cleaned_cells] * (
                            (lats_to_fit[active_cleaned_cells] / (LAT_map[active_cleaned_cells])) ** 2)
                    for k in range(len(active_to_interpolate)):
                        if len(active_border[k]) > 0:
                            slow_CV[active_to_interpolate[k]] = np.mean(slow_CV[active_border[k]])
                else:  # No elements to clean
                    # sigma_new = sigma_old*(lat_simulated/lat_clinical)^2 for sigma = CV^2 see https://opencarp.org/documentation/examples/02_ep_tissue/03a_study_prep_tunecv
                    slow_CV[active_cells_band] = slow_CV[active_cells_band] * (
                            (lats_to_fit[active_cells_band] / (LAT_map[active_cells_band])) ** 2)

                slow_CV = np.where(slow_CV > 3.5, 3.5, slow_CV)  # Set an upper bound in CV of 2.15 m/s
                slow_CV = np.where(slow_CV < 0.15, 0.15, slow_CV)  # Set a lower bound in CV of 0.35 m/s

                meshNew.CellData.append(slow_CV, "slow_CV")

                vtk_xml_unstructured_grid_writer(job.ID + f"/endo_cleaned_{l}.vtu", meshNew.VTKObject)

                LAT_diff = RMSE
                os.rename(simid + '/low_CV.dat', simid + '/low_CV_old.dat')
                f = open(simid + '/low_CV.dat', 'w')
                for i in slow_CV:
                    f.write(f"{i:.4f}\n")
                f.close()
                it += 1
            else:
                old_cells = np.union1d(old_cells, active_cells_old_old)
                slow_CV[:] = slow_CV_old[:]
                LATs_diff = np.zeros((model.GetNumberOfCells(),))
                LATs_diff[old_cells] = lats_to_fit_old[old_cells] - LAT_map[old_cells]
                meshNew.CellData.append(slow_CV, "slow_CV")
                meshNew.CellData.append(LATs_diff, "LATs_diff")
                meshNew.CellData.append(slow_CV_old, "slow_CV_old")
                final_diff.append(LAT_diff)

                vtk.vtkXMLUnstructuredGridWriter(job.ID + f"/endo_cleaned_{l}.vtu", meshNew.VTKObject)
                break

            err = RMSE

    cmd = tools.carp_cmd('stimulation.par')
    g_scale = ['-ge_scale_vec', simid + '/low_CV_old.dat',
               '-gi_scale_vec', simid + '/low_CV_old.dat']
    # Set different tissue properties
    cmd += tissue_0
    cmd += tissue_1
    cmd += tissue_2
    cmd += tissue_3
    cmd += tissue_4 + bilayer + g_scale

    # cmd += fibrotic_tissue
    cmd += lat
    # Setting the stimulus at the sinus node
    prepace = ['-num_stim', 1,
               '-write_statef', writestatef,
               '-num_tsav', 1,
               '-tsav[0]', tsav_state,
               '-stimulus[0].stimtype', 0,
               '-stimulus[0].strength', 30.0,
               '-stimulus[0].duration', 2.0,
               '-stimulus[0].npls', 1,
               '-stimulus[0].ctr_def', 1,
               '-stimulus[0].x0', stim_pt[0],
               '-stimulus[0].xd', 3000,
               '-stimulus[0].y0', stim_pt[1],
               '-stimulus[0].yd', 3000,
               '-stimulus[0].z0', stim_pt[2],
               '-stimulus[0].zd', 3000]

    cmd += tissue_init + prepace
    cmd += ['-simID', simid,
            '-dt', 20,
            '-spacedt', 1,
            '-mass_lumping', args.M_lump,
            '-timedt', 10,
            '-num_tsav', 1,
            '-tsav[0]', tsav_state,
            '-tend', tsav_state + 2 * args.step + 0.1,
            '-meshname', meshname_e]
    # Run simulation
    remove_trash2(simid)
    job.carp(cmd)

    model_cleaned = Methods_fit_to_clinical_LAT.vtk_thr(meshNew.VTKObject, 2, "CELLS", "idss", 0, 0)

    cleaned_ids = vtk_to_numpy(model_cleaned.GetPointData().GetArray('Global_ids')).astype(int)

    lats = np.loadtxt(simid + '/init_acts_ACTs-thresh.dat')

    lats_to_fit = vtk_to_numpy(model.GetPointData().GetArray('lat')) - min_LAT

    RMSE = mean_squared_error(lats[cleaned_ids], lats_to_fit[cleaned_ids], squared=False)

    final_diff.append(RMSE)

    print(RMSE)

    print(f"Final last ACT: {last_ACT}")
    print(f"Final giL: {args.giL}")
    print(f"Final geL: {args.geL}")
    f = open(job.ID + '/err.dat', 'w')
    for i in final_diff:
        f.write(f"{i:.4f}\n")
    f.close()

    if os.path.exists('RMSE_patients.txt'):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    f = open('RMSE_patients.txt', append_write)
    f.write(f"{args.mesh} {args.step} {args.thr} {RMSE:.2f}\n")
    f.close()

    slow_CV = np.loadtxt(simid + '/low_CV_old.dat')
    slow_CV_bil = np.ones((bilayer_n_cells,))
    slow_CV_bil[endo_ids] = slow_CV
    slow_CV_bil[endo_ids + len(endo_ids)] = slow_CV

    f = open(meshfold + f'/low_CV_3_{args.step}_{args.thr}.dat', 'w')
    for i in slow_CV_bil:
        f.write(f"{i:.4f}\n")
    f.close()

    meshNew = dsa.WrapDataObject(new_endo)
    meshNew.PointData.append(lats, "lat_s")
    pt_cell = vtk.vtkPointDataToCellData()
    pt_cell.SetInputData(meshNew.VTKObject)
    pt_cell.AddPointDataArray("lat_s")
    pt_cell.PassPointDataOn()
    pt_cell.CategoricalDataOff()
    pt_cell.ProcessAllArraysOff()
    pt_cell.Update()

    meshNew.CellData.append(LAT_map, "LAT_to_clean")
    LATs_diff = vtk_to_numpy(pt_cell.GetOutput().GetCellData().GetArray('lat_s')) - LAT_map
    meshNew.CellData.append(slow_CV, "slow_CV")
    meshNew.CellData.append(LATs_diff, "LATs_diff")

    vtk_xml_unstructured_grid_writer(job.ID + "/endo_final.vtu", meshNew.VTKObject)


if __name__ == '__main__':
    run()
