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
import os

from vtk_opencarp_helper_methods.vtk_methods.exporting import vtk_xml_unstructured_grid_writer, vtk_polydata_writer

EXAMPLE_DIR = os.path.dirname(os.path.realpath(__file__))

from carputils.carpio import igb
from carputils import tools
from la_calculate_gradient import la_calculate_gradient
from vtk.numpy_interface import dataset_adapter as dsa


def la_laplace(args, job, model):
    meshdir = args.mesh + '_surf/LA'
    surfdir = f'{args.mesh}_surf/'
    parfdir = os.path.join(EXAMPLE_DIR, 'Parfiles')

    if args.mesh_type == 'vol':
        ####################################
        # Solver for the phi laplace solution
        ####################################
        cmd = tools.carp_cmd(parfdir + '/la_lps_phi.par')
        simid = job.ID + '/Lp_phi'
        cmd += ['-simID', simid,
                '-meshname', meshdir,
                '-stimulus[0].vtx_file', surfdir + 'ids_ENDO',
                '-stimulus[1].vtx_file', surfdir + 'ids_EPI']

        # Run simulation
        job.carp(cmd)

    #####################################
    # Solver for the ab laplace solution
    #####################################
    cmd = tools.carp_cmd(parfdir + '/la_lps_ab.par')
    simid = job.ID + '/Lp_ab'
    cmd += ['-simID', simid,
            '-meshname', meshdir,
            '-stimulus[0].vtx_file', surfdir + 'ids_LPV',
            '-stimulus[1].vtx_file', surfdir + 'ids_RPV',
            '-stimulus[2].vtx_file', surfdir + 'ids_MV',
            '-stimulus[3].vtx_file', surfdir + 'ids_LAA']

    # Run simulation
    job.carp(cmd)

    #####################################
    # Solver for the v laplace solution
    #####################################
    cmd = tools.carp_cmd(parfdir + '/la_lps_v.par')
    simid = job.ID + '/Lp_v'
    cmd += ['-simID', simid,
            '-meshname', meshdir,
            '-stimulus[0].vtx_file', surfdir + 'ids_LPV',
            '-stimulus[1].vtx_file', surfdir + 'ids_RPV']

    # Run simulation
    job.carp(cmd)

    #####################################
    # Solver for the r laplace solution
    #####################################
    cmd = tools.carp_cmd(parfdir + '/la_lps_r.par')
    simid = job.ID + '/Lp_r'
    cmd += ['-simID', simid,
            '-meshname', meshdir,
            '-stimulus[0].vtx_file', surfdir + 'ids_LPV',
            '-stimulus[1].vtx_file', surfdir + 'ids_RPV',
            '-stimulus[2].vtx_file', surfdir + 'ids_LAA',
            '-stimulus[3].vtx_file', surfdir + 'ids_MV']

    # Run simulation
    job.carp(cmd)

    #####################################
    # Solver for the r2 laplace solution
    #####################################
    cmd = tools.carp_cmd(parfdir + '/la_lps_r2.par')
    simid = job.ID + '/Lp_r2'
    cmd += ['-simID', simid,
            '-meshname', meshdir,
            '-stimulus[0].vtx_file', surfdir + 'ids_LPV',
            '-stimulus[1].vtx_file', surfdir + 'ids_RPV',
            '-stimulus[2].vtx_file', surfdir + 'ids_MV']

    # Run simulation
    job.carp(cmd)

    """
    generate .vtu files that contain the result of laplace solution as point/cell data
    """

    meshNew = dsa.WrapDataObject(model)

    name_list = ['r', 'r2', 'v', 'ab']
    if args.mesh_type == 'vol':
        name_list = ['phi', 'r', 'r2', 'v', 'ab']
    for var in name_list:
        data = igb.IGBFile(job.ID + "/Lp_" + str(var) + "/phie.igb").data()
        # add the vtk array to model
        meshNew.PointData.append(data, "phie_" + str(var))

    if args.debug == 1:
        # write
        simid = job.ID + "/Laplace_Result"
        try:
            os.makedirs(simid)
        except OSError:
            print(f"Creation of the directory {simid} failed")
        else:
            print(f"Successfully created the directory {simid} ")
        if args.mesh_type == "vol":
            vtk_xml_unstructured_grid_writer(simid + "/LA_with_laplace.vtu", meshNew.VTKObject)
        else:
            vtk_polydata_writer(simid + "/LA_with_laplace.vtp", meshNew.VTKObject, store_xml=True)

    """
    calculate the gradient
    """
    output = la_calculate_gradient(args, meshNew.VTKObject, job)

    return output


def laplace_0_1(args, job, model, name1, name2, outname):
    meshdir = args.mesh + '_surf/LA'
    surfdir = f'{args.mesh}_surf/'
    parfdir = os.path.join(EXAMPLE_DIR, 'Parfiles')
    #####################################
    # Solver for the ab laplace solution
    #####################################
    cmd = tools.carp_cmd(parfdir + '/la_lps_phi.par')
    simid = job.ID + '/Lp_ab'
    cmd += ['-simID', simid,
            '-meshname', meshdir,
            '-stimulus[0].vtx_file', surfdir + f'ids_{name1}',
            '-stimulus[1].vtx_file', surfdir + f'ids_{name2}']

    if name1 == "4":
        cmd = tools.carp_cmd(parfdir + '/la_lps_phi_0_1_4.par')
        simid = job.ID + '/Lp_ab'
        cmd += ['-simID', simid,
                '-meshname', meshdir,
                '-stimulus[0].vtx_file', surfdir + 'ids_LSPV',
                '-stimulus[1].vtx_file', surfdir + 'ids_LIPV',
                '-stimulus[2].vtx_file', surfdir + 'ids_RSPV',
                '-stimulus[3].vtx_file', surfdir + 'ids_RIPV']
    # Run simulation
    job.carp(cmd)

    meshNew = dsa.WrapDataObject(model)
    data = igb.IGBFile(job.ID + "/Lp_ab/phie.igb").data()
    meshNew.PointData.append(data, outname)

    if args.debug == 1:
        # write
        simid = job.ID + "/Laplace_Result"
        try:
            os.makedirs(simid)
        except OSError:
            print(f"Creation of the directory {simid} failed")
        else:
            print(f"Successfully created the directory {simid} ")

        vtk_xml_unstructured_grid_writer(simid + "/LA_with_laplace.vtu", meshNew.VTKObject)
    return meshNew.VTKObject
