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
from carputils import tools
from ra_calculate_gradient import ra_calculate_gradient
import vtk
from carputils.carpio import igb
from vtk.numpy_interface import dataset_adapter as dsa

EXAMPLE_DIR = os.path.dirname(os.path.realpath(__file__))


def ra_laplace(args, job, model):
    meshdir = args.mesh + '_surf/RA'
    surfdir = f'{args.mesh}_surf/'
    parfdir = os.path.join(EXAMPLE_DIR, 'Parfiles')

    if args.mesh_type == 'vol':
        ####################################
        # Solver for the phi laplace soluton
        ####################################
        cmd = tools.carp_cmd(parfdir + '/ra_lps_phi.par')
        simid = job.ID + '/Lp_phi'
        cmd += ['-simID', simid,
                '-meshname', meshdir,
                '-stimulus[0].vtx_file', surfdir + 'ids_ENDO',
                '-stimulus[1].vtx_file', surfdir + 'ids_EPI']

        # Run simulation
        job.carp(cmd)

    #####################################
    # Solver for the ab laplace soluton
    #####################################
    cmd = tools.carp_cmd(parfdir + '/ra_lps_ab.par')
    simid = job.ID + '/Lp_ab'
    cmd += ['-simID', simid,
            '-meshname', meshdir,
            '-stimulus[0].vtx_file', surfdir + 'ids_SVC',
            '-stimulus[1].vtx_file', surfdir + 'ids_IVC',
            '-stimulus[2].vtx_file', surfdir + 'ids_TV_S',
            '-stimulus[3].vtx_file', surfdir + 'ids_TV_F',
            '-stimulus[4].vtx_file', surfdir + 'ids_RAA']

    # Run simulation
    job.carp(cmd)

    #####################################
    # Solver for the v laplace soluton
    #####################################
    cmd = tools.carp_cmd(parfdir + '/ra_lps_v.par')
    simid = job.ID + '/Lp_v'
    cmd += ['-simID', simid,
            '-meshname', meshdir,
            '-stimulus[0].vtx_file', surfdir + 'ids_SVC',
            '-stimulus[1].vtx_file', surfdir + 'ids_RAA',
            '-stimulus[2].vtx_file', surfdir + 'ids_IVC']

    # Run simulation
    job.carp(cmd)

    #####################################
    # Solver for the v2 laplace soluton
    #####################################
    cmd = tools.carp_cmd(parfdir + '/ra_lps_phi.par')
    simid = job.ID + '/Lp_v2'
    cmd += ['-simID', simid,
            '-meshname', meshdir,
            '-stimulus[0].vtx_file', surfdir + 'ids_IVC',
            '-stimulus[1].vtx_file', surfdir + 'ids_RAA']

    # Run simulation
    job.carp(cmd)

    if args.mesh_type == 'vol':
        #####################################
        # Solver for the r laplace soluton
        #####################################
        cmd = tools.carp_cmd(parfdir + '/ra_lps_r_vol.par')
        simid = job.ID + '/Lp_r'
        cmd += ['-simID', simid,
                '-meshname', meshdir,
                '-stimulus[0].vtx_file', surfdir + 'ids_TOP_ENDO',
                '-stimulus[1].vtx_file', surfdir + 'ids_TOP_EPI',
                '-stimulus[2].vtx_file', surfdir + 'ids_TV_F',
                '-stimulus[3].vtx_file', surfdir + 'ids_TV_S']

        # Run simulation
        job.carp(cmd)

    else:
        cmd = tools.carp_cmd(parfdir + '/ra_lps_r.par')
        simid = job.ID + '/Lp_r'
        cmd += ['-simID', simid,
                '-meshname', meshdir,
                '-stimulus[0].vtx_file', surfdir + 'ids_TOP_ENDO',
                '-stimulus[1].vtx_file', surfdir + 'ids_TV_F',
                '-stimulus[2].vtx_file', surfdir + 'ids_TV_S']

        # Run simulation
        job.carp(cmd)

    #####################################
    # Solver for the w laplace soluton
    #####################################
    cmd = tools.carp_cmd(parfdir + '/ra_lps_w.par')
    simid = job.ID + '/Lp_w'
    cmd += ['-simID', simid,
            '-meshname', meshdir,
            '-stimulus[0].vtx_file', surfdir + 'ids_TV_S',
            '-stimulus[1].vtx_file', surfdir + 'ids_TV_F']

    # Run simulation
    job.carp(cmd)

    """
    generate .vtu files that contain the result of laplace solution as point/cell data
    """
    meshNew = dsa.WrapDataObject(model)

    name_list = ['r', 'v', 'v2', 'ab', 'w']
    if args.mesh_type == 'vol':
        name_list = ['phi', 'r', 'v', 'v2', 'ab', 'w']
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
            writer = vtk.vtkXMLUnstructuredGridWriter()
            writer.SetFileName(simid + "/RA_with_laplace.vtu")
        else:
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(simid + "/RA_with_laplace.vtp")
        writer.SetInputData(meshNew.VTKObject)
        writer.Write()
    """
    calculate the gradient
    """
    output = ra_calculate_gradient(args, meshNew.VTKObject, job)

    return output
