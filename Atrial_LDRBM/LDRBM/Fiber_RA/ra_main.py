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
import datetime
import os
import warnings

import numpy as np
import pandas as pd
import vtk
from carputils import tools

import Atrial_LDRBM.LDRBM.Fiber_RA.Methods_RA as Method
from Atrial_LDRBM.LDRBM.Fiber_RA.create_bridges import add_free_bridge
from Atrial_LDRBM.LDRBM.Fiber_RA.ra_generate_fiber import ra_generate_fiber
from Atrial_LDRBM.LDRBM.Fiber_RA.ra_laplace import ra_laplace
from vtk_opencarp_helper_methods.openCARP.exporting import write_to_pts, write_to_elem, write_to_lon
from vtk_opencarp_helper_methods.vtk_methods.converters import vtk_to_numpy


def parser():
    # Generate the standard command line parser
    parser = tools.standard_parser()
    # Add arguments    
    parser.add_argument('--mesh',
                        type=str,
                        default="",
                        help='path to meshname')
    parser.add_argument('--ifmt',
                        type=str,
                        default="vtk",
                        help='input mesh format')
    parser.add_argument('--mesh_type',
                        default='bilayer',
                        choices=['vol',
                                 'bilayer'],
                        help='Mesh type')
    parser.add_argument('--debug',
                        type=int,
                        default=1,
                        help='path to meshname')
    parser.add_argument('--scale',
                        type=int,
                        default=1,
                        help='normal unit is mm, set scaling factor if different')
    parser.add_argument('--ofmt',
                        default='vtu',
                        choices=['vtu', 'vtk'],
                        help='Output mesh format')
    parser.add_argument('--normals_outside',
                        type=int,
                        default=0,
                        help='set to 1 if surface normals are pointing outside')  # expects normals to be pointing inside
    parser.add_argument('--add_bridges',
                        type=int,
                        default=1,
                        help='set to 1 to compute and add interatrial bridges, 0 otherwise')
    parser.add_argument('--just_bridges',
                        type=int,
                        default=0,
                        help='set to 1 to only check bridges')
    parser.add_argument('--laplace',
                        type=int,
                        default=1,
                        help='set to 1 to run laplace solutions')

    return parser


def jobID(args):
    ID = f'{args.mesh}_fibers'
    return ID


@tools.carpexample(parser, jobID)
def run(args, job):
    RA_mesh = args.mesh + '_surf/RA'

    if args.mesh_type == "bilayer":
        reader = vtk.vtkPolyDataReader()
    else:
        reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(RA_mesh + '.vtk')
    reader.Update()
    RA = reader.GetOutput()

    if args.normals_outside:
        reverse = vtk.vtkReverseSense()
        reverse.ReverseCellsOn()
        reverse.ReverseNormalsOn()
        reverse.SetInputConnection(reader.GetOutputPort())
        reverse.Update()

        RA = reverse.GetOutput()

    pts = vtk_to_numpy(RA.GetPoints().GetData())

    write_to_pts(RA_mesh + '.pts', pts)

    write_to_elem(RA_mesh + '.elem', RA, np.ones(RA.GetNumberOfCells(), dtype=int))

    fibers = np.zeros((RA.GetNumberOfCells(), 6))
    fibers[:, 0] = 1
    fibers[:, 4] = 1

    write_to_lon(RA_mesh + '.lon', fibers, [fiber[3:6] for fiber in fibers])
    warnings.warn("Test if lon is storred correctly ra_main.py l120 ff.")

    start_time = datetime.datetime.now()
    print('[Step 1] Solving laplace-dirichlet... ' + str(start_time))
    if args.laplace:
        output_laplace = ra_laplace(args, job, RA)
    else:
        output_laplace = Method.smart_reader(job.ID + "/gradient/RA_with_lp_res_gradient.vtu")
        print("Reading Laplace: " + job.ID + "/gradient/RA_with_lp_res_gradient.vtu")

    end_time = datetime.datetime.now()
    running_time = end_time - start_time
    print('[Step 1] Solving laplace-dirichlet...done! ' + str(end_time) + '\nRunning time: ' + str(running_time) + '\n')

    start_time = datetime.datetime.now()
    print('[Step 2] Generating fibers... ' + str(start_time))

    if args.just_bridges:
        la_epi = Method.smart_reader(job.ID + "/result_LA/LA_epi_with_fiber.vtu")
        model = Method.smart_reader(job.ID + "/result_RA/RA_epi_with_fiber.vtu")
        df = pd.read_csv(args.mesh + "_surf/rings_centroids.csv")
        CS_p = np.array(df["CS"])
        add_free_bridge(args, la_epi, model, CS_p, df, job)

        args.atrium = "LA_RA"
        os.system("meshtool convert -imsh={} -ifmt=carp_txt -omsh={} -ofmt=carp_txt -scale={}".format(
            '{}_fibers/result_RA/{}_bilayer_with_fiber'.format(args.mesh, args.atrium),
            '{}_fibers/result_RA/{}_bilayer_with_fiber_um'.format(args.mesh, args.atrium), 1000 * args.scale))
        os.system("meshtool convert -imsh={} -ifmt=carp_txt -omsh={} -ofmt=vtk".format(
            '{}_fibers/result_RA/{}_bilayer_with_fiber_um'.format(args.mesh, args.atrium),
            '{}_fibers/result_RA/{}_bilayer_with_fiber_um'.format(args.mesh, args.atrium)))
    else:
        ra_generate_fiber(output_laplace, args, job)

    end_time = datetime.datetime.now()
    running_time = end_time - start_time
    print('[Step 2] Generating fibers...done! ' + str(end_time) + '\nRunning time: ' + str(running_time) + '\n')


if __name__ == '__main__':
    run()
