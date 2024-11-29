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
import warnings

import numpy as np
import vtk
from carputils import tools

from Atrial_LDRBM.LDRBM.Fiber_LA.la_generate_fiber import la_generate_fiber
from Atrial_LDRBM.LDRBM.Fiber_LA.la_laplace import la_laplace
from vtk_opencarp_helper_methods.openCARP.exporting import write_to_pts, write_to_elem, write_to_lon
from vtk_opencarp_helper_methods.vtk_methods.converters import vtk_to_numpy
from vtk_opencarp_helper_methods.vtk_methods.reader import smart_reader


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
                        default=0,
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
                        default=1,
                        help='set to 1 if surface normals are pointing outside')

    return parser


def jobID(args):
    ID = f'{args.mesh}_fibers'
    return ID


@tools.carpexample(parser, jobID)
def run(args, job):
    LA = init_mesh_and_fibers(args, "LA")

    start_time = datetime.datetime.now()
    init_start_time = datetime.datetime.now()

    print('[Step 1] Solving laplace-dirichlet... ' + str(start_time))
    output_laplace = la_laplace(args, job, LA)
    end_time = datetime.datetime.now()
    running_time = end_time - start_time
    print('[Step 1] Solving laplace-dirichlet...done! ' + str(end_time) + '\nRunning time: ' + str(running_time) + '\n')

    start_time = datetime.datetime.now()

    print('[Step 2] Generating fibers... ' + str(start_time))
    la_generate_fiber(output_laplace, args, job)
    end_time = datetime.datetime.now()
    running_time = end_time - start_time
    print('[Step 2] Generating fibers...done! ' + str(end_time) + '\nRunning time: ' + str(running_time) + '\n')
    fin_end_time = datetime.datetime.now()
    tot_running_time = fin_end_time - init_start_time
    print('Total running time: ' + str(tot_running_time))


def init_mesh_and_fibers(args, atrium):
    """
    Initializes the mesh and fibers for the fiber generation.
    Sores both to disk for further processing.

    :param args:
    :param atrium: 'LA' or 'RA'
    :return: The loaded mesh
    """
    mesh = args.mesh + f'_surf/{atrium}'
    atrial_mesh = smart_reader(mesh + '.vtk')
    if args.normals_outside:
        reverse = vtk.vtkReverseSense()
        reverse.ReverseCellsOn()
        reverse.ReverseNormalsOn()
        reverse.SetInputData(atrial_mesh)
        reverse.Update()

        atrial_mesh = reverse.GetOutput()
    pts = vtk_to_numpy(atrial_mesh.GetPoints().GetData())
    write_to_pts(mesh + '.pts', pts)
    write_to_elem(mesh + '.elem', atrial_mesh, np.ones(atrial_mesh.GetNumberOfCells(), dtype=int))
    init_fibers(atrial_mesh, atrium, mesh)
    return atrial_mesh


def init_fibers(atrial_mesh, atrium, mesh):
    """
    Initializes fibers with ones and stores them to disk for further processing.
    :param atrial_mesh:
    :param atrium:
    :param mesh:
    :return:
    """
    fibers = np.zeros((atrial_mesh.GetNumberOfCells(), 6))
    fibers[:, 0] = 1
    fibers[:, 4] = 1
    warnings.warn(f"Test if lon is stored correctly {atrium}_main.py l116 ff.")
    write_to_lon(mesh + '.lon', fibers, [fiber[3:6] for fiber in fibers], precession=1)


if __name__ == '__main__':
    run()
