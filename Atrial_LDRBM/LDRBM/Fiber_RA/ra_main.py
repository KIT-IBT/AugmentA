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
import numpy as np
import vtk
from vtk.util import numpy_support
import subprocess as sp
import datetime
from carputils import tools
from ra_laplace import ra_laplace
from ra_generate_fiber import ra_generate_fiber

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

    return parser

def jobID(args):
    ID = '{}_fibers'.format(args.mesh)
    return ID

@tools.carpexample(parser, jobID)
def run(args, job):
    
    RA_mesh = args.mesh+'_surf/RA'
    
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(RA_mesh+'.vtk')
    reader.Update()
    RA = reader.GetOutput()

    # # Warning: set -1 if pts normals are pointing outside
    # reverse = vtk.vtkReverseSense()
    # reverse.ReverseCellsOn()
    # reverse.ReverseNormalsOn()
    # reverse.SetInputConnection(reader.GetOutputPort())
    # reverse.Update()

    # RA = reverse.GetOutput()

    pts = numpy_support.vtk_to_numpy(RA.GetPoints().GetData())
    cells = numpy_support.vtk_to_numpy(RA.GetPolys().GetData())
    cells = cells.reshape(int(len(cells)/4),4)[:,1:]
    
    with open(RA_mesh+'.pts',"w") as f:
        f.write("{}\n".format(len(pts)))
        for i in range(len(pts)):
            f.write("{} {} {}\n".format(pts[i][0], pts[i][1], pts[i][2]))
    
    with open(RA_mesh+'.elem',"w") as f:
        f.write("{}\n".format(len(cells)))
        for i in range(len(cells)):
            f.write("Tr {} {} {} 1\n".format(cells[i][0], cells[i][1], cells[i][2]))
    
    fibers = np.zeros((len(cells),6))
    fibers[:,0]=1
    fibers[:,4]=1
    
    with open(RA_mesh+'.lon',"w") as f:
        f.write("2\n")
        for i in range(len(fibers)):
            f.write("{} {} {} {} {} {}\n".format(fibers[i][0], fibers[i][1], fibers[i][2], fibers[i][3],fibers[i][4],fibers[i][5]))
            
    start_time = datetime.datetime.now()
    print('[Step 1] Solving laplace-ddirichlet... ' + str(start_time))
    output_laplace = ra_laplace(args, job, RA)
    end_time = datetime.datetime.now()
    running_time = end_time - start_time
    print('[Step 1] Solving laplace-ddirichlet...done! ' + str(end_time) + '\nRunning time: ' + str(running_time) + '\n')

    start_time = datetime.datetime.now()
    print('[Step 2] Generating fibers... ' + str(start_time))
    ra_generate_fiber(output_laplace, args, job)
    end_time = datetime.datetime.now()
    running_time = end_time - start_time
    print('[Step 2] Generating fibers...done! ' + str(end_time) + '\nRunning time: ' + str(running_time) + '\n')


if __name__ == '__main__':
    run()