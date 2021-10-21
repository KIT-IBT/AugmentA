#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:55:02 2021

@author: Luca Azzolin
"""
import os
import subprocess as sp
import datetime
import vtk
import numpy as np
from vtk.util import numpy_support
from carputils import settings
from carputils import tools
from carputils import mesh
from la_laplace import la_laplace
from la_generate_fiber import la_generate_fiber
import Method

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
                        choices=['vtu','vtk'],
                        help='Output mesh format')
    parser.add_argument('--normals_outside',
                    type=int,
                    default=1,
                    help='set to 1 if surface normals are pointing outside')

    return parser

def jobID(args):
    ID = '{}_fibers'.format(args.mesh)
    return ID

@tools.carpexample(parser, jobID)
def run(args, job):
    
    LA_mesh = args.mesh+'_surf/LA'
    
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(LA_mesh+'.vtk')
    reader.Update()
    LA = reader.GetOutput()
    
    if args.normals_outside:
        reverse = vtk.vtkReverseSense()
        reverse.ReverseCellsOn()
        reverse.ReverseNormalsOn()
        reverse.SetInputConnection(reader.GetOutputPort())
        reverse.Update()

        LA = reverse.GetOutput()

    pts = numpy_support.vtk_to_numpy(LA.GetPoints().GetData())
    
    with open(LA_mesh+'.pts',"w") as f:
        f.write("{}\n".format(len(pts)))
        for i in range(len(pts)):
            f.write("{} {} {}\n".format(pts[i][0], pts[i][1], pts[i][2]))
    
    with open(LA_mesh+'.elem',"w") as f:
            f.write("{}\n".format(LA.GetNumberOfCells()))
            for i in range(LA.GetNumberOfCells()):
                cell = LA.GetCell(i)
                if cell.GetNumberOfPoints() == 2:
                    f.write("Ln {} {} {}\n".format(cell.GetPointIds().GetId(0), cell.GetPointIds().GetId(1), 1))
                elif cell.GetNumberOfPoints() == 3:
                    f.write("Tr {} {} {} {}\n".format(cell.GetPointIds().GetId(0), cell.GetPointIds().GetId(1), cell.GetPointIds().GetId(2), 1))
                elif cell.GetNumberOfPoints() == 4:
                    f.write("Tt {} {} {} {} {}\n".format(cell.GetPointIds().GetId(0), cell.GetPointIds().GetId(1), cell.GetPointIds().GetId(2), cell.GetPointIds().GetId(3), 1))
    
    fibers = np.zeros((LA.GetNumberOfCells(),6))
    fibers[:,0]=1
    fibers[:,4]=1
    
    with open(LA_mesh+'.lon',"w") as f:
        f.write("2\n")
        for i in range(len(fibers)):
            f.write("{} {} {} {} {} {}\n".format(fibers[i][0], fibers[i][1], fibers[i][2], fibers[i][3],fibers[i][4],fibers[i][5]))
            
    
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

if __name__ == '__main__':
    run()
