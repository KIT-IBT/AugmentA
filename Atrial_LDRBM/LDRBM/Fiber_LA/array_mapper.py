#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:55:02 2021

@author: Luca Azzolin
"""
import vtk
import Method
from carputils import tools
from vtk.numpy_interface import dataset_adapter as dsa

def parser():
    # Generate the standard command line parser
    parser = tools.standard_parser()
    # Add arguments    
    parser.add_argument('--mesh1',
                    type=str,
                    default="",
                    help='path to meshname')
    parser.add_argument('--mesh2',
                    type=str,
                    default="",
                    help='path to meshname')
    parser.add_argument('--idat',
                    type=str,
                    default="",
                    help='input mesh format')
    parser.add_argument('--odat',
                    type=str,
                    default="",
                    help='input mesh format')
    parser.add_argument('--pts_or_cells',
                        default='points',
                        choices=['points',
                                 'cells'],
                        help='Mesh type')

    return parser

def jobID(args):
    ID = '{}_fibers'.format(args.mesh1.split('/')[-1])
    return ID

@tools.carpexample(parser, jobID)
def run(args, job):
    
    mesh1 = Method.smart_reader(args.mesh1)
    
    mesh2 = Method.smart_reader(args.mesh2)
    
    if args.pts_or_cells == "points":
        Method.point_array_mapper(mesh1, mesh2, args.mesh2, args.idat)
    else:
        Method.cell_array_mapper(mesh1, mesh2, args.mesh2, args.idat)
    
if __name__ == '__main__':
    run()