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

EXAMPLE_DESCRIPTIVE_NAME = 'AugmentA: Patient-specific Augmented Atrial model Generation Tool'
EXAMPLE_AUTHOR = 'Luca Azzolin <luca.azzolin@kit.edu>'

import sys
import os
import shutil
import argparse
from pipeline import AugmentA

EXAMPLE_DIR = os.path.dirname(os.path.realpath(__file__))

def parser():
    # Generate the standard command line parser
    parser = argparse.ArgumentParser(description='AugmentA: Patient-specific Augmented Atrial model Generation Tool')
    parser.add_argument('--mesh',
                        type=str,
                        default="",
                        help='full path to mesh with extension')
    parser.add_argument('--closed_surface',
                        type=int,
                        default=0,
                        help='set to 0 if the input surface is open, 1 if the input surface is closed (generate volume)')
    parser.add_argument('--open_orifices',
                        type=int,
                        default=0,
                        help='set to 1 to proceed with the opening of the atrial orifices, 0 otherwise')
    parser.add_argument('--MRI',
                        type=int,
                        default=0,
                        help='set to 1 if the input is derived from tomographic image segmentation, 0 for electroanatomical map')
    parser.add_argument('--use_curvature_to_open',
                        type=int,
                        default=1,
                        help='set to 1 to use the surface curvature to open the atrial orifices, 0 to pick the locations manually')
    parser.add_argument('--SSM_fitting',
                        type=int,
                        default=0,
                        help='set to 1 to proceed with the fitting of a given SSM, 0 otherwise')
    parser.add_argument('--atrium',
                        default="LA",
                        choices=['LA','RA','LA_RA'],
                        help='write LA or RA')
    parser.add_argument('--SSM_file',
                        type=str,
                        default="mesh/meanshape",
                        help='statistical shape model file with extension .h5')
    parser.add_argument('--SSM_basename',
                        type=str,
                        default="mesh/meanshape",
                        help='statistical shape model basename')
    parser.add_argument('--scale',
                        type=int,
                        default=1,
                        help='the pipeline expects an input in mm, use scale to change that')
    parser.add_argument('--resample_input',
                        type=int,
                        default=0,
                        help='1 to resample, 0 otherwise')
    parser.add_argument('--target_mesh_resolution',
                        type=float,
                        default=0.4,
                        help='target mesh resolution in mm')
    parser.add_argument('--normals_outside',
                        type=int,
                        default=1,
                        help='set to 1 if surface normals are pointing outside, 0 otherwise')
    parser.add_argument('--add_bridges',
                        type=int,
                        default=1,
                        help='set to 1 to compute and add interatrial bridges, 0 otherwise')
    parser.add_argument('--ofmt',
                        default='vtu',
                        choices=['vtu','vtk'],
                        help='Output mesh format')
    parser.add_argument('--debug',
                        type=int,
                        default=0,
                        help='set to 1 to debug step by step, 0 otherwise')
    parser.add_argument('--automatedLM', # automated landmark definition for both appendages apexes
                        type=int,
                        default=0,
                        help='set to 1 if a coordinate.dat file containing the appendage apex locations is available, 0 otherwise')
    return parser

def run():

    args = parser().parse_args()
    # delete all files and subfolders with the mesh prefix to avoid that previously generated bridges/surfaces are accidently used in the pipeline
    
    mesh_basename = args.mesh.split('.')[0].split('/')[-1]
    # list all folders
    folderlist = [ f for f in os.listdir(os.path.abspath('')+'/mesh') if not '.' in f and f.startswith(mesh_basename)]
    # list all files
    filelist = [ f for f in os.listdir(os.path.abspath('')+'/mesh') if '.' in f and f.startswith(mesh_basename) and not f == args.mesh.split('/')[-1] and not f.endswith('.dat')]

    for f in filelist:
        os.remove(os.path.join(os.path.abspath('')+'/mesh', f))

    for f in folderlist:
        shutil.rmtree(os.path.join(os.path.abspath('')+'/mesh', f))


    # In case both atria are given process LA first and RA later
    if args.atrium == 'LA_RA':
        args.atrium = 'LA'
        AugmentA(args)
        args.atrium = 'RA'
        AugmentA(args)
    else:
        AugmentA(args)
    
if __name__ == '__main__':
    run()