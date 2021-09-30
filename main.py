import sys
from glob import glob
from shutil import copyfile
import pandas as pd
import os
import vtk
from vtk.util import numpy_support
from vtk.numpy_interface import dataset_adapter as dsa
import argparse
from scipy.spatial import cKDTree
import numpy as np
import collections
import pyvista as pv
import pymeshfix

#import pymeshlab_resample

#EXAMPLE_DIR = os.path.dirname(os.path.realpath(__file__))
from open_orifices_with_curvature import open_orifices_with_curvature
from open_orifices_manually import open_orifices_manually

sys.path.append('Atrial_LDRBM/Generate_Boundaries')
sys.path.append('Atrial_LDRBM/LDRBM/Fiber_LA')

import la_main

#print(EXAMPLE_DIR+'/Atrial_LDRBM/Generate_Boundaries')
import extract_rings

# Generate the standard command line parser
parser = argparse.ArgumentParser(description='AIM-GP')
parser.add_argument('--mesh',
                    type=str,
                    default="",
                    help='full path to mesh with extension')
parser.add_argument('--closed_surface',
                    type=int,
                    default=1,
                    help='set to 0 if the input surface is open, 1 to proceed with the opening of the atrial orifices')
parser.add_argument('--use_curvature_to_open',
                    type=int,
                    default=1,
                    help='set to 1 to use the surface curvature to open the atrial orifices, 0 to pick the locations manually')
parser.add_argument('--SSM_fitting',
                    type=int,
                    default=0,
                    help='set to 1 to proceed with the fitting of a given SSM, 0 otherwise')
parser.add_argument('--atrium',
                    type=str,
                    default="LA",
                    help='write LA or RA')
parser.add_argument('--scale',
                    type=int,
                    default=1,
                    help='set 1 to scale mesh1 to mesh2')
parser.add_argument('--target_mesh_resolution',
                    type=float,
                    default=0.4,
                    help='target mesh resolution in mm')
parser.add_argument('--MRI',
                    type=int,
                    default=0,
                    help='set to 1 if the input is an MRI segmentation')
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
                    help='set to 1 to debug step by step')

args = parser.parse_args()

def run(args):

    if args.closed_surface:
        # Open atrial orifices
        if args.use_curvature_to_open:
            print("Opening atrial orifices using curvature")
            #cut_valves_veins.run(["--mesh",args.mesh,"--atrium",args.atrium,"--scale",str(args.scale),"--MRI",str(args.MRI),"--debug",str(args.debug)])
            open_orifices_with_curvature(args.mesh, args.atrium, args.MRI, scale=args.scale, debug=args.debug)
        else:
            print("Opening atrial orifices manually")
            open_orifices_manually(args.mesh, args.atrium, args.MRI, scale=args.scale, debug=args.debug)
    else:
        # Manually select the appendage apex and extract rings
        print("Manually select the appendage apex and extract rings")
        p = pv.Plotter(notebook=False)
        mesh_from_vtk = pv.PolyData(args.mesh)
        p.add_mesh(mesh_from_vtk, 'r')
        p.add_text('Select the appendage apex and close the window')
        p.enable_point_picking(meshfix.mesh, use_mesh=True)
        p.show()

        if p.picked_point is not None:
            apex = p.picked_point

        model = smart_reader(args.mesh)
        loc = vtk.vtkPointLocator()
        loc.SetDataSet(model)
        loc.BuildLocator()
        apex_id = loc.FindClosestPoint(apex)
        if args.atrium == "LA":
            args.LAA = apex_id
        elif args.atrium == "RA":
            args.RAA = apex_id

        extract_rings.run(["--mesh",args.mesh,"--LAA",str(args.LAA),"--RAA",str(args.RAA)])

    #if args.SSM_fitting:

    extension = args.mesh.split('/')[-1]
    basename = args.mesh[:-len(extension)]
    la_main.run(["--mesh",basename+'/LA_cutted'])

if __name__ == '__main__':
    run(args)