import sys
from glob import glob
from shutil import copyfile
import pandas as pd
import os
from string import Template
import argparse
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

EXAMPLE_DIR = os.path.dirname(os.path.realpath(__file__))
from open_orifices_with_curvature import open_orifices_with_curvature
from open_orifices_manually import open_orifices_manually
from prealign_meshes import prealign_meshes
from getmarks import get_landmarks
from create_SSM_instance import create_SSM_instance
from resample_surf_mesh import resample_surf_mesh

sys.path.append('Atrial_LDRBM/Generate_Boundaries')
sys.path.append('Atrial_LDRBM/LDRBM/Fiber_LA')

import la_main

#print(EXAMPLE_DIR+'/Atrial_LDRBM/Generate_Boundaries')
from extract_rings import label_atrial_orifices
pv.set_plot_theme('dark')
n_cpu=os.cpu_count()
if not n_cpu % 2:
    n_cpu = int(n_cpu/2)

def parser():
    # Generate the standard command line parser
    parser = argparse.ArgumentParser(description='PI-AGENT: Personalized Integrated Atria Generation Tool')
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
                        help='set 1 to scale mesh1 to mesh2')
    parser.add_argument('--target_mesh_resolution',
                        type=float,
                        default=0.4,
                        help='target mesh resolution in mm')
    parser.add_argument('--MRI',
                        type=int,
                        default=0,
                        help='set to 1 if the input is an MRI segmentation')
    parser.add_argument('--mesh_type',
                        default='bilayer',
                        choices=['vol',
                                 'bilayer'],
                        help='Mesh type')
    parser.add_argument('--debug',
                        type=int,
                        default=0,
                        help='set to 1 to debug step by step, 0 otherwise')
    return parser

def run():

    args = parser().parse_args()

    args.SSM_file = os.path.abspath(args.SSM_file)
    args.SSM_basename = os.path.abspath(args.SSM_basename)
    args.mesh = os.path.abspath(args.mesh)
    
    # if args.closed_surface:
    #     # Open atrial orifices
    #     if args.use_curvature_to_open:
    #         # Opening atrial orifices using curvature
    #         print("Opening atrial orifices using curvature")
    #         open_orifices_with_curvature(args.mesh, args.atrium, args.MRI, scale=args.scale, debug=args.debug)
    #     else:
    #         # Opening atrial orifices manually
    #         print("Opening atrial orifices manually")
    #         open_orifices_manually(args.mesh, args.atrium, args.MRI, scale=args.scale, debug=args.debug)
    # else:
    #     if args.SSM_fitting:
    #         # Manually select the appendage apex and extract rings, these are going to be used to compute the landmarks for the fitting
    #         print("Manually select the appendage apex and extract rings")
            
    #         p = pv.Plotter(notebook=False)
    #         mesh_from_vtk = pv.PolyData(args.mesh)
    #         p.add_mesh(mesh_from_vtk, 'r')
    #         p.add_text('Select the appendage apex and close the window',position='lower_left')
    #         p.enable_point_picking(mesh_from_vtk, use_mesh=True)
    #         p.show()

    #         if p.picked_point is not None:
    #             apex = p.picked_point
    #         else:
    #             raise ValueError("Please select the appendage apex")

    #         tree = cKDTree(mesh_from_vtk.points.astype(np.double))
    #         dd, apex_id = tree.query(apex)

    #         LAA = ""
    #         RAA = ""
    #         if args.atrium == "LA":
    #             LAA = apex_id
    #         elif args.atrium == "RA":
    #             RAA = apex_id

    #         label_atrial_orifices(args.mesh,LAA,RAA)
    #     else:
    #         # Atrial orifices already open
    #         print("Atrial orifices already open, proceed to resampling")

    extension = args.mesh.split('/')[-1]
    mesh_dir = args.mesh[:-len(extension)]

    if args.SSM_fitting:

        # Generate SSM landmarks if not present
        if not os.path.isfile(args.SSM_basename+'_surf/landmarks.json'):
            label_atrial_orifices(args.SSM_basename,6329,21685)  # 6329 LAA apex id and 21685 RAA apex id in meanshape from Nagel et al. 2020
            get_landmarks(args.SSM_basename, 0, 1)

        # Rigid alignment of target mesh to SSM mean instance
        prealign_meshes(mesh_dir+'LA_cutted', args.SSM_basename, args.atrium, 0)
        # Landmarks generation
        get_landmarks(mesh_dir+'LA_cutted', 1, 1)

        # Create Scalismo ICP-GP fitting algorithm script 
        with open('Registration_ICP_GP_template.txt','r') as f:
            lines = f.readlines()
        lines = ''.join(lines)

        temp_obj = Template(lines)
        SSM_fit_file = temp_obj.substitute(SSM_file=args.SSM_file,SSM_dir=args.SSM_basename+'_surf',target_dir=mesh_dir+'LA_cutted_surf')
        with open(mesh_dir+'LA_cutted_surf'+'/Registration_ICP_GP.txt','w') as f:
            f.write(SSM_fit_file)

        # Create SSM instance
        if os.path.isfile(mesh_dir+'LA_cutted_surf/coefficients.txt'):
            create_SSM_instance(args.SSM_file+'.h5', mesh_dir+'LA_cutted_surf/coefficients.txt',mesh_dir+'LA_cutted_surf/LA_fit.obj')
        else:
            raise ValueError("Create coefficients.txt file including the SSM coefficients from Scalismo")

        # Resample surface mesh with given target average edge length
        resample_surf_mesh(mesh_dir+'LA_cutted_surf/LA_fit', target_mesh_resolution=0.4, find_apex_with_curv=1, scale=1)

        # Label atrial orifices using LAA id found in the resampling algorithm
        df = pd.read_csv('{}_mesh_data.csv'.format(mesh_dir+'LA_cutted_surf/LA_fit'))
        label_atrial_orifices('{}_res.obj'.format(mesh_dir+'LA_cutted_surf/LA_fit'),LAA_id=int(df["LAA_id"]))

        # Atrial region annotation and fiber generation using LDRBM
        la_main.run(["--mesh",'{}_res'.format(mesh_dir+'LA_cutted'), "--np", str(n_cpu)])

    else:

        # #Convert mesh from vtk to obj
        # meshin = pv.read('{}.vtk'.format(mesh_dir+'LA_cutted'))
        # pv.save_meshio('{}.obj'.format(mesh_dir+'LA_cutted'), meshin, "obj")

        # # Resample surface mesh with given target average edge length
        # resample_surf_mesh('{}'.format(mesh_dir+'LA_cutted'), target_mesh_resolution=0.4, find_apex_with_curv=1, scale=1)

        # # Label atrial orifices using LAA id found in the resampling algorithm
        # df = pd.read_csv('{}_mesh_data.csv'.format(mesh_dir+'LA_cutted'))
        # label_atrial_orifices('{}_res.obj'.format(mesh_dir+'LA_cutted'),LAA_id=int(df["LAA_id"]))

        # Atrial region annotation and fiber generation using LDRBM
        la_main.run(["--mesh",'{}_res'.format(mesh_dir+'LA_cutted'), "--np", str(n_cpu)])

        geom = pv.Line()
        bil = pv.read('{}_res_fibers/result_LA/LA_bilayer_with_fiber.vtk'.format(mesh_dir+'LA_cutted'))
        mask = bil['elemTag'] >99
        bil['elemTag'][mask] = 0
        mask = bil['elemTag'] >80
        bil['elemTag'][mask] = 20
        mask = bil['elemTag'] >10
        bil['elemTag'][mask] = bil['elemTag'][mask]-10
        fibers = bil.glyph(orient="fiber",factor=0.5,geom=geom, scale="elemTag")
        p = pv.Plotter(notebook=False)
        p.add_mesh(bil, scalars="elemTag",show_scalar_bar=False,cmap='tab20')
        p.add_mesh(fibers,show_scalar_bar=False,cmap='tab20',line_width=10,render_lines_as_tubes=True)

        # endo = pv.read('{}_res_fibers/result_LA/LA_endo_with_fiber.vtk'.format(mesh_dir+'LA_cutted'))
        # endo_fibers = endo.glyph(orient="fiber",factor=0.5,geom=geom, scale="elemTag")
        # epi = pv.read('{}_res_fibers/result_LA/LA_epi_with_fiber.vtk'.format(mesh_dir+'LA_cutted'))
        # mask = epi['elemTag'] >80
        # epi['elemTag'][mask] = 20
        # epi_fibers = epi.glyph(orient="fiber",factor=0.3,geom=geom, scale="elemTag")
        # p = pv.Plotter(notebook=False)
        # p.add_mesh(endo, scalars="elemTag",show_scalar_bar=False,cmap='tab20')
        # p.add_mesh(epi, scalars="elemTag",show_scalar_bar=False,cmap='tab20')
        # p.add_mesh(endo_fibers,show_scalar_bar=False,cmap='tab20',line_width=10,render_lines_as_tubes=True)
        # p.add_mesh(epi_fibers,show_scalar_bar=False,cmap='tab20',line_width=10,render_lines_as_tubes=True)
        p.show()

if __name__ == '__main__':
    run()