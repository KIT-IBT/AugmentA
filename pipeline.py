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
from glob import glob
from shutil import copyfile
import pandas as pd
import os
from string import Template
import argparse

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

sys.path.append('standalones')
from open_orifices_with_curvature import open_orifices_with_curvature
from open_orifices_manually import open_orifices_manually
from prealign_meshes import prealign_meshes
from getmarks import get_landmarks
from create_SSM_instance import create_SSM_instance
from resample_surf_mesh import resample_surf_mesh

import vtk
from vtk.util import numpy_support
from vtk.numpy_interface import dataset_adapter as dsa

sys.path.append('Atrial_LDRBM/Generate_Boundaries')
sys.path.append('Atrial_LDRBM/LDRBM/Fiber_LA')
sys.path.append('Atrial_LDRBM/LDRBM/Fiber_RA')
import la_main
import ra_main
from extract_rings import label_atrial_orifices
from extract_rings import smart_reader
from extract_rings_TOP_epi_endo import label_atrial_orifices_TOP_epi_endo
from separate_epi_endo import separate_epi_endo
from generate_mesh import generate_mesh
from generate_surf_id import generate_surf_id

pv.set_plot_theme('dark')
n_cpu = os.cpu_count()
if not n_cpu % 2:
    n_cpu = int(n_cpu / 2)


def AugmentA(args):
    args.SSM_file = os.path.abspath(args.SSM_file)
    args.SSM_basename = os.path.abspath(args.SSM_basename)
    args.mesh = os.path.abspath(args.mesh)

    extension = args.mesh.split('/')[-1]
    mesh_dir = args.mesh[:-len(extension)]

    extension = args.mesh.split('.')[-1]
    meshname = args.mesh[:-(len(extension) + 1)]

    if args.closed_surface:
        separate_epi_endo(args.mesh, args.atrium)
        meshname_old = str(meshname)
        meshname = meshname_old + "_{}_epi".format(args.atrium)
    else:

        if args.open_orifices:
            # Open atrial orifices
            if args.use_curvature_to_open:
                # Opening atrial orifices using curvature
                print("Opening atrial orifices using curvature")
                apex_id = open_orifices_with_curvature(args.mesh, args.atrium, args.MRI, scale=args.scale,
                                                       debug=args.debug)
            else:
                # Opening atrial orifices manually
                print("Opening atrial orifices manually")
                apex_id = open_orifices_manually(args.mesh, args.atrium, args.MRI, scale=args.scale, debug=args.debug)
            meshname = mesh_dir + args.atrium + "_cutted"
        else:
            if not args.resample_input and args.find_appendage:  # don't open orifices and don´t resample, find appendage
                # Manually select the appendage apex and extract rings, these are going to be used to compute the landmarks for the fitting if SSM is selected
                print("Manually select the appendage apex and extract rings")

                mesh_data = dict()

                # Make sure that the mesh is a Polydata
                geo_filter = vtk.vtkGeometryFilter()
                geo_filter.SetInputData(smart_reader(args.mesh))
                geo_filter.Update()
                polydata = geo_filter.GetOutput()

                mesh_from_vtk = pv.PolyData(polydata)
                p = pv.Plotter(notebook=False)
                p.add_mesh(mesh_from_vtk, 'r')
                p.add_text('Select the appendage apex and close the window', position='lower_left')
                p.enable_point_picking(mesh_from_vtk, use_picker=True)
                p.show()

                if p.picked_point is not None:
                    apex = p.picked_point
                else:
                    raise ValueError("Please select the appendage apex")
                p.close()

                tree = cKDTree(mesh_from_vtk.points.astype(np.double))
                dd, apex_id = tree.query(apex)

                LAA = ""
                RAA = ""

                if args.atrium == "LA":
                    LAA = apex_id
                    mesh_data[args.atrium + "A_id"] = [apex_id]
                elif args.atrium == "RA":
                    RAA = apex_id
                    mesh_data[args.atrium + "A_id"] = [apex_id]
                elif args.atrium == "LA_RA":
                    LAA = apex_id
                    mesh_data["LAA_id"] = [apex_id]

                    mesh_from_vtk = pv.PolyData(polydata)
                    p = pv.Plotter(notebook=False)
                    p.add_mesh(mesh_from_vtk, 'r')
                    p.add_text('Select the RA appendage apex and close the window', position='lower_left')
                    p.enable_point_picking(mesh_from_vtk, use_picker=True)
                    p.show()

                    if p.picked_point is not None:
                        apex = p.picked_point
                    else:
                        raise ValueError("Please select the appendage apex")
                    p.close()

                    tree = cKDTree(mesh_from_vtk.points.astype(np.double))
                    dd, apex_id = tree.query(apex)
                    RAA = apex_id

                    mesh_data["RAA_id"] = [apex_id]

                fname = '{}_mesh_data.csv'.format(meshname)
                df = pd.DataFrame(mesh_data)
                df.to_csv(fname, float_format="%.2f", index=False)

                # print("Labelling atrial orifices")
                # label_atrial_orifices(args.mesh,LAA,RAA)

                # Label atrial orifices using apex id found in the resampling algorithm
                # df = pd.read_csv('{}_mesh_data.csv'.format(meshname))

                # Atrial orifices already open
                print("Atrial orifices already open")
                processed_mesh = meshname

    if args.SSM_fitting and not args.closed_surface:

        # Generate SSM landmarks if not present
        if not os.path.isfile(args.SSM_basename + '_surf/landmarks.json'):
            label_atrial_orifices(args.SSM_basename, 6329,
                                  21685)  # 6329 LAA apex id and 21685 RAA apex id in meanshape from Nagel et al. 2020
            get_landmarks(args.SSM_basename, 0, 1)

        # Rigid alignment of target mesh to SSM mean instance
        prealign_meshes(mesh_dir + args.atrium + '_cutted', args.SSM_basename, args.atrium, 0)
        # Landmarks generation
        get_landmarks(mesh_dir + args.atrium + '_cutted', 1, 1)

        # Create Scalismo ICP-GP fitting algorithm script 
        with open('template/Registration_ICP_GP_template.txt', 'r') as f:
            lines = f.readlines()
        lines = ''.join(lines)

        temp_obj = Template(lines)
        SSM_fit_file = temp_obj.substitute(SSM_file=args.SSM_file, SSM_dir=args.SSM_basename + '_surf',
                                           target_dir=mesh_dir + args.atrium + '_cutted_surf')
        with open(mesh_dir + args.atrium + '_cutted_surf' + '/Registration_ICP_GP.txt', 'w') as f:
            f.write(SSM_fit_file)

        # Create SSM instance
        if os.path.isfile(mesh_dir + args.atrium + '_cutted_surf/coefficients.txt'):
            create_SSM_instance(args.SSM_file + '.h5', mesh_dir + args.atrium + '_cutted_surf/coefficients.txt',
                                mesh_dir + args.atrium + '_cutted_surf/' + args.atrium + '_fit.obj')
        else:
            raise ValueError("Create coefficients.txt file including the SSM coefficients from Scalismo")

        if args.resample_input:
            # Resample surface mesh with given target average edge length
            resample_surf_mesh(mesh_dir + args.atrium + '_cutted_surf/' + args.atrium + '_fit',
                               target_mesh_resolution=args.target_mesh_resolution, find_apex_with_curv=1,
                               scale=args.scale, apex_id=apex_id, atrium=args.atrium)
            processed_mesh = mesh_dir + args.atrium + '_cutted_surf/' + args.atrium + '_fit_res'
        else:
            processed_mesh = mesh_dir + args.atrium + '_cutted_surf/' + args.atrium + '_fit'

        # Label atrial orifices using apex id found in the resampling algorithm
        df = pd.read_csv('{}_mesh_data.csv'.format(mesh_dir + args.atrium + '_cutted_surf/' + args.atrium + '_fit'))
        if args.atrium == "LA":
            label_atrial_orifices(processed_mesh + 'obj', LAA_id=int(df[args.atrium + "A_id"]))
            # Atrial region annotation and fiber generation using LDRBM
            la_main.run(
                ["--mesh", processed_mesh, "--np", str(n_cpu), "--normals_outside", str(args.normals_outside), "--ofmt",
                 args.ofmt, "--debug", str(args.debug), "--overwrite-behaviour", "append"])
        elif args.atrium == "RA":
            label_atrial_orifices(processed_mesh + 'obj', RAA_id=int(df[args.atrium + "A_id"]))
            # Atrial region annotation and fiber generation using LDRBM
            ra_main.run(
                ["--mesh", processed_mesh, "--np", str(n_cpu), "--normals_outside", str(args.normals_outside), "--ofmt",
                 args.ofmt, "--debug", str(args.debug), "--overwrite-behaviour", "append"])

    elif not args.SSM_fitting:

        if args.resample_input and args.find_appendage:
            print("Resample surface mesh with given target average edge length")
            # Make sure there is an .obj mesh file

            # Convert mesh from vtk to obj
            meshin = pv.read('{}.vtk'.format(meshname))
            pv.save_meshio('{}.obj'.format(meshname), meshin, "obj")

            # if args.atrium =='LA_RA':
            apex_id = -1  # Find new location with resampled mesh

            # Here finds the LAA_id and/or RAA_id for the remeshed geometry
            resample_surf_mesh('{}'.format(meshname), target_mesh_resolution=args.target_mesh_resolution,
                               find_apex_with_curv=0, scale=args.scale, apex_id=apex_id, atrium=args.atrium)
            processed_mesh = '{}_res'.format(meshname)

            # Convert mesh from ply to obj
            meshin = pv.read('{}.ply'.format(processed_mesh))
            pv.save_meshio('{}.obj'.format(processed_mesh), meshin, "obj")

            # p = pv.Plotter(notebook=False)
            #
            # if args.use_curvature_to_open:
            #     print("Propose appendage apex location using surface curvature")
            #     os.system("meshtool query curvature -msh={}.obj -size={}".format(meshname, 30*args.scale))
            #     curv = np.loadtxt('{}.curv.dat'.format(meshname))
            #     mesh_curv = pv.read('{}.obj'.format(meshname))
            #
            #     apex = mesh_curv.points[np.argmax(curv),:]
            #
            #     point_cloud = pv.PolyData(apex)
            #
            #     p.add_mesh(point_cloud, color='w', point_size=30.*args.scale, render_points_as_spheres=True)
            #
            # p.add_mesh(meshin,color='r')
            # p.enable_point_picking(meshin, use_mesh=True)
            # p.add_text('Select the appendage apex and close the window',position='lower_left')
            #
            # p.show()
            #
            # if p.picked_point is not None:
            #     apex = p.picked_point
            #
            # print("Apex coordinates: ", apex)
            # p.close()
            # mesh_data = dict()
            # tree = cKDTree(meshin.points.astype(np.double))
            # dist, apex_id = tree.query(apex)
            #
            # mesh_data[args.atrium+"A_id"] = [apex_id]
            #
            # fname = '{}_mesh_data.csv'.format(meshname)
            # df = pd.DataFrame(mesh_data)
            # df.to_csv(fname, float_format="%.2f", index=False)

        elif not args.resample_input and not args.find_appendage:  # do not resample and do not find appendage

            processed_mesh = meshname  # Provide mesh with _res in  the name

            # if not args.closed_surface:
            #     #Convert mesh from vtk to obj
            #     meshin = pv.read('{}.vtk'.format(meshname))
            #     pv.save_meshio('{}.obj'.format(meshname), meshin, "obj")
            # else:
            #     meshin = pv.read('{}.obj'.format(meshname))

        # Label atrial orifices using apex id found in the resampling algorithm
        df = pd.read_csv('{}_mesh_data.csv'.format(processed_mesh))

        if args.atrium == "LA_RA":
            if not os.path.exists(processed_mesh + '.obj'):
                meshin = pv.read('{}.vtk'.format(processed_mesh))
                pv.save_meshio('{}.obj'.format(processed_mesh), meshin, "obj")

            label_atrial_orifices(processed_mesh + '.obj', LAA_id=int(df["LAA_id"]),
                                  RAA_id=int(df["RAA_id"]))  # Label both
            # Do the LA first
            la_main.run(
                ["--mesh", processed_mesh, "--np", str(n_cpu), "--normals_outside", str(args.normals_outside), "--ofmt",
                 args.ofmt, "--debug", str(args.debug), "--overwrite-behaviour", "append"])
            args.atrium = "RA"
            ra_main.run(
                ["--mesh", processed_mesh, "--np", str(n_cpu), "--normals_outside", str(args.normals_outside), "--ofmt",
                 args.ofmt, "--debug", str(args.debug), "--overwrite-behaviour", "append"])
            args.atrium = "LA_RA"
            os.system("meshtool convert -imsh={} -ifmt=carp_txt -omsh={} -ofmt=carp_txt -scale={}".format(
                '{}_fibers/result_RA/{}_bilayer_with_fiber'.format(processed_mesh, args.atrium),
                '{}_fibers/result_RA/{}_bilayer_with_fiber_um'.format(processed_mesh, args.atrium), 1000 * args.scale))
            os.system("meshtool convert -imsh={} -ifmt=carp_txt -omsh={} -ofmt=vtk".format(
                '{}_fibers/result_RA/{}_bilayer_with_fiber_um'.format(processed_mesh, args.atrium),
                '{}_fibers/result_RA/{}_bilayer_with_fiber_um'.format(processed_mesh, args.atrium)))
        if args.atrium == "LA":
            label_atrial_orifices(processed_mesh + '.obj', LAA_id=int(df[args.atrium + "A_id"]))
            # Atrial region annotation and fiber generation using LDRBM
            if args.closed_surface:
                generate_mesh(meshname_old + '_{}'.format(args.atrium))
                generate_surf_id(meshname_old, args.atrium)
                processed_mesh = meshname_old + "_{}_vol".format(args.atrium)
                la_main.run(
                    ["--mesh", processed_mesh, "--np", str(n_cpu), "--normals_outside", str(0), "--mesh_type", "vol",
                     "--ofmt", args.ofmt, "--debug", str(args.debug), "--overwrite-behaviour", "append"])
            else:
                la_main.run(
                    ["--mesh", processed_mesh, "--np", str(n_cpu), "--normals_outside", str(args.normals_outside),
                     "--ofmt", args.ofmt, "--debug", str(args.debug), "--overwrite-behaviour", "append"])
                os.system("meshtool convert -imsh={} -ifmt=carp_txt -omsh={} -ofmt=carp_txt -scale={}".format(
                    '{}_fibers/result_{}/{}_bilayer_with_fiber'.format(processed_mesh, args.atrium, args.atrium),
                    '{}_fibers/result_{}/{}_bilayer_with_fiber_um'.format(processed_mesh, args.atrium, args.atrium),
                    1000 * args.scale))
                os.system("meshtool convert -imsh={} -ifmt=carp_txt -omsh={} -ofmt=vtk".format(
                    '{}_fibers/result_{}/{}_bilayer_with_fiber_um'.format(processed_mesh, args.atrium, args.atrium),
                    '{}_fibers/result_{}/{}_bilayer_with_fiber_um'.format(processed_mesh, args.atrium, args.atrium)))

        elif args.atrium == "RA":
            # Atrial region annotation and fiber generation using LDRBM
            if args.closed_surface:
                label_atrial_orifices_TOP_epi_endo(processed_mesh + '.obj', RAA_id=int(df[args.atrium + "A_id"]))
                generate_mesh(meshname_old + '_{}'.format(args.atrium))
                generate_surf_id(meshname_old, args.atrium)
                processed_mesh = meshname_old + "_{}_vol".format(args.atrium)
                ra_main.run(
                    ["--mesh", processed_mesh, "--np", str(n_cpu), "--normals_outside", str(0), "--mesh_type", "vol",
                     "--ofmt", args.ofmt, "--debug", str(args.debug), "--overwrite-behaviour", "append"])
            else:
                label_atrial_orifices(processed_mesh + '.obj', RAA_id=int(df[args.atrium + "A_id"]))
                ra_main.run(
                    ["--mesh", processed_mesh, "--np", str(n_cpu), "--normals_outside", str(args.normals_outside),
                     "--ofmt", args.ofmt, "--debug", str(args.debug), "--overwrite-behaviour", "append"])

    if args.debug:
        if args.closed_surface:
            bil = pv.read(
                '{}_fibers/result_{}/{}_vol_with_fiber.{}'.format(processed_mesh, args.atrium, args.atrium, args.ofmt))
        else:
            if args.atrium == 'LA_RA':
                bil = pv.read(
                    '{}_fibers/result_RA/{}_bilayer_with_fiber.{}'.format(processed_mesh, args.atrium,
                                                                          args.ofmt))
            else:
                bil = pv.read(
                    '{}_fibers/result_{}/{}_bilayer_with_fiber.{}'.format(processed_mesh, args.atrium, args.atrium,
                                                                          args.ofmt))
            geom = pv.Line()
        mask = bil['elemTag'] > 99
        bil['elemTag'][mask] = 0
        mask = bil['elemTag'] > 80
        bil['elemTag'][mask] = 20
        mask = bil['elemTag'] > 10
        bil['elemTag'][mask] = bil['elemTag'][mask] - 10
        mask = bil['elemTag'] > 50
        bil['elemTag'][mask] = bil['elemTag'][mask] - 50
        p = pv.Plotter(notebook=False)
        if not args.closed_surface:
            fibers = bil.glyph(orient="fiber", factor=0.5, geom=geom, scale="elemTag")
            p.add_mesh(fibers, show_scalar_bar=False, cmap='tab20', line_width=10, render_lines_as_tubes=True)
        p.add_mesh(bil, scalars="elemTag", show_scalar_bar=False, cmap='tab20')
        p.show()
        p.close()
