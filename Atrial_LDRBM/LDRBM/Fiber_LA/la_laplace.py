#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 15:37:47 2020

@author: tz205
"""
import os
import sys
EXAMPLE_DIR = os.path.dirname(os.path.realpath(__file__))

#os.environ['CARPUTILS_SETTINGS'] = '/home/luca/carputils/settings.yaml'
#os.environ['PATH'] = '/home/luca/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/home/luca/openCARP/_build/bin:/home/luca/carputils/bin:/home/luca/meshtool:/home/luca/meshalyzer:/home/luca/openCARP/_build/bin:/home/luca/carputils/bin:/home/luca/meshtool:/home/luca/meshalyzer'

from carputils.carpio import igb
from carputils import tools
from la_calculate_gradient import la_calculate_gradient
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import numpy as np
from vtk.numpy_interface import dataset_adapter as dsa

def la_laplace(args, job, model):
    meshdir = args.mesh+'_surf/LA'
    surfdir ='{}_surf/'.format(args.mesh)
    parfdir = os.path.join(EXAMPLE_DIR,'Parfiles')

    if args.mesh_type == 'vol':
        ####################################
        # Solver for the phi laplace soluton
        ####################################
        cmd = tools.carp_cmd(parfdir + '/la_lps_phi.par')
        simid = job.ID+'/Lp_phi'
        cmd += ['-simID', simid,
                '-meshname', meshdir,
                '-stimulus[0].vtx_file', surfdir + 'ids_endo',
                '-stimulus[1].vtx_file', surfdir + 'ids_epi']
    
        # Run simulation
        job.carp(cmd)
    #elif args.mesh_type == 'bilayer':
     #   n_pts = mesh.Mesh(meshdir).n_pts()
      #  phi = np.zeros((n_pts,1))
       # phi[np.loadtxt(surfdir + 'ids_epi.vtx', skiprows=2)] = 1
        #igb.IGBFile(simid + "/phie.igb").write(phi)

    #####################################
    # Solver for the ab laplace solution
    #####################################
    cmd = tools.carp_cmd(parfdir + '/la_lps_ab.par')
    simid = job.ID+'/Lp_ab'
    cmd += ['-simID', simid,
            '-meshname', meshdir,
            '-stimulus[0].vtx_file', surfdir + 'ids_LPV',
            '-stimulus[1].vtx_file', surfdir + 'ids_RPV',
            '-stimulus[2].vtx_file', surfdir + 'ids_MV',
            '-stimulus[3].vtx_file', surfdir + 'ids_LAA']

    # Run simulation
    job.carp(cmd)

    #####################################
    # Solver for the v laplace solution
    #####################################
    cmd = tools.carp_cmd(parfdir + '/la_lps_v.par')
    simid = job.ID+'/Lp_v'
    cmd += ['-simID', simid,
            '-meshname', meshdir,
            '-stimulus[0].vtx_file', surfdir + 'ids_LPV',
            '-stimulus[1].vtx_file', surfdir + 'ids_RPV']

    # Run simulation
    job.carp(cmd)

    #####################################
    # Solver for the r laplace solution
    #####################################
    cmd = tools.carp_cmd(parfdir + '/la_lps_r.par')
    simid = job.ID+'/Lp_r'
    cmd += ['-simID', simid,
            '-meshname', meshdir,
            '-stimulus[0].vtx_file', surfdir + 'ids_LPV',
            '-stimulus[1].vtx_file', surfdir + 'ids_RPV',
            '-stimulus[2].vtx_file', surfdir + 'ids_LAA',
            '-stimulus[3].vtx_file', surfdir + 'ids_MV']

    # Run simulation
    job.carp(cmd)
    
    #####################################
    # Solver for the r2 laplace solution
    #####################################
    cmd = tools.carp_cmd(parfdir + '/la_lps_r2.par')
    simid = job.ID+'/Lp_r2'
    cmd += ['-simID', simid,
            '-meshname', meshdir,
            '-stimulus[0].vtx_file', surfdir + 'ids_LPV',
            '-stimulus[1].vtx_file', surfdir + 'ids_RPV',
            '-stimulus[2].vtx_file', surfdir + 'ids_MV']

    # Run simulation
    job.carp(cmd)

    """
    generate .vtu files that contain the result of laplace solution as point/cell data
    """
    
    meshNew = dsa.WrapDataObject(model)
    
    name_list = ['r', 'r2', 'v', 'ab']
    if args.mesh_type == 'vol':
        name_list = ['phi', 'r', 'r2', 'v', 'ab']
    for var in name_list:
        data = igb.IGBFile(job.ID+"/Lp_" + str(var) + "/phie.igb").data()
        # add the vtk array to model
        meshNew.PointData.append(data, "phie_"+str(var))
        
    if args.debug == 1:
        # write
        simid = job.ID+"/Laplace_Result"
        try:
            os.makedirs(simid)
        except OSError:
            print ("Creation of the directory %s failed" % simid)
        else:
            print ("Successfully created the directory %s " % simid)
        if args.mesh_type == "vol":
            writer = vtk.vtkUnstructuredGridWriter()
        else:
            writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(simid+"/LA_with_laplace.vtk")
        writer.SetInputData(meshNew.VTKObject)
        writer.Write()
    """
    calculate the gradient
    """
    output = la_calculate_gradient(args, meshNew.VTKObject, job)
    
    return output

def laplace_0_1(args, job, model, name1,name2, outname):
    
    meshdir = args.mesh+'_surf/LA'
    surfdir ='{}_surf/'.format(args.mesh)
    parfdir = os.path.join(EXAMPLE_DIR,'Parfiles')
    #####################################
    # Solver for the ab laplace solution
    #####################################
    cmd = tools.carp_cmd(parfdir + '/la_lps_phi.par')
    simid = job.ID+'/Lp_ab'
    cmd += ['-simID', simid,
            '-meshname', meshdir,
            '-stimulus[0].vtx_file', surfdir + 'ids_{}'.format(name1),
            '-stimulus[1].vtx_file', surfdir + 'ids_{}'.format(name2)]
    
    if name1 == "4":
        cmd = tools.carp_cmd(parfdir + '/la_lps_phi_0_1_4.par')
        simid = job.ID+'/Lp_ab'
        cmd += ['-simID', simid,
            '-meshname', meshdir,
            '-stimulus[0].vtx_file', surfdir + 'ids_LSPV',
            '-stimulus[1].vtx_file', surfdir + 'ids_LIPV',
            '-stimulus[2].vtx_file', surfdir + 'ids_RSPV',
            '-stimulus[3].vtx_file', surfdir + 'ids_RIPV']
    # Run simulation
    job.carp(cmd)
    
    meshNew = dsa.WrapDataObject(model)
    data = igb.IGBFile(job.ID+"/Lp_ab/phie.igb").data()
    meshNew.PointData.append(data, outname)
    
    if args.debug == 1:
        # write
        simid = job.ID+"/Laplace_Result"
        try:
            os.makedirs(simid)
        except OSError:
            print ("Creation of the directory %s failed" % simid)
        else:
            print ("Successfully created the directory %s " % simid)

        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(simid+"/LA_with_laplace.vtk")
        writer.SetInputData(meshNew.VTKObject)
        writer.Write()
    # if args.debug == 1:

    #     simid = job.ID+"/Laplace_Result"
    #     try:
    #         os.makedirs(simid)
    #     except OSError:
    #         print ("Creation of the directory %s failed" % simid)
    #     else:
    #         print ("Successfully created the directory %s " % simid)
    #     if args.mesh_type == "vol":
    #         writer = vtk.vtkUnstructuredGridWriter()
    #     else:
    #         writer = vtk.vtkUnstructuredGridWriter()
    #     writer.SetFileName(simid+"/LA_with_LAA_bb.vtk")
    #     writer.SetInputData(meshNew.VTKObject)
    #     writer.Write()
    
    return meshNew.VTKObject