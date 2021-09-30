#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 15:37:47 2020

@author: tz205
"""
import os
#os.environ['CARPUTILS_SETTINGS'] = '/Volumes/bordeaux/IBT/openCARP/.config/carputils/settings.yaml'
#os.environ['PATH'] = '/Volumes/bordeaux/IBT/bin/macosx:/Volumes/bordeaux/IBT/src/CardioMechanics/trunk/src/Scripts:/Volumes/bordeaux/IBT/bin/macosx:/Volumes/bordeaux/IBT/pl:/Volumes/bordeaux/IBT/python:/Volumes/bordeaux/IBT/thirdparty/macosx/bin:/Volumes/bordeaux/IBT/thirdparty/macosx/openMPI-64bit/bin:/opt/X11/bin:/Applications/MATLAB_R2020a.app/bin/:/opt/local/bin:/opt/local/sbin:/usr/bin:/bin:/usr/sbin:/sbin:/Volumes/bordeaux/IBT/openCARP/bin:/Volumes/bordeaux/IBT/openCARP/bin:/usr/local/bin'

from carputils import tools
from ra_calculate_gradient import ra_calculate_gradient
import vtk
from carputils.carpio import igb
from vtk.numpy_interface import dataset_adapter as dsa

EXAMPLE_DIR = os.path.dirname(__file__)

def ra_laplace(args, job, model):
    meshdir = args.mesh+'_surf/RA'
    surfdir ='{}_surf/'.format(args.mesh)
    parfdir = os.path.join(EXAMPLE_DIR,'Parfiles')
    
    if args.mesh_type == 'vol':
        ####################################
        #Solver for the phi laplace soluton
        ####################################
        cmd = tools.carp_cmd(parfdir+'/ra_lps_phi.par')
        simid = job.ID+'/Lp_phi'
        cmd += [ '-simID', simid,
                  '-meshname', meshdir,
                  '-stimulus[0].vtx_file', surfdir+'ids_endo',
                  '-stimulus[1].vtx_file', surfdir+'ids_epi']
        
        #Run simulation
        job.carp(cmd)
    

    #####################################
    #Solver for the ab laplace soluton
    #####################################
    cmd = tools.carp_cmd(parfdir+'/ra_lps_ab.par')
    simid = job.ID+'/Lp_ab'
    cmd += [ '-simID', simid,
              '-meshname', meshdir,
              '-stimulus[0].vtx_file', surfdir+'ids_SVC',
              '-stimulus[1].vtx_file', surfdir+'ids_IVC',
              '-stimulus[2].vtx_file', surfdir+'ids_TV_S',
              '-stimulus[3].vtx_file', surfdir+'ids_TV_F',
              '-stimulus[4].vtx_file', surfdir+'ids_RAA']
        
    #Run simulation
    job.carp(cmd)
    
    #####################################
    #Solver for the v laplace soluton
    #####################################
    cmd = tools.carp_cmd(parfdir+'/ra_lps_v.par')
    simid = job.ID+'/Lp_v'
    cmd += [ '-simID', simid,
              '-meshname', meshdir,
              '-stimulus[0].vtx_file', surfdir+'ids_SVC',
              '-stimulus[1].vtx_file', surfdir+'ids_RAA',
              '-stimulus[2].vtx_file', surfdir+'ids_IVC']
        
    #Run simulation
    job.carp(cmd)
    
    #####################################
    #Solver for the v2 laplace soluton
    #####################################
    cmd = tools.carp_cmd(parfdir+'/ra_lps_phi.par')
    simid = job.ID+'/Lp_v2'
    cmd += [ '-simID', simid,
              '-meshname', meshdir,
              '-stimulus[0].vtx_file', surfdir+'ids_IVC',
              '-stimulus[1].vtx_file', surfdir+'ids_RAA']
        
    #Run simulation
    job.carp(cmd)
    
    if args.mesh_type == 'vol':
        #####################################
        #Solver for the r laplace soluton
        #####################################
        cmd = tools.carp_cmd(parfdir+'/ra_lps_r_vol.par')
        simid = job.ID+'/Lp_r'
        cmd += [ '-simID', simid,
                  '-meshname', meshdir,
                  '-stimulus[0].vtx_file', surfdir+'ids_TOP_ENDO',
                  '-stimulus[1].vtx_file', surfdir+'ids_top_epi',
                  '-stimulus[2].vtx_file', surfdir+'ids_TV_F',
                  '-stimulus[3].vtx_file', surfdir+'ids_TV_S']
            
        #Run simulation
        job.carp(cmd)
    
    else:
        cmd = tools.carp_cmd(parfdir+'/ra_lps_r.par')
        simid = job.ID+'/Lp_r'
        cmd += [ '-simID', simid,
                  '-meshname', meshdir,
                  '-stimulus[0].vtx_file', surfdir+'ids_TOP_ENDO',
                  '-stimulus[1].vtx_file', surfdir+'ids_TV_F',
                  '-stimulus[2].vtx_file', surfdir+'ids_TV_S']
            
        #Run simulation
        job.carp(cmd)
        
    #####################################
    #Solver for the w laplace soluton
    #####################################
    cmd = tools.carp_cmd(parfdir+'/ra_lps_w.par')
    simid = job.ID+'/Lp_w'
    cmd += [ '-simID', simid,
              '-meshname', meshdir,
              '-stimulus[0].vtx_file', surfdir+'ids_TV_S',
              '-stimulus[1].vtx_file', surfdir+'ids_TV_F']
        
    #Run simulation
    job.carp(cmd)

    """
    generate .vtu files that contain the result of laplace solution as point/cell data
    """
    meshNew = dsa.WrapDataObject(model)

    name_list = ['r','v','v2','ab','w']
    if args.mesh_type == 'vol':
        name_list = ['phi','r','v','v2','ab','w']
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
        writer.SetFileName(simid+"/RA_with_laplace.vtk")
        writer.SetInputData(meshNew.VTKObject)
        writer.Write()
    """
    calculate the gradient
    """
    output = ra_calculate_gradient(args, meshNew.VTKObject, job)
    
    return output
    