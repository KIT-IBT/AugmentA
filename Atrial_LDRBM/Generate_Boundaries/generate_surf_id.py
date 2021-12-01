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
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import shutil
import os, sys
from glob import glob
from scipy.spatial import cKDTree
import numpy as np

sys.path.append('Atrial_LDRBM/Generate_Boundaries')
from extract_rings import smart_reader

def write_surf_ids(outdir, name, ii):

    fname = outdir+'/ids_{}.vtx'.format(name)
    f = open(fname, 'w')
    if isinstance(ii, int):
        f.write('1\n')
        f.write('extra\n')
        f.write('{}\n'.format(ii))
    else:
        f.write('{}\n'.format(len(ii)))
        f.write('extra\n')
        for i in ii:
            f.write('{}\n'.format(i))
    f.close()

def generate_surf_id(meshname, atrium):
    """The whole model"""
    
    vol = smart_reader(meshname+"_{}_vol.vtk".format(atrium))
    whole_model_points_coordinate = vtk.util.numpy_support.vtk_to_numpy(vol.GetPoints().GetData())

    tree = cKDTree(whole_model_points_coordinate)
    epi_pts = vtk.util.numpy_support.vtk_to_numpy(smart_reader(meshname+'_{}_epi.obj'.format(atrium)).GetPoints().GetData())
    dd, ii = tree.query(epi_pts)

    epi_ids = np.array(ii)
    outdir = meshname+"_{}_vol_surf".format(atrium)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    shutil.copyfile(meshname+"_{}_vol.vtk".format(atrium), outdir+'/{}.vtk'.format(atrium))
    shutil.copyfile(meshname+"_{}_epi_surf/rings_centroids.csv".format(atrium), outdir+'/rings_centroids.csv')

    write_surf_ids(outdir, "EPI", ii)

    dd, ii = tree.query(vtk.util.numpy_support.vtk_to_numpy(smart_reader(meshname+'_{}_endo.obj'.format(atrium)).GetPoints().GetData()))
    ii = np.setdiff1d(ii, epi_ids)

    write_surf_ids(outdir, "ENDO", ii)

    fol_name = meshname+'_{}_epi_surf'.format(atrium)
    
    ids_files = glob(fol_name+'/ids_*')

    for i in range(len(ids_files)):
        ids = np.loadtxt(ids_files[i], skiprows=2, dtype=int)
        name = ids_files[i].split('ids_')[-1][:-4]
        dd, ii = tree.query(epi_pts[ids,:])
        write_surf_ids(outdir, name, ii)

if __name__ == '__main__':
    generate_surf_id()
