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
import shutil
import sys
from glob import glob

import numpy as np
from scipy.spatial import cKDTree

from vtk_opencarp_helper_methods.vtk_methods.converters import vtk_to_numpy
from vtk_opencarp_helper_methods.vtk_methods.exporting import write_to_vtx

sys.path.append('Atrial_LDRBM/Generate_Boundaries')
from extract_rings import smart_reader


def write_surf_ids(outdir, name, ii):
    write_to_vtx(outdir + f'/ids_{name}.vtx', ii)


def generate_surf_id(meshname, atrium, resampled=False):
    """The whole model"""

    vol = smart_reader(meshname + f"_{atrium}_vol.vtk")
    whole_model_points_coordinate = vtk_to_numpy(vol.GetPoints().GetData())

    tree = cKDTree(whole_model_points_coordinate)
    epi_pts = vtk_to_numpy(
        smart_reader(meshname + f'_{atrium}_epi.obj').GetPoints().GetData())
    dd, ii = tree.query(epi_pts)

    epi_ids = np.array(ii)
    outdir = meshname + f"_{atrium}_vol_surf"

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    shutil.copyfile(meshname + f"_{atrium}_vol.vtk", outdir + f'/{atrium}.vtk')
    resampled = "_res" if resampled else ""
    shutil.copyfile(meshname + f"_{atrium}_epi{resampled}_surf/rings_centroids.csv", outdir + '/rings_centroids.csv')

    write_surf_ids(outdir, "EPI", ii)

    dd, ii = tree.query(vtk_to_numpy(
        smart_reader(meshname + f'_{atrium}_endo.obj').GetPoints().GetData()))
    ii = np.setdiff1d(ii, epi_ids)

    write_surf_ids(outdir, "ENDO", ii)

    fol_name = meshname + f'_{atrium}_epi_surf'

    ids_files = glob(fol_name + '/ids_*')

    for i in range(len(ids_files)):
        ids = np.loadtxt(ids_files[i], skiprows=2, dtype=int)
        name = ids_files[i].split('ids_')[-1][:-4]
        dd, ii = tree.query(epi_pts[ids, :])
        write_surf_ids(outdir, name, ii)


if __name__ == '__main__':
    generate_surf_id()
