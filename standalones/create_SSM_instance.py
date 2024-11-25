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

import h5py
import argparse
import pyvista as pv
import numpy as np


def parser():
    parser = argparse.ArgumentParser(description='Cut veins manually')
    parser.add_argument('--SSM_file',
                        type=str,
                        default="",
                        help='path to SSM')
    parser.add_argument('--coefficients_file',
                        type=str,
                        default="",
                        help='path to SSM coefficients')
    parser.add_argument('--output_file',
                        type=str,
                        default="",
                        help='path to output')

    return parser


def create_SSM_instance(SSM_file, coefficients_file, output_file):
    r = np.loadtxt(coefficients_file, delimiter=',')
    with h5py.File(SSM_file, "r") as f:
        mean_pts = list(f["model"]["mean"])[0]
        mean_cells = np.vstack(list(f["representer"]["cells"])).T
        mean_cells = np.c_[np.ones(len(mean_cells), dtype=int) * 3, mean_cells]
        pca_basisfunctions = np.vstack(list(f["model"]["pcaBasis"])).T
        pca_var = list(f["model"]["pcaVariance"])[0]

        for i in range(len(pca_basisfunctions)):
            mean_pts = mean_pts + r[i] * pca_basisfunctions[i, :] * np.sqrt(pca_var[i])

        mean_pts = mean_pts.reshape(int(len(mean_pts) / 3), 3)

        surf = pv.PolyData(mean_pts, mean_cells)
        pv.save_meshio(output_file, surf, "obj")


def run():
    args = parser().parse_args()
    create_SSM_instance(args.SSM_file, args.coefficients_file, args.output_file)


if __name__ == '__main__':
    run()
