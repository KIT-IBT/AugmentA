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
import subprocess

def generate_mesh(path, la_mesh_scale=1):
    # # generate mesh in vtk form

    subprocess.run(["meshtool",
                    "generate",
                    "mesh",
                    #"-scale="  + str(la_mesh_scale),
                    "-surf=" + str(path)+'.obj',
                    "-ofmt=vtk",
                    "-outmsh="+ str(path)+'_vol'])

if __name__ == '__main__':
    generate_mesh()
