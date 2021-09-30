#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:23:31 2020

@author: tz205

Generate mesh:
    meshname.elem
    meshname.pts
    meshname.lon
    meshname.surf
"""
import os
import subprocess

# path of mesh
os.environ['PATH'] = '/Volumes/bordeaux/IBT/bin/macosx:/Volumes/bordeaux/IBT/src/CardioMechanics/trunk/src/Scripts:/Volumes/bordeaux/IBT/bin/macosx:/Volumes/bordeaux/IBT/pl:/Volumes/bordeaux/IBT/python:/Volumes/bordeaux/IBT/thirdparty/macosx/bin:/Volumes/bordeaux/IBT/thirdparty/macosx/openMPI-64bit/bin:/opt/X11/bin:/Applications/MATLAB_R2020a.app/bin/:/opt/local/bin:/opt/local/sbin:/usr/bin:/bin:/usr/sbin:/sbin:/Volumes/bordeaux/IBT/openCARP/bin:/Volumes/bordeaux/IBT/openCARP/bin:/usr/local/bin'

def generate_mesh():
    # generate mesh in carp_txt format
    subprocess.run(["meshtool",
                    "generate",
                    "mesh",
                    "-scale=0.6",
                    "-ofmt=vtk",
                    "-surf=model/LA.vtk",
                    "-outmsh=../../LDRBM/Fiber_LA/Mesh/LA_mesh"])
    # convert mesh in vtk
    subprocess.run(["meshtool",
                    "convert",
                    "-imsh=../../LDRBM/Fiber_LA/Mesh/LA_mesh",
                    "-omsh=../../LDRBM/Fiber_LA/Mesh/LA_mesh.vtk"])

    #### modify the ion file #####    why????
    # get the num of element
    # Commented by Luca
    # file_read = open("../../LDRBM/Fiber_LA/Mesh/LA_mesh.elem", 'r')
    # lines = file_read.readlines()
    # num = int(lines[0])
    # file_read.close()
    # # write the ion file
    # file_write_obj = open("../../LDRBM/Fiber_LA/Mesh/LA_mesh.lon", 'w')
    # file_write_obj.writelines('1')
    # file_write_obj.write('\n')
    # for i in range(num):
    #     file_write_obj.writelines('1 0 0')
    #     file_write_obj.write('\n')
    # file_write_obj.close()


if __name__ == '__main__':
    run()
