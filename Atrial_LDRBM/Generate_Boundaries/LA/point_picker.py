#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 11:20:20 2020

@author: tz205

mayavi point picker

TODO:
    return the coordinate and get the point id for the pipline
"""

import numpy as np
from mayavi import mlab

def picker_callback(picker_obj):
    # retrieve VTK actors of picked objects
    picked = picker_obj.actors
    if mesh.actor.actor._vtk_obj in [o._vtk_obj for o in picked]:
        # Get mouse position on-click
        position = picker_obj.pick_position
        # Move the cursor in scene according to new clicked coordinates
        print("Data indices: ",position)
        coordinates = position
        print("Coord indices: ",coordinates)
        cursor3d.mlab_source.reset(x=position[0],
                                   y=position[1],
                                   z=position[2])


# @tools.carpexample(parser, jobID)
def run(Input):
    global coordinates
    coordinates = []
    # Some logic to select 'mesh' and the data index when picking.
    xyz = np.loadtxt("model/"+str(Input)+".pts",skiprows=1)
    triangles = np.loadtxt("model/"+str(Input)+".elem",skiprows=1, usecols=(1,2,3))
    # Plot the data
    global mesh, cursor3d
    fig = mlab.figure('Click to pick points')
    mlab.clf()

    mesh = mlab.triangular_mesh(xyz[:,0],xyz[:,1],xyz[:,2], triangles, transparent=False, representation='fancymesh', scale_factor = 700., color=(0.9, 0.0, 0.0))
    surf = mlab.triangular_mesh(xyz[:,0],xyz[:,1],xyz[:,2], triangles, transparent=False)
    print("Please select a point on the peak of appendage")
    # Plot the cursor which will be updated at each click on the plane 
    cursor3d = mlab.points3d(0., 0., 0., mode='axes',color=(0., 0., 0.), scale_factor=5000.)

    picker = fig.on_mouse_pick(picker_callback)

    # Decrease the tolerance, so that we can more easily select a precise
    # point.
    picker.tolerance = 0.01

    mlab.show()

    print(coordinates)

    
if __name__ == '__main__':
    Input = input('You want LA or LA_high: ')
    run(Input)

