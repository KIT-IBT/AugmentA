#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Take LA.vtk as input

Detecte 5 rings on the model
calculate the center points of each ring and save them into a txt file
'''
import os

os.environ['CARPUTILS_SETTINGS'] = '/Volumes/bordeaux/IBT/openCARP/.config/carputils/settings.yaml'
# os.environ['PATH'] = '/Volumes/bordeaux/IBT/bin/macosx:/Volumes/bordeaux/IBT/src/CardioMechanics/trunk/src/Scripts:/Volumes/bordeaux/IBT/bin/macosx:/Volumes/bordeaux/IBT/pl:/Volumes/bordeaux/IBT/python:/Volumes/bordeaux/IBT/thirdparty/macosx/bin:/Volumes/bordeaux/IBT/thirdparty/macosx/openMPI-64bit/bin:/opt/X11/bin:/Applications/MATLAB_R2020a.app/bin/:/opt/local/bin:/opt/local/sbin:/usr/bin:/bin:/usr/sbin:/sbin:/Volumes/bordeaux/IBT/openCARP/bin:/Volumes/bordeaux/IBT/openCARP/bin:/usr/local/bin'
# path of meshtool
os.environ['PATH'] = '/Volumes/koala/Users/tz205/IBT_tz205_la816_MA/Sources/aneic-meshtool-3f54e98259b3'
import subprocess
from mayavi import mlab
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
import math

"""
distance(p0,p1,digits=2)

calculate the distance between two points
"""


def distance(p0, p1, digits=4):
    a = map(lambda x: (x[0] - x[1]) ** 2, zip(p0, p1))
    return round(math.sqrt(sum(a)), digits)


"""
GetCellEdgeLength(vtkdata)
    
Input is the Vtk data e.g reader.GetOutput()
return value is the mean length of cells edge
"""


def GetCellEdgeLength(vtkdata):
    cell_id = vtk.vtkIdList()
    cell_id.InsertNextId(5)
    extractor = vtk.vtkExtractCells()
    extractor.SetInputData(vtkdata)
    extractor.AddCellList(cell_id)
    extractor.Update()
    extraction = extractor.GetOutput().GetPoints().GetData()
    coordinates = vtk.util.numpy_support.vtk_to_numpy(extraction)
    d1 = distance((coordinates[0][0], coordinates[0][1], coordinates[0][2]),
                  (coordinates[1][0], coordinates[1][1], coordinates[1][2]))

    d2 = distance((coordinates[0][0], coordinates[0][1], coordinates[0][2]),
                  (coordinates[2][0], coordinates[2][1], coordinates[2][2]))

    d3 = distance((coordinates[2][0], coordinates[2][1], coordinates[2][2]),
                  (coordinates[1][0], coordinates[1][1], coordinates[1][2]))
    mean_d = np.mean([d1, d2, d3])

    return mean_d


"""
GetCellEdgeStatistic(Input)

Using the meshtool to get the mean cell edge length and average number of connections of point 
"""


def GetCellEdgeStatistic():
    subprocess.run(["meshtool",
                    "query",
                    "edges",
                    "-ifmt=vtk",
                    "-msh=model/LA.vtk",
                    "-odat=data/edge"])
    length = np.loadtxt('data/edge.len.dat')
    connection = np.loadtxt('data/edge.concnt.dat')

    mean = np.mean(length)
    connection = round(np.mean(connection))
    return mean, connection


"""
write_points_into_tex(ring1_center, ring2_center, ring3_center, ring4_center, ring5_center)

write the coordinates of 5 points in points_data.txt
"""


def write_points_into_tex(ring1_center, ring2_center, ring3_center, ring4_center, ring5_center):
    mylist = [[str(ring1_center[0]) + ' ', str(ring1_center[1]) + ' ', str(ring1_center[2]) + ' '],
              [str(ring2_center[0]) + ' ', str(ring2_center[1]) + ' ', str(ring2_center[2]) + ' '],
              [str(ring3_center[0]) + ' ', str(ring3_center[1]) + ' ', str(ring3_center[2]) + ' '],
              [str(ring4_center[0]) + ' ', str(ring4_center[1]) + ' ', str(ring4_center[2]) + ' '],
              [str(ring5_center[0]) + ' ', str(ring5_center[1]) + ' ', str(ring5_center[2]) + ' ']]
    file_write_obj = open("data/points_data.txt", 'w')
    for var in mylist:
        file_write_obj.writelines(var)
        file_write_obj.write('\n')
    file_write_obj.close()


def run():
    global mesh, cursor3d
    fig = mlab.figure('Atrium, detected rings and centers of rings')
    mlab.clf()

    # Calculate the curvature
    subprocess.run(["meshtool",
                    "query",
                    "curvature",
                    "-ifmt=vtk",
                    "-msh=model/LA.vtk",
                    "-size=3"])
    # Check the vtk is polydata or unstructured grid
    data_checker = vtk.vtkDataSetReader()
    data_checker.SetFileName('model/LA.vtk')
    data_checker.Update()

    if data_checker.IsFilePolyData():
        reader = vtk.vtkPolyDataReader()
    elif data_checker.IsFileUnstructuredGrid():
        reader = vtk.vtkUnstructuredGridReader()

    reader.SetFileName('model/LA.vtk')
    reader.Update()
    points_data = reader.GetOutput().GetPoints().GetData()
    points_coordinate = vtk.util.numpy_support.vtk_to_numpy(points_data)

    print('################------Coordinate of points--------################')
    print('%d points are found' % len(points_coordinate))
    print(points_coordinate)
    curvature = np.loadtxt('model/LA.curv.dat')  # load the curvature data of each points
    k = 0.7  # let user set this value, default is 0.7
    threshold = k * max(curvature)  # size=2.7 thr.=0.52
    print("Maximal curvature:{}".format(max(curvature)))
    ring_points_index = np.where(curvature > threshold)  # pick up the points whose curvature bigger than threshold
    ring_points = points_coordinate[ring_points_index]  # get the points whose curvature bigger than threshold
    print('################------Selected points on ring--------################')
    print('%d points on rings are found' % len(ring_points_index[0]))
    print(ring_points)

    # Using DBSCAN to sort the rings
    # edge_length = GetCellEdgeLength(reader.GetOutput())
    edge_length, connection_number = GetCellEdgeStatistic()
    print("[Cell Edge Statistic] Average cell edge length: %f" % edge_length)
    print("[Cell Edge Statistic] Number of Connections: %d" % connection_number)
    y_pred = DBSCAN(eps=edge_length * 1.5, min_samples=connection_number).fit_predict(ring_points)  # 1.6
    n_clusters_ = len(set(y_pred)) - (1 if -1 in y_pred else 0)
    n_noise_ = list(y_pred).count(-1)
    print('Number of clusters: %d' % n_clusters_)
    print('Number of noise points: %d' % n_noise_)
    # print(list(y_pred))

    # Pick up top 5 biggest regions
    # Filter out the -1, namly  noise group
    top_six = Counter(y_pred).most_common(6)
    top_five = []
    for i in range(0, 6):
        if top_six[i][0] != -1:
            top_five += [top_six[i][0]]

    temp = locals()
    for j in range(0, 5):
        temp['ring' + str(j + 1)] = [i for i, x in enumerate(y_pred) if x == top_five[j]]

    # Ring points list
    ring1 = ring_points[np.array(temp['ring1'])]
    ring2 = ring_points[np.array(temp['ring2'])]
    ring3 = ring_points[np.array(temp['ring3'])]
    ring4 = ring_points[np.array(temp['ring4'])]
    ring5 = ring_points[np.array(temp['ring5'])]

    # Calculate the centers of each ring
    ring1_center = [np.mean(ring1[:, 0]), np.mean(ring1[:, 1]), np.mean(ring1[:, 2])]
    ring2_center = [np.mean(ring2[:, 0]), np.mean(ring2[:, 1]), np.mean(ring2[:, 2])]
    ring3_center = [np.mean(ring3[:, 0]), np.mean(ring3[:, 1]), np.mean(ring3[:, 2])]
    ring4_center = [np.mean(ring4[:, 0]), np.mean(ring4[:, 1]), np.mean(ring4[:, 2])]
    ring5_center = [np.mean(ring5[:, 0]), np.mean(ring5[:, 1]), np.mean(ring5[:, 2])]

    # write the centers of each ring into points_data.txt
    write_points_into_tex(ring1_center, ring2_center, ring3_center, ring4_center, ring5_center)

    # Save coordinates of points that detected on the biggest ring
    ring1_list = ring1.tolist()
    file_write_obj = open("data/ring_points_data.txt", 'w')
    for var in ring1_list:
        file_write_obj.writelines(str(var).replace(',', ' ').replace('[', '').replace(']', ''))
        file_write_obj.write('\n')
    file_write_obj.close()

    # Show the segmentation of rings
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(ring_points[:, 0], ring_points[:, 1], ring_points[:, 2], c=y_pred)
    plt.show()

    # # Plot 3d  model, ring points and center points of each ring
    # source = mlab.pipeline.open('model/LA.vtk')  # Open the source
    # surf = mlab.pipeline.surface(source, color=(0.9, 0.0, 0.0))

    # # Ring points
    # points =mlab.points3d(ring1[:,0], ring1[:,1], ring1[:,2], scale_factor = 1,color=(0.0, 0.9, 0.9))
    # points =mlab.points3d(ring2[:,0], ring2[:,1], ring2[:,2], scale_factor = 1,color=(0.1, 0.4, 0.9))
    # points =mlab.points3d(ring3[:,0], ring3[:,1], ring3[:,2], scale_factor = 1,color=(0.2, 0.2, 0.9))
    # points =mlab.points3d(ring4[:,0], ring4[:,1], ring4[:,2], scale_factor = 1,color=(0.0, 0.5, 0.5))
    # points =mlab.points3d(ring5[:,0], ring5[:,1], ring5[:,2], scale_factor = 1,color=(0.8, 0.2, 0.4))

    # # Center points
    # points =mlab.points3d(ring1_center[0], ring1_center[1], ring1_center[2], scale_factor = 2,color=(0.0, 0.9, 0.0))
    # points =mlab.points3d(ring2_center[0], ring2_center[1], ring2_center[2], scale_factor = 2,color=(0.0, 0.9, 0.0))
    # points =mlab.points3d(ring3_center[0], ring3_center[1], ring3_center[2], scale_factor = 2,color=(0.0, 0.9, 0.0))
    # points =mlab.points3d(ring4_center[0], ring4_center[1], ring4_center[2], scale_factor = 2,color=(0.0, 0.9, 0.0))
    # points =mlab.points3d(ring5_center[0], ring5_center[1], ring5_center[2], scale_factor = 2,color=(0.0, 0.9, 0.0))
    # points =mlab.points3d(ring_points[:,0], ring_points[:,1], ring_points[:,2], scale_factor = 1,color=(0.0, 0.9, 0.9))

    # print('The detetion result shows in mayavi window, close the window to continue...')
    # mlab.show()


if __name__ == '__main__':
    run()
