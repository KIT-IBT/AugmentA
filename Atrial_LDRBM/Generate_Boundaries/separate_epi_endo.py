import os
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import math
import csv
import sys

sys.path.append('Atrial_LDRBM/Generate_Boundaries')
from extract_rings import smart_reader


def separate_epi_endo(path, atrium):
    extension = path.split('.')[-1]
    meshname = path[:-(len(extension) + 1)]

    with open('Atrial_LDRBM/element_tag.csv') as f:
        tag_dict = {}
        reader = csv.DictReader(f)
        for row in reader:
            tag_dict[row['name']] = row['tag']
    if atrium == "LA":
        left_atrial_wall_epi = int(tag_dict['left_atrial_wall_epi'])
        left_atrial_wall_endo = int(tag_dict['left_atrial_wall_endo'])
    elif atrium == "RA":
        right_atrial_wall_endo = int(tag_dict['right_atrial_wall_endo'])
        right_atrial_wall_epi = int(tag_dict['right_atrial_wall_epi'])

    model = smart_reader(path)

    thresh = vtk.vtkThreshold()
    thresh.SetInputData(model)
    if atrium == "LA":
        thresh.ThresholdBetween(left_atrial_wall_endo, left_atrial_wall_epi)
    elif atrium == "RA":
        thresh.ThresholdBetween(right_atrial_wall_endo, right_atrial_wall_epi)
    thresh.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_CELLS", "tag")
    thresh.Update()

    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(thresh.GetOutput())
    geo_filter.Update()

    writer = vtk.vtkOBJWriter()
    writer.SetFileName(meshname + "_{}.obj".format(atrium))
    writer.SetInputData(geo_filter.GetOutput())
    writer.Write()

    thresh = vtk.vtkThreshold()
    thresh.SetInputData(model)
    if atrium == "LA":
        thresh.ThresholdBetween(left_atrial_wall_epi, left_atrial_wall_epi)
    elif atrium == "RA":
        thresh.ThresholdBetween(right_atrial_wall_epi, right_atrial_wall_epi)
    thresh.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_CELLS", "tag")
    thresh.Update()

    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(thresh.GetOutput())
    geo_filter.Update()
    la_epi = geo_filter.GetOutput()

    writer = vtk.vtkOBJWriter()
    writer.SetFileName(meshname + "_{}_epi.obj".format(atrium))
    writer.SetInputData(la_epi)
    writer.Write()

    thresh = vtk.vtkThreshold()
    thresh.SetInputData(model)
    if atrium == "LA":
        thresh.ThresholdBetween(left_atrial_wall_endo, left_atrial_wall_endo)
    elif atrium == "RA":
        thresh.ThresholdBetween(right_atrial_wall_endo, right_atrial_wall_endo)
    thresh.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_CELLS", "tag")
    thresh.Update()

    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(thresh.GetOutput())
    geo_filter.Update()
    la_endo = geo_filter.GetOutput()

    writer = vtk.vtkOBJWriter()
    writer.SetFileName(meshname + "_{}_endo.obj".format(atrium))
    writer.SetInputData(la_endo)
    writer.Write()
