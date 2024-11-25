import os
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import math
import csv
import sys

from vtk_opencarp_helper_methods.vtk_methods.thresholding import get_threshold_between

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

    if atrium == "LA":
        thresh = get_threshold_between(model, left_atrial_wall_endo, left_atrial_wall_epi,
                                       "vtkDataObject::FIELD_ASSOCIATION_CELLS", "tag")
    elif atrium == "RA":
        thresh = get_threshold_between(model, right_atrial_wall_endo, right_atrial_wall_epi,
                                       "vtkDataObject::FIELD_ASSOCIATION_CELLS", "tag")
    else:
        raise ValueError("Atrium has to be LA or RA")

    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(thresh.GetOutput())
    geo_filter.Update()

    writer = vtk.vtkOBJWriter()
    writer.SetFileName(meshname + f"_{atrium}.obj")
    writer.SetInputData(geo_filter.GetOutput())
    writer.Write()

    if atrium == "LA":
        thresh = get_threshold_between(model, left_atrial_wall_epi, left_atrial_wall_epi,
                                       "vtkDataObject::FIELD_ASSOCIATION_CELLS", "tag")
    elif atrium == "RA":
        thresh = get_threshold_between(right_atrial_wall_epi, right_atrial_wall_epi,
                                       "vtkDataObject::FIELD_ASSOCIATION_CELLS", "tag")
    else:
        raise ValueError("Atrium has to be LA or RA")

    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(thresh.GetOutput())
    geo_filter.Update()
    la_epi = geo_filter.GetOutput()

    writer = vtk.vtkOBJWriter()
    writer.SetFileName(meshname + f"_{atrium}_epi.obj")
    writer.SetInputData(la_epi)
    writer.Write()

    if atrium == "LA":
        thresh = get_threshold_between(model, left_atrial_wall_endo, left_atrial_wall_endo,
                                       "vtkDataObject::FIELD_ASSOCIATION_CELLS", "tag")
    elif atrium == "RA":
        thresh = get_threshold_between(model, right_atrial_wall_endo, right_atrial_wall_endo,
                                       "vtkDataObject::FIELD_ASSOCIATION_CELLS", "tag")
    else:
        raise ValueError("Atrium has to be LA or RA")

    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(thresh.GetOutput())
    geo_filter.Update()
    la_endo = geo_filter.GetOutput()

    writer = vtk.vtkOBJWriter()
    writer.SetFileName(meshname + f"_{atrium}_endo.obj")
    writer.SetInputData(la_endo)
    writer.Write()
