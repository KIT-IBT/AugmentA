import csv
import sys

from vtk_opencarp_helper_methods.vtk_methods.exporting import vtk_obj_writer, vtk_polydata_writer
from vtk_opencarp_helper_methods.vtk_methods.filters import apply_vtk_geom_filter
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

    vtk_obj_writer(meshname + f"_{atrium}.obj", apply_vtk_geom_filter(thresh.GetOutput()))
    vtk_polydata_writer(meshname + f"_{atrium}.vtk", apply_vtk_geom_filter(thresh.GetOutput()))
    if atrium == "LA":
        thresh = get_threshold_between(model, left_atrial_wall_epi, left_atrial_wall_epi,
                                       "vtkDataObject::FIELD_ASSOCIATION_CELLS", "tag")
    elif atrium == "RA":
        thresh = get_threshold_between(right_atrial_wall_epi, right_atrial_wall_epi,
                                       "vtkDataObject::FIELD_ASSOCIATION_CELLS", "tag")
    else:
        raise ValueError("Atrium has to be LA or RA")

    la_epi = apply_vtk_geom_filter(thresh.GetOutput())

    vtk_obj_writer(meshname + f"_{atrium}_epi.obj", la_epi)
    vtk_polydata_writer(meshname + f"_{atrium}_epi.vtk", la_epi)
    if atrium == "LA":
        thresh = get_threshold_between(model, left_atrial_wall_endo, left_atrial_wall_endo,
                                       "vtkDataObject::FIELD_ASSOCIATION_CELLS", "tag")
    elif atrium == "RA":
        thresh = get_threshold_between(model, right_atrial_wall_endo, right_atrial_wall_endo,
                                       "vtkDataObject::FIELD_ASSOCIATION_CELLS", "tag")
    else:
        raise ValueError("Atrium has to be LA or RA")

    la_endo = apply_vtk_geom_filter(thresh.GetOutput())

    vtk_obj_writer(meshname + f"_{atrium}_endo.obj", la_endo)
    vtk_polydata_writer(meshname + f"_{atrium}_endo.vtk", la_endo)
