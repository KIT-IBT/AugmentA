import csv
import sys
import os

from typing import Dict, Tuple, Any

from vtk_opencarp_helper_methods.vtk_methods.exporting import vtk_obj_writer, vtk_polydata_writer
from vtk_opencarp_helper_methods.vtk_methods.filters import apply_vtk_geom_filter
from vtk_opencarp_helper_methods.vtk_methods.thresholding import get_threshold_between

sys.path.append('Atrial_LDRBM/Generate_Boundaries')
from extract_rings import smart_reader

def load_element_tags(csv_filepath: str = 'Atrial_LDRBM/element_tag.csv') -> Dict[str, str]:
    """
    Load element tags from CSV file. Returns a dictionary mapping element names to tag values.
    """
    if not os.path.exists(csv_filepath):
        raise FileNotFoundError(f"CSV file {csv_filepath} not found.")

    tag_dict: Dict[str, str] = {}

    try:
        with open(csv_filepath, newline='') as f:
            reader = csv.DictReader(f)

            for row in reader:
                if 'name' not in row or 'tag' not in row:
                    raise ValueError("CSV file missing required 'name' or 'tag' columns.")

                tag_dict[row['name']] = row['tag']

    except Exception as e:
        raise RuntimeError(f"Error reading CSV file {csv_filepath}: {e}")
    return tag_dict

def get_wall_tags(tag_dict: Dict[str, str], atrium: str) -> Tuple[int, int]:
    """
    Get epicardial and endocardial tags for the specified atrium. Returns a tuple: (epicardial_tag, endocardial_tag)
    """
    if atrium == "LA":
        try:
            epi_tag = int(tag_dict['left_atrial_wall_epi'])
            endo_tag = int(tag_dict['left_atrial_wall_endo'])
        except KeyError as e:
            raise KeyError(f"Missing expected tag for LA: {e}")

    elif atrium == "RA":
        try:
            epi_tag = int(tag_dict['right_atrial_wall_epi'])
            endo_tag = int(tag_dict['right_atrial_wall_endo'])
        except KeyError as e:
            raise KeyError(f"Missing expected tag for RA: {e}")

    else:
        raise ValueError("Atrium must be 'LA' or 'RA'.")
    return epi_tag, endo_tag

def threshold_model(model: Any,
                    lower_threshold: int,
                    upper_threshold: int,
                    field_association: str = "vtkDataObject::FIELD_ASSOCIATION_CELLS",
                    field_name: str = "tag") -> Any:
    """
    Apply thresholding on the model between the specified thresholds. Returns the threshold object.
    """
    thresh = get_threshold_between(model, lower_threshold, upper_threshold, field_association, field_name)
    if not thresh:
        raise RuntimeError("Thresholding failed: no output obtained.")
    return thresh

def write_filtered_meshes(meshname: str, atrium: str, suffix: str, filtered_mesh: Any) -> None:
    """
    Write the filtered mesh to OBJ and VTK files. The filename is constructed as: meshname_{atrium}{suffix}.(obj|vtk)
    """
    base_filename = f"{meshname}_{atrium}{suffix}"
    vtk_obj_writer(base_filename + ".obj", filtered_mesh)
    vtk_polydata_writer(base_filename + ".vtk", filtered_mesh)

def separate_epi_endo(path: str, atrium: str) -> None:
    """
    Separate epicardial and endocardial surfaces from the input mesh.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mesh file {path} not found.")

    extension = path.split('.')[-1]
    meshname = path[:-(len(extension) + 1)]

    # Load element tags from CSV
    tag_dict = load_element_tags()
    epicardial_tag, endocardial_tag = get_wall_tags(tag_dict, atrium)

    # Read the mesh model
    model = smart_reader(path)

    # Combined threshold: from endocardium to epicardium
    if atrium in ["LA", "RA"]:
        combined_thresh = threshold_model(model, endocardial_tag, epicardial_tag)
    else:
        raise ValueError("Atrium must be 'LA' or 'RA'.")
    filtered_combined = apply_vtk_geom_filter(combined_thresh.GetOutput())
    write_filtered_meshes(meshname, atrium, "", filtered_combined)

    # Epicardial surface: threshold where both lower and upper are epicardial tag
    if atrium in ["LA", "RA"]:
        epi_thresh = threshold_model(model, epicardial_tag, epicardial_tag)
    filtered_epi = apply_vtk_geom_filter(epi_thresh.GetOutput())
    write_filtered_meshes(meshname, atrium, "_epi", filtered_epi)

    # Endocardial surface: threshold where both lower and upper are endocardial tag
    if atrium in ["LA", "RA"]:
        endo_thresh = threshold_model(model, endocardial_tag, endocardial_tag)
    filtered_endo = apply_vtk_geom_filter(endo_thresh.GetOutput())
    write_filtered_meshes(meshname, atrium, "_endo", filtered_endo)


