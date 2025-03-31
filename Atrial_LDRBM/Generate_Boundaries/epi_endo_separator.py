import os
import vtk
from vtk_opencarp_helper_methods.vtk_methods.reader import smart_reader
from vtk_opencarp_helper_methods.vtk_methods.thresholding import get_threshold_between
from vtk_opencarp_helper_methods.vtk_methods.filters import apply_vtk_geom_filter
from file_manager import write_vtk, write_obj

def _prepare_output_directory(mesh_path: str, suffix: str = "_vol_surf") -> str:
    base, _ = os.path.splitext(mesh_path)
    outdir = f"{base}{suffix}"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir

def separate_epi_endo(mesh_path: str, atrium: str, element_tags: dict) -> None:
    """
    Separates the epicardial and endocardial surfaces.
    Writes both VTK and OBJ files.
    """
    if atrium not in ["LA", "RA"]:
        raise ValueError("Atrium must be 'LA' or 'RA'.")

    model = smart_reader(mesh_path)
    if atrium == "LA":
        epi_tag = int(element_tags.get('left_atrial_wall_epi', 0))
        endo_tag = int(element_tags.get('left_atrial_wall_endo', 0))
    else:
        epi_tag = int(element_tags.get('right_atrial_wall_epi', 0))
        endo_tag = int(element_tags.get('right_atrial_wall_endo', 0))

    combined_thresh = get_threshold_between(model, endo_tag, epi_tag,
                                            "vtkDataObject::FIELD_ASSOCIATION_CELLS", "tag")
    filtered_combined = apply_vtk_geom_filter(combined_thresh.GetOutput())

    outdir = _prepare_output_directory(mesh_path, "_vol_surf")
    write_vtk(os.path.join(outdir, f"{atrium}.vtk"), filtered_combined)
    write_obj(os.path.join(outdir, f"{atrium}.obj"), filtered_combined)

    epi_thresh = get_threshold_between(model, epi_tag, epi_tag,
                                       "vtkDataObject::FIELD_ASSOCIATION_CELLS", "tag")
    filtered_epi = apply_vtk_geom_filter(epi_thresh.GetOutput())
    write_vtk(os.path.join(outdir, f"{atrium}_epi.vtk"), filtered_epi)
    write_obj(os.path.join(outdir, f"{atrium}_epi.obj"), filtered_epi)

    endo_thresh = get_threshold_between(model, endo_tag, endo_tag,
                                        "vtkDataObject::FIELD_ASSOCIATION_CELLS", "tag")
    filtered_endo = apply_vtk_geom_filter(endo_thresh.GetOutput())
    write_vtk(os.path.join(outdir, f"{atrium}_endo.vtk"), filtered_endo)
    write_obj(os.path.join(outdir, f"{atrium}_endo.obj"), filtered_endo)
