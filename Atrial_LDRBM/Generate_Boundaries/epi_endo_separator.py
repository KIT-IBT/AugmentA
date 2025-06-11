import os
import vtk
from typing import Dict, Any  # Added Any for type hinting VTK objects

# Assuming mesh.py contains MeshReader (though not used directly in the fixed logic below)
# from Atrial_LDRBM.Generate_Boundaries.mesh import MeshReader
from vtk_opencarp_helper_methods.vtk_methods.reader import smart_reader
from vtk_opencarp_helper_methods.vtk_methods.thresholding import get_threshold_between
from vtk_opencarp_helper_methods.vtk_methods.filters import apply_vtk_geom_filter
from Atrial_LDRBM.Generate_Boundaries.mesh import Mesh


def _prepare_output_directory(mesh_path: str, suffix: str = "_vol_surf") -> str:
    """
    Prepares an output directory based on mesh path.
    Example: /path/to/mesh_RA_vol.vtk -> /path/to/mesh_vol_surf/
    """
    base, _ = os.path.splitext(mesh_path)

    # Handle potential _RA / _LA suffixes if present in mesh_path
    base_parts = base.split('_')
    if len(base_parts) > 1 and base_parts[-1] in ['RA', 'LA', 'epi', 'endo']:
        base = '_'.join(base_parts[:-1])

    outdir = f"{base}{suffix}"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def separate_epi_endo(mesh_path: str, atrium: str, element_tags: Dict[str, str]) -> None:
    """
    Separates the epicardial and endocardial surfaces from a tagged volumetric mesh.
    Writes both VTK and OBJ files for combined, epi, and endo surfaces, using the
    original base naming convention expected by downstream modules (e.g., mesh_RA_epi.obj).
    """
    if atrium not in ["LA", "RA"]:
        raise ValueError("Atrium must be 'LA' or 'RA'.")

    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Input mesh file not found: {mesh_path}")

    print(f"--- Separating {atrium} Epi/Endo from: {mesh_path} ---")

    try:
        # Read the volumetric mesh using smart_reader to preserve cell data
        model_ug: vtk.vtkUnstructuredGrid | vtk.vtkPolyData | None = smart_reader(
            mesh_path)  # Type hint allows various VTK dataset types
        if model_ug is None or not hasattr(model_ug, 'GetCellData') or model_ug.GetNumberOfPoints() == 0:
            raise ValueError(f"Failed to read mesh or mesh is empty/invalid: {mesh_path}")
        if model_ug.GetCellData().GetArray("tag") is None:
            raise ValueError(f"Mesh {mesh_path} is missing required 'tag' cell data array.")
        print(
            f"  Successfully read tagged mesh with {model_ug.GetNumberOfPoints()} points and {model_ug.GetNumberOfCells()} cells.")

    except Exception as e:
        print(f"Error loading mesh {mesh_path}: {e}")
        raise

    base_vol_path, _ = os.path.splitext(mesh_path)
    if base_vol_path.endswith('_vol'):
        original_base_name = base_vol_path[:-4]
        print(f"Derived original base name for output: {original_base_name}")
    else:
        print(
            f"Warning: Input mesh path '{mesh_path}' did not end with '_vol'. Using '{base_vol_path}' as base for output.")
        original_base_name = base_vol_path

    if atrium == "LA":
        epi_tag_key = 'left_atrial_wall_epi'
        endo_tag_key = 'left_atrial_wall_endo'
    else:  # atrium == "RA":
        epi_tag_key = 'right_atrial_wall_epi'
        endo_tag_key = 'right_atrial_wall_endo'

    try:
        if epi_tag_key not in element_tags or endo_tag_key not in element_tags:
            raise KeyError(f"Missing required tags '{epi_tag_key}' or '{endo_tag_key}' in element_tags dictionary.")
        epi_tag = int(element_tags[epi_tag_key])
        endo_tag = int(element_tags[endo_tag_key])

        print(f"Using Tags - Epi: {epi_tag}, Endo: {endo_tag}")

    except KeyError as e:
        raise e
    except ValueError as e:
        raise ValueError(f"Invalid tag value found in element_tags dictionary. Check CSV format. Original error: {e}")

    # --- Process Surfaces ---
    def threshold_and_write(lower_tag: int, upper_tag: int, suffix: str, surface_name: str):
        """Internal helper to threshold, filter, and write surface."""
        try:
            print(f"  Processing {surface_name} surface (Tags: {lower_tag}-{upper_tag})...")
            thresh = get_threshold_between(model_ug, lower_tag, upper_tag,
                                           "vtkDataObject::FIELD_ASSOCIATION_CELLS", "tag")

            threshold_output = thresh.GetOutput()
            if threshold_output is None or threshold_output.GetNumberOfCells() == 0:
                print(f"Warning: Thresholding for {surface_name} (tags {lower_tag}-{upper_tag}) yielded no cells.")
                return  # Skip writing if no cells match the threshold

            # Apply geometry filter to get the surface representation
            filtered_surface = apply_vtk_geom_filter(threshold_output)

            if filtered_surface and filtered_surface.GetNumberOfPoints() > 0:
                # Write outputs using the ORIGINAL base name structure
                output_path_vtk = f"{original_base_name}_{atrium}{suffix}.vtk"  # e.g., mesh_RA_epi.vtk
                output_path_obj = f"{original_base_name}_{atrium}{suffix}.obj"  # e.g., mesh_RA_epi.obj
                print(f"Writing {surface_name}: {output_path_vtk}, {output_path_obj}")
                mesh_to_save = Mesh(filtered_surface)
                mesh_to_save.save(output_path_vtk)
                mesh_to_save.save(output_path_obj)
            else:
                print(f"Warning: {surface_name} surface extraction yielded no geometry after geometry filter.")

        except Exception as e:
            print(f"  Error during {surface_name} surface processing: {e}")

    # Combined Wall Surface (Tags from endo to epi)
    threshold_and_write(endo_tag, epi_tag, "", "Combined Wall")

    # Epicardial Surface (Tag == epi_tag)
    threshold_and_write(epi_tag, epi_tag, "_epi", "Epicardial")

    # Endocardial Surface (Tag == endo_tag)
    threshold_and_write(endo_tag, endo_tag, "_endo", "Endocardial")
