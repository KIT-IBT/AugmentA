import os
import shutil
import numpy as np
from glob import glob
import vtk
from scipy.spatial import cKDTree

from vtk_opencarp_helper_methods.vtk_methods.exporting import write_to_vtx
from vtk_opencarp_helper_methods.vtk_methods.converters import vtk_to_numpy
from vtk_opencarp_helper_methods.vtk_methods.reader import vtx_reader
from vtk_opencarp_helper_methods.vtk_methods.filters import generate_ids

from Atrial_LDRBM.Generate_Boundaries.mesh import Mesh


def _prepare_output_directory(vol_mesh_path: str, atrium: str) -> str:
    """
    Determines and creates the standard output directory for surface ID generation.
    For example, given a volumetric mesh path like '/path/to/mesh_RA_vol.vtk', this function
    will create and return '/path/to/mesh_RA_vol_surf/'.
    """
    # Split the file path to remove the extension (e.g., '.vtk')
    base_vol_path, _ = os.path.splitext(vol_mesh_path)

    # Remove the '_vol' suffix if it exists, so that we get the common base name for the surfaces
    if base_vol_path.endswith('_vol'):
        base_name = base_vol_path[:-4]
    else:
        base_name = base_vol_path

    # Append '_vol_surf' to denote that this directory will hold the surface files
    outdir = f"{base_name}_vol_surf"

    os.makedirs(outdir, exist_ok=True)
    return outdir


def _map_ring_vtx_files(surf_proc_dir: str, outdir: str,
                        epicardial_polydata: vtk.vtkPolyData,
                        volumetric_kdtree: cKDTree, debug: bool = False):
    if not epicardial_polydata.GetPointData() or not epicardial_polydata.GetPointData().GetArray("Ids"):
        epicardial_polydata = generate_ids(epicardial_polydata, "Ids", "CellIds")

    # Create a mapping from each epicardial point's id to its 3D coordinates
    epi_id_to_coord = {}
    try:
        epi_ids_array = vtk_to_numpy(epicardial_polydata.GetPointData().GetArray("Ids"))
        epi_points_data = epicardial_polydata.GetPoints()

        for i in range(epicardial_polydata.GetNumberOfPoints()):
            point_id = int(epi_ids_array[i])
            point_coord = epi_points_data.GetPoint(i)
            epi_id_to_coord[point_id] = point_coord

    except Exception as e:
        if debug:
            print(f"Error creating epicardial ID-to-coordinate map: {e}. Skipping ring file mapping.")
        return

    # Find all VTX files in the source directory that match the pattern.
    vtx_pattern = os.path.join(surf_proc_dir, 'ids_*.vtx')
    source_vtx_files = glob(vtx_pattern)
    num_mapped = 0

    excluded_files = {'ids_EPI.vtx', 'ids_ENDO.vtx'}

    for file_path in source_vtx_files:
        file_name = os.path.basename(file_path)
        if file_name in excluded_files:
            continue

        # Read the original point IDs from the VTX file.
        source_point_ids = vtx_reader(file_path)

        if not source_point_ids:
            if debug:
                print(f"Warning: No valid IDs found in {file_name}. Skipping this file.")
            continue

        # For each source id, retrieve its coordinate from the epicardial mapping.
        coords_to_map = []
        missing_ids_count = 0
        for pid in source_point_ids:
            coord = epi_id_to_coord.get(pid)
            if coord:
                coords_to_map.append(coord)
            else:
                missing_ids_count += 1

        if missing_ids_count > 0 and debug:
            print(f"Warning: {missing_ids_count} IDs in {file_name} could not be mapped.")

        if not coords_to_map:
            if debug:
                print(f"Warning: No coordinates available for {file_name}. Skipping file.")
            continue

        # Use the KDTree built on the volumetric mesh to find the nearest volumetric point for each epicardial coordinate.
        _, mapped_indices = volumetric_kdtree.query(np.array(coords_to_map))

        # Write the mapped volumetric indices to a new VTX file in the output directory.
        output_vtx_path = os.path.join(outdir, file_name)
        write_to_vtx(output_vtx_path, mapped_indices)
        num_mapped += 1


def generate_surf_id(vol_mesh_path: str, atrium: str, resampled: bool = False, debug: bool = False) -> None:
    """
    Generates surface ID files by mapping points from surface meshes (epicardium, endocardium,
    and rings) onto the volumetric mesh. It creates an output directory, processes the surfaces,
    maps ring VTX files, and copies base files as needed.

    Args:
        vol_mesh_path (str): Path to the volumetric mesh file (e.g., 'mesh_RA_vol.vtk').
        atrium (str): The atrium identifier ('LA' or 'RA').
        resampled (bool): Flag indicating if resampled surfaces should be used.
        debug (bool): If True, prints detailed debug information.
    """
    if not isinstance(vol_mesh_path, str) or not vol_mesh_path:
        raise ValueError("vol_mesh_path must be a non-empty string.")
    if not isinstance(atrium, str) or not atrium:
        raise ValueError("atrium must be a non-empty string.")

    if debug:
        print(f"Generating surface IDs for {atrium} using volumetric mesh: {vol_mesh_path}")

    # Derive a base name by removing the extension and '_vol'
    base_vol_path, _ = os.path.splitext(vol_mesh_path)
    if base_vol_path.endswith('_vol'):
        base_name = base_vol_path[:-4]
    else:
        base_name = base_vol_path

    try:
        volumetric_mesh = Mesh.from_file(vol_mesh_path).get_polydata()

        # Extract coordinates for each point in the volumetric mesh.
        volumetric_coords = vtk_to_numpy(volumetric_mesh.GetPoints().GetData())

        # Build a KDTree from the volumetric mesh points for efficient nearest neighbor queries.
        volumetric_kdtree = cKDTree(volumetric_coords)

        if debug:
            print(f"Loaded volumetric mesh with {volumetric_mesh.GetNumberOfPoints()} points; KDTree built.")

    except FileNotFoundError:
        raise FileNotFoundError(f"Volumetric mesh file not found at {vol_mesh_path}.")
    except Exception as e:
        raise RuntimeError(f"Failed to load volumetric mesh or build KDTree: {e}")

    outdir = _prepare_output_directory(vol_mesh_path, atrium)

    epi_surf_path = f"{base_name}_epi.obj"
    epi_indices = np.array([])
    epicardial_polydata = None
    try:
        epicardial_polydata = Mesh.from_file(epi_surf_path).get_polydata()

        # Get coordinates of epicardial points for KDTree mapping.
        epi_pts_coords = vtk_to_numpy(epicardial_polydata.GetPoints().GetData())

        # Query the volumetric KDTree to find the nearest volumetric point for each epicardial point.
        _, epi_indices = volumetric_kdtree.query(epi_pts_coords)

        # Write the mapped volumetric indices to a VTX file.
        write_to_vtx(os.path.join(outdir, "ids_EPI.vtx"), epi_indices)

    except FileNotFoundError:
        print(f"Warning: Epicardial surface file not found at {epi_surf_path}. Skipping epicardium processing.")
    except Exception as e:
        print(f"Warning: Error processing epicardial surface {epi_surf_path}: {e}")

    # The endocardial surface file is expected to be named like '<base_name>_endo.obj'
    endo_surf_path = f"{base_name}_endo.obj"
    try:
        endocardial_polydata = Mesh.from_file(endo_surf_path).get_polydata()

        # Get coordinates from the endocardial surface
        endo_pts_coords = vtk_to_numpy(endocardial_polydata.GetPoints().GetData())

        # volumetric KDTree for each point
        _, endo_indices_raw = volumetric_kdtree.query(endo_pts_coords)

        # Remove any points already mapped to the epicardium to avoid overlaps
        if epi_indices.size > 0:
            endo_indices_filtered = np.setdiff1d(endo_indices_raw, epi_indices, assume_unique=True)
        else:
            endo_indices_filtered = endo_indices_raw

        write_to_vtx(os.path.join(outdir, "ids_ENDO.vtx"), endo_indices_filtered)

    except FileNotFoundError:
        print(f"Warning: Endocardial surface file not found at {endo_surf_path}. Skipping endocardium processing.")
    except Exception as e:
        print(f"Warning: Error processing endocardial surface {endo_surf_path}: {e}")

    try:
        shutil.copyfile(vol_mesh_path, os.path.join(outdir, f"{atrium}.vtk"))

        res_suffix = "_res" if resampled else ""
        surf_proc_dir = f"{base_name}_epi{res_suffix}_surf"
        centroids_src_path = os.path.join(surf_proc_dir, "rings_centroids.csv")
        centroids_dst_path = os.path.join(outdir, "rings_centroids.csv")

        shutil.copyfile(centroids_src_path, centroids_dst_path)
    except Exception as e:
        print(f"Warning: Error copying base files (volume mesh / centroids): {e}")

    # map the ring VTX files
    if epicardial_polydata is not None:
        _map_ring_vtx_files(surf_proc_dir, outdir, epicardial_polydata, volumetric_kdtree, debug=debug)
    elif debug:
        print("Skipping ring VTX file mapping: missing epicardial_polydata or volumetric_kdtree.")
