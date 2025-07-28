#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from typing import Any, Tuple, List, Union

import pymeshfix
import pyvista as pv
import vtk

from vtk_openCARP_methods_ibt.AugmentA_methods.point_selection import pick_point
from vtk_openCARP_methods_ibt.vtk_methods.exporting import vtk_polydata_writer
from vtk_openCARP_methods_ibt.vtk_methods.finder import find_closest_point
from vtk_openCARP_methods_ibt.vtk_methods.helper_methods import cut_mesh_with_radius
from vtk_openCARP_methods_ibt.vtk_methods.mapper import point_array_mapper
from vtk_openCARP_methods_ibt.vtk_methods.reader import smart_reader
from vtk_openCARP_methods_ibt.AugmentA_methods.vtk_operations import extract_largest_region

# Use dark theme for PyVista plots
pv.set_plot_theme("dark")


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Cut veins manually')
    parser.add_argument('--mesh', type=str, default="", help='path to mesh')
    parser.add_argument('--atrium', type=str, default="", help='write LA or RA')
    parser.add_argument('--min_cutting_radius', type=float, default=7.5, help='radius to cut veins/valves in mm')
    parser.add_argument('--max_cutting_radius', type=float, default=17.5, help='radius to cut veins/valves in mm')
    parser.add_argument('--scale', type=float, default=1.0, help='scaling factor (if mesh units != mm)')
    parser.add_argument('--MRI', type=int, default=0, help='set to 1 if the input is an MRI segmentation')
    parser.add_argument('--debug', type=int, default=0, help='debug flag')
    return parser


def _clean_mesh(meshpath: str, atrium: str) -> Tuple[str, str]:
    """
    Clean the mesh from holes and self-intersecting triangles.

    :param meshpath: Path to the input mesh file (must exist)
    :param atrium: 'LA' or 'RA', used to name output files
    :return: Tuple of:
             - Directory containing the original mesh
             - Path to the cleaned VTK file
    """
    if not isinstance(meshpath, str) or not meshpath:
        raise ValueError("meshpath must be a non-empty string.")
    if not isinstance(atrium, str) or not atrium:
        raise ValueError("atrium must be a non-empty string.")
    if not os.path.isfile(meshpath):
        raise FileNotFoundError(f"Mesh file not found: {meshpath}")

    # Directory of the input mesh and base for cleaned filenames
    mesh_dir = os.path.dirname(meshpath)
    clean_base = os.path.join(mesh_dir, f"{atrium}_clean")
    clean_path_vtk = clean_base + ".vtk"
    clean_path_obj = clean_base + ".obj"

    meshin = pv.read(meshpath)
    if meshin.n_points == 0 or meshin.n_cells == 0:
        raise ValueError("Loaded mesh is empty or invalid.")

    try:
        meshfix = pymeshfix.MeshFix(meshin)
        meshfix.repair()
    except Exception as e:
        raise RuntimeError(f"Mesh repair failed: {e}")

    if meshfix.mesh.n_points == 0 or meshfix.mesh.n_cells == 0:
        raise RuntimeError("Repaired mesh is empty or invalid.")

    try:
        # Save the repaired mesh as VTK
        meshfix.mesh.save(clean_path_vtk)

        # Also save an OBJ version for compatibility
        pv.save_meshio(clean_path_obj, meshfix.mesh, "obj")
    except Exception as e:
        raise RuntimeError(f"Failed to save cleaned mesh: {e}")

    return mesh_dir, clean_path_vtk  # Return path to VTK


def _map_mesh(meshpath: str, clean_path: str) -> Any:
    """
    Map point data from the original mesh to the cleaned mesh.

    :param meshpath:  Path to the original mesh (with data arrays)
    :param clean_path: Path to the cleaned mesh (to receive data arrays)
    :return: Cleaned mesh polydata with mapped point data
    """
    if not isinstance(meshpath, str) or not meshpath:
        raise ValueError("meshpath must be a non-empty string.")
    if not isinstance(clean_path, str) or not clean_path:
        raise ValueError("clean_path must be a non-empty string.")

    print(f"Mapping data from {meshpath} to {clean_path}")

    try:
        # Read both original and cleaned meshes
        mesh_with_data = smart_reader(meshpath)
        mesh_clean = smart_reader(clean_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read mesh files: {e}")

    try:
        # Transfer all point data arrays from mesh_with_data to mesh_clean
        mapped_mesh = point_array_mapper(mesh_with_data, mesh_clean, "all")
        return mapped_mesh
    except Exception as e:
        raise RuntimeError(f"Data mapping failed: {e}")


def _get_orifices(atrium: str) -> List[str]:
    """
    Return the list of orifices for the given atrium.

    :param atrium: 'LA', 'RA', or 'LA_RA'
    :return: List of orifice names
    """
    if atrium == "LA":
        orifices = [
            "mitral valve",
            "left inferior pulmonary vein",
            "left superior pulmonary vein",
            "right inferior pulmonary vein",
            "right superior pulmonary vein"
        ]
    elif atrium == "RA" or atrium == "LA_RA":
        orifices = [
            "tricuspid valve",
            "inferior vena cava",
            "superior vena cava",
            "coronary sinus"
        ]
    else:
        raise ValueError(f"Unknown/Unsupported atrium type for orifices: {atrium}")

    return orifices


def open_orifices_manually(
        meshpath: str,
        atrium: str,
        MRI: Union[int, bool],
        scale: float = 1.0,
        size: float = 30,
        min_cutting_radius: float = 7.5,
        max_cutting_radius: float = 17.5,
        LAA: Union[str, int] = "",
        RAA: Union[str, int] = "",
        debug: int = 0
) -> Tuple[str, int]:
    """
    Open atrial orifices manually by cleaning the mesh, mapping data,
    letting user pick points, cutting holes, and identifying apex.

    :param meshpath: Path to the input mesh file
    :param atrium: 'LA', 'RA', or 'LA_RA'
    :param MRI: 1 if input is an MRI segmentation, else 0
    :param scale: Scaling factor if mesh units are not mm
    :param size: Compatibility placeholder (unused)
    :param min_cutting_radius: Radius for non-valve orifices in mm
    :param max_cutting_radius: Radius for valves in mm
    :param LAA: Placeholder for LAA apex ID (unused)
    :param RAA: Placeholder for RAA apex ID (unused)
    :param debug: Debug flag (0 or 1)
    :return: Tuple of:
        - Path to the cut VTK file
        - Apex point ID on the cut mesh
    """
    print(f"--- Starting Manual Orifice Opening for {atrium} ---")

    # Clean the mesh and obtain the cleaned VTK file
    try:
        full_path, clean_path = _clean_mesh(meshpath, atrium)
    except Exception as e:
        raise RuntimeError(f"Mesh cleaning failed: {e}")

    # Map data from the original mesh onto the cleaned mesh
    try:
        mesh_mapped = _map_mesh(meshpath, clean_path)
    except Exception as e:
        raise RuntimeError(f"Mesh mapping failed: {e}")

    current_mesh_vtk = mesh_mapped

    # Process each orifice by user point picking and cutting
    try:
        orifices = _get_orifices(atrium)
    except Exception as e:
        raise RuntimeError(f"Orifice list error: {e}")

    for r_idx, r_name in enumerate(orifices):
        print(f"Processing orifice {r_idx + 1}/{len(orifices)}: {r_name}")
        try:
            mesh_pv_for_picking = pv.PolyData(current_mesh_vtk)

            if mesh_pv_for_picking.n_points == 0:
                raise RuntimeError(f"Mesh empty before picking {r_name}")

            picked_pt = pick_point(mesh_pv_for_picking, f"center of the {r_name}")

            if picked_pt is None:
                raise RuntimeError(f"Picking cancelled or failed for {r_name}")

        except Exception as e:
            raise RuntimeError(f"Point picking failed for {r_name}: {e}")

        # Determine radius based on orifice type
        if 'valve' in r_name.lower():
            selected_radius = max_cutting_radius
        else:
            selected_radius = min_cutting_radius
        print(f"Cutting '{r_name}' with radius {selected_radius} at {picked_pt}")


        try:
            current_mesh_vtk = cut_mesh_with_radius(current_mesh_vtk, picked_pt, selected_radius)
            if current_mesh_vtk is None or current_mesh_vtk.GetNumberOfPoints() == 0:
                raise RuntimeError("Mesh empty after cutting.")
        except Exception as e:
            raise RuntimeError(f"Cutting failed for {r_name}: {e}")

    try:
        model_final_vtk = extract_largest_region(current_mesh_vtk)
        if model_final_vtk is None or model_final_vtk.GetNumberOfPoints() == 0:
            raise RuntimeError("Mesh empty after extracting largest region.")
    except Exception as e:
        raise RuntimeError(f"Largest region extraction failed: {e}")

    # Define final output path
    cutted_path = os.path.join(full_path, f"{atrium}_cutted.vtk")
    print(f"Saving final cut mesh to: {cutted_path}")
    try:
        vtk_polydata_writer(cutted_path, model_final_vtk)
    except Exception as e:
        raise RuntimeError(f"Failed to save cut mesh: {e}")

    # Manually pick the appendage apex on the final cut mesh
    try:
        mesh_pv_final = pv.PolyData(model_final_vtk)
        apex_coord = pick_point(mesh_pv_final, f"{atrium} appendage apex")
        if apex_coord is None:
            raise RuntimeError("Apex point picking cancelled or failed.")
    except Exception as e:
        raise RuntimeError(f"Error during apex picking setup: {e}")

    # Find the closest point ID on the final VTK model
    try:
        apex_id = find_closest_point(model_final_vtk, apex_coord)
        if apex_id < 0:
            raise RuntimeError("Could not find closest point for apex.")
    except Exception as e:
        raise RuntimeError(f"Apex ID determination failed: {e}")

    return cutted_path, apex_id


def run_standalone():
    args = parser().parse_args()
    try:
        cut_mesh, final_apex_id = open_orifices_manually(
            args.mesh,
            args.atrium,
            args.MRI,
            args.scale,
            30,
            args.min_cutting_radius,
            args.max_cutting_radius,
            "",
            "",
            args.debug
        )
        print(f"Standalone run complete. Cut mesh: {cut_mesh}, Apex ID: {final_apex_id}")
    except Exception as e:
        print(f"Standalone execution failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    run_standalone()
