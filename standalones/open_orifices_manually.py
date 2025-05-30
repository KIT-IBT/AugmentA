#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODIFIED version of open_orifices_manually.py
- Removed call to old extract_rings.run()
- Returns cut mesh path and apex_id
"""
import argparse
import os
from typing import Any, Tuple, List, Dict, Union

import pymeshfix
import pyvista as pv
import vtk
import sys

from vtk_opencarp_helper_methods.AugmentA_methods.point_selection import pick_point
from vtk_opencarp_helper_methods.vtk_methods.exporting import vtk_polydata_writer
from vtk_opencarp_helper_methods.vtk_methods.finder import find_closest_point
from vtk_opencarp_helper_methods.vtk_methods.helper_methods import cut_mesh_with_radius
from vtk_opencarp_helper_methods.vtk_methods.mapper import point_array_mapper
from vtk_opencarp_helper_methods.vtk_methods.reader import smart_reader
from vtk_opencarp_helper_methods.AugmentA_methods.vtk_operations import extract_largest_region



pv.set_plot_theme('dark')


def parser() -> argparse.ArgumentParser:
    # Keep parser as is
    parser = argparse.ArgumentParser(description='Cut veins manually')
    parser.add_argument('--mesh', type=str, default="", help='path to mesh')
    parser.add_argument('--atrium', type=str, default="", help='write LA or RA')
    # Add other arguments if needed by the function logic itself
    parser.add_argument('--min_cutting_radius', type=float, default=7.5, help='radius to cut veins/valves in mm')
    parser.add_argument('--max_cutting_radius', type=float, default=17.5, help='radius to cut veins/valves in mm')
    parser.add_argument('--scale', type=float, default=1.0, help='scaling factor (if mesh units != mm)')
    parser.add_argument('--MRI', type=int, default=0, help='set to 1 if the input is an MRI segmentation')
    parser.add_argument('--debug', type=int, default=0, help='debug flag')
    # LAA/RAA args are passed to the function but not used internally here for parsing
    return parser


def _clean_mesh(meshpath: str, atrium: str) -> Tuple[str, str]:
    """
    Clean the mesh from holes and self-intersecting triangles.

    Returns:
        full_path: Directory containing the mesh.
        clean_path: Path to the cleaned VTK file.
    """
    if not os.path.exists(meshpath):
        raise FileNotFoundError(f"Mesh file {meshpath} not found.")

    mesh_dir = os.path.dirname(meshpath)
    clean_base = os.path.join(mesh_dir, f"{atrium}_clean")
    clean_path_vtk = clean_base + ".vtk"
    clean_path_obj = clean_base + ".obj"

    meshin = pv.read(meshpath)
    meshfix = pymeshfix.MeshFix(meshin)
    meshfix.repair()

    try:
        meshfix.mesh.save(clean_path_vtk)
        # Also save OBJ as it might be needed by other steps implicitly
        pv.save_meshio(clean_path_obj, meshfix.mesh, "obj")
        print(f"  Cleaned mesh saved.")
    except Exception as e:
        raise RuntimeError(f"Error saving cleaned mesh: {e}")
    return mesh_dir, clean_path_vtk  # Return path to VTK


def _map_mesh(meshpath: str, clean_path: str) -> Any:
    """
    Map point data from the original mesh to the cleaned mesh.
    """
    print(f"  Mapping data from {meshpath} to {clean_path}")
    try:
        mesh_with_data = smart_reader(meshpath)
        mesh_clean = smart_reader(clean_path)
        mapped_mesh = point_array_mapper(mesh_with_data, mesh_clean, "all")
        print(f"  Data mapping complete.")
        return mapped_mesh
    except Exception as e:
        raise RuntimeError(f"Error during data mapping: {e}")


def _get_orifices(atrium: str) -> List[str]:
    """
    Return the list of orifices for the given atrium.
    """
    if atrium == "LA":
        orifices = ['mitral valve',
                    'left inferior pulmonary vein',
                    'left superior pulmonary vein',
                    'right inferior pulmonary vein',
                    'right superior pulmonary vein']

    elif atrium == "RA" or atrium == "LA_RA":
        orifices = ['tricuspid valve',
                    'inferior vena cava',
                    'superior vena cava',
                    'coronary sinus']

    else:
        raise ValueError(f"Unknown/Unsupported atrium type for orifices: {atrium}")

    return orifices

def open_orifices_manually(meshpath: str,
                           atrium: str,
                           MRI: Union[int, bool],  # Keep args even if not used directly here
                           scale: float = 1.0,
                           size: float = 30,  # Keep args even if not used directly here
                           min_cutting_radius: float = 7.5,
                           max_cutting_radius: float = 17.5,
                           LAA: Union[str, int] = "",  # Keep args even if not used directly here
                           RAA: Union[str, int] = "",  # Keep args even if not used directly here
                           debug: int = 0) -> Tuple[str, int]:  # MODIFIED: Return path and apex_id
    """
    Open atrial orifices manually by cleaning the mesh, mapping data,
    letting user pick points, cutting holes, and identifying apex.
    MODIFIED: Returns the path to the cut mesh and the apex ID.

    Returns:
        Tuple[str, int]: Path to the cut VTK file, apex ID.
    """
    print(f"--- Starting Manual Orifice Opening for {atrium} ---")
    # Clean the mesh
    full_path, clean_path = _clean_mesh(meshpath, atrium)

    # Map point data to the cleaned mesh
    mesh_mapped = _map_mesh(meshpath, clean_path)
    current_mesh_vtk = mesh_mapped  # Start cutting from mapped mesh

    # Process each orifice: pick a point and cut the mesh with the appropriate radius
    orifices = _get_orifices(atrium)
    print("Processing orifices...")
    for r_idx, r_name in enumerate(orifices):
        print(f"  Processing orifice {r_idx + 1}/{len(orifices)}: {r_name}")
        try:
            mesh_pv_for_picking = pv.PolyData(current_mesh_vtk)
            if mesh_pv_for_picking.n_points == 0:
                raise ValueError(f"Mesh became empty before picking {r_name}")
            picked_pt = pick_point(mesh_pv_for_picking, f"center of the {r_name}")
            if picked_pt is None:
                raise RuntimeError(f"Point picking cancelled or failed for {r_name}.")
        except Exception as e:
            raise RuntimeError(f"Error during point picking setup for {r_name}: {e}")

        # Determine radius based on orifice type
        if 'valve' in r_name.lower():
            selected_radius = max_cutting_radius
        else:
            selected_radius = min_cutting_radius
        print(f"    Cutting '{r_name}' with radius {selected_radius} at {picked_pt}")

        # Cut the VTK object
        try:
            current_mesh_vtk = cut_mesh_with_radius(current_mesh_vtk, picked_pt, selected_radius)
            if current_mesh_vtk is None or current_mesh_vtk.GetNumberOfPoints() == 0:
                raise ValueError("Mesh became empty after cutting.")
        except Exception as e:
            raise RuntimeError(f"Error cutting mesh for {r_name}: {e}")

    # Extract largest region after all cuts
    print("Extracting largest connected region...")
    model_final_vtk = extract_largest_region(current_mesh_vtk)
    if model_final_vtk is None or model_final_vtk.GetNumberOfPoints() == 0:
        raise RuntimeError("Mesh empty after extracting largest region.")

    # Define final output path
    cutted_path = os.path.join(full_path, f"{atrium}_cutted.vtk")
    print(f"Saving final cut mesh to: {cutted_path}")
    vtk_polydata_writer(cutted_path, model_final_vtk)

    # Manually pick the appendage apex on the final cut mesh
    print("Select appendage apex...")
    try:
        mesh_pv_final = pv.PolyData(model_final_vtk)
        apex_coord = pick_point(mesh_pv_final, f"{atrium} appendage apex")
        if apex_coord is None:
            raise RuntimeError("Apex point picking cancelled or failed.")
    except Exception as e:
        raise RuntimeError(f"Error during apex picking setup: {e}")

    # Find the closest point ID on the final VTK model
    print("Finding closest point ID for apex...")
    apex_id = find_closest_point(model_final_vtk, apex_coord)
    if apex_id < 0:
        raise RuntimeError("Could not find closest point for selected apex.")
    print(f"Apex ID determined: {apex_id}")


    print(f"--- Manual Orifice Opening Finished ---")
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

