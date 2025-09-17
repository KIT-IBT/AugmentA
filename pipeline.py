# pipeline.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:55:02 2021

@author: Luca Azzolin

Copyright 2021 Luca Azzolin

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

"""

# TODO: !!!Done but needs testing!!! Allow appendage point to be provided from a text file instead of manual picking.
# TODO: Enable or disable steps by choice so that we resample first for X amount of meshes, then the user can pick the apex point.

import os
import sys
import traceback
from string import Template
from typing import Dict, Tuple, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import cKDTree

from standalones.open_orifices_with_curvature import open_orifices_with_curvature
from standalones.open_orifices_manually import open_orifices_manually
from standalones.prealign_meshes import prealign_meshes
from standalones.getmarks import get_landmarks
from standalones.create_SSM_instance import create_SSM_instance
from standalones.resample_surf_mesh import resample_surf_mesh

from Atrial_LDRBM.LDRBM.Fiber_LA import la_main
from Atrial_LDRBM.LDRBM.Fiber_RA import ra_main

from vtk_openCARP_methods_ibt.AugmentA_methods.point_selection import pick_point, pick_point_with_preselection
from vtk_openCARP_methods_ibt.vtk_methods.filters import apply_vtk_geom_filter
from vtk_openCARP_methods_ibt.vtk_methods.mapper import mapp_ids_for_folder
from vtk_openCARP_methods_ibt.vtk_methods.normal_orientation import are_normals_outside
from vtk_openCARP_methods_ibt.vtk_methods.reader import smart_reader

from Atrial_LDRBM.Generate_Boundaries.atrial_boundary_generator import AtrialBoundaryGenerator
from Atrial_LDRBM.Generate_Boundaries.workflow_paths import WorkflowPaths

EXAMPLE_DESCRIPTIVE_NAME = 'AugmentA: Patient-specific Augmented Atrial model Generation Tool'
EXAMPLE_AUTHOR = 'Luca Azzolin <luca.azzolin@kit.edu>'

LAA_APEX_ID_FOR_SSM = 6329
RAA_APEX_ID_FOR_SSM = 21685

pv.set_plot_theme('dark')

# Determine number of CPU cores for parallel processing
n_cpu = os.cpu_count()
if n_cpu is not None and n_cpu % 2 == 0:
    n_cpu = int(n_cpu / 2)


# --- Helper Functions ---
def _load_apex_ids(csv_base: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Loads LAA and RAA IDs from '<csv_base>_mesh_data.csv'.
    :param csv_base: Base path (without extension) used to form '<csv_base>_mesh_data.csv'
    :return: Tuple containing (laa_id, raa_id). Each is an integer if found or None if missing.
    """
    csv_path = f"{csv_base}_mesh_data.csv"
    if not os.path.exists(csv_path):
        return None, None

    try:
        df = pd.read_csv(csv_path)
        laa = int(df["LAA_id"][0]) if "LAA_id" in df.columns and pd.notna(df["LAA_id"][0]) else None
        raa = int(df["RAA_id"][0]) if "RAA_id" in df.columns and pd.notna(df["RAA_id"][0]) else None
        return laa, raa
    except FileNotFoundError:
        return None, None
    except pd.errors.EmptyDataError:
        return None, None
    except (ValueError, KeyError, IndexError):
        return None, None


def _load_apex_ids_from_file(filepath: str) -> Dict[str, int]:
    """
    Reads a CSV file to load appendage apex IDs. The CSV file must have 'atrium' and 'id' columns.
    :param filepath: Path to the CSV file.
    :return: A dictionary mapping atrium names ('LAA', 'RAA') to integer IDs.
    """
    if not isinstance(filepath, str) or not filepath:
        raise ValueError("filepath must be a non-empty string.")
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Apex ID file not found: {filepath}")

    try:
        df = pd.read_csv(filepath)
        if "atrium" not in df.columns or "id" not in df.columns:
            raise ValueError("CSV must contain 'atrium' and 'id' columns.")

        # Set 'atrium' column as index and convert the 'id' column to a dictionary
        apex_ids = df.set_index('atrium')['id'].to_dict()

        # Ensure IDs are integers
        for key, value in apex_ids.items():
            apex_ids[key] = int(value)

        return apex_ids
    except Exception as e:
        raise RuntimeError(f"Error parsing apex file {filepath}: {e}")


def _save_apex_ids(csv_base: str, ids: Dict[str, int]) -> None:
    """
    Saves apex IDs to '<csv_base>_mesh_data.csv'.
    :param csv_base: Base path (without extension) where '<csv_base>_mesh_data.csv' will be written
    :param ids: Dictionary with keys 'LAA_id' and/or 'RAA_id' mapping to integer IDs
    :return: None
    """
    df = pd.DataFrame(ids)
    df.to_csv(f"{csv_base}_mesh_data.csv", float_format="%.2f", index=False)


def _ensure_obj_available(base_path_no_ext: str, original_extension: str = ".vtk") -> str:
    """
    Ensures a .obj file exists for the given base path. Converts from .vtk or .ply if .obj is missing.
    :param base_path_no_ext: Base path without extension (e.g., '/path/to/mesh')
    :param original_extension: Original mesh extension to look for if .obj is not present (default: ".vtk")
    :return: Path to the .obj file (e.g., '/path/to/mesh.obj')
    """
    base_path = Path(base_path_no_ext)
    obj_path = base_path.with_suffix(".obj")

    # Return if OBJ already exists
    if obj_path.exists():
        return str(obj_path)

    # Define source candidates in priority order
    source_candidates = [
        base_path.with_suffix(original_extension),
        base_path.with_suffix(".vtk"),
        base_path.with_suffix(".ply")
    ]

    # Find first existing source file (excluding obj_path itself)
    for source_path in source_candidates:
        if source_path.exists() and source_path != obj_path:
            print(f"Converting {source_path} to {obj_path}")
            try:
                pv_mesh = pv.read(str(source_path))
                pv.save_meshio(str(obj_path), pv_mesh, file_format="obj")
                return str(obj_path)
            except Exception as e:
                raise RuntimeError(f"Failed to convert {source_path} to OBJ: {e}") from e

    # No suitable source found
    available_files = [p for p in source_candidates if p.exists()]
    raise FileNotFoundError(
        f"Cannot ensure OBJ file: {obj_path}. "
        f"No suitable source found for base '{base_path_no_ext}'. "
        f"Available files: {available_files if available_files else 'None'}"
    )


def _setup(args) -> Tuple[WorkflowPaths, AtrialBoundaryGenerator]:
    """Initializes and returns the path and boundary generator objects."""
    paths = WorkflowPaths(initial_mesh_path=args.mesh, atrium=args.atrium)
    args.SSM_file = os.path.abspath(args.SSM_file)
    args.SSM_basename = os.path.abspath(args.SSM_basename)

    if args.normals_outside < 0:
        polydata = smart_reader(str(paths.initial_mesh))
        args.normals_outside = int(are_normals_outside(polydata))

    generator = AtrialBoundaryGenerator(
        mesh_path=str(paths.initial_mesh),
        la_apex=getattr(args, "LAA", None),
        ra_apex=getattr(args, "RAA", None),
        la_base=getattr(args, "LAA_base", None),
        ra_base=getattr(args, "RAA_base", None),
        debug=bool(args.debug)
    )
    return paths, generator


# pipeline.py
def _prepare_surface(paths: WorkflowPaths, generator: AtrialBoundaryGenerator, args) -> int:
    """
    Handle all surface preparation steps: epi/endo separation,
    orifice opening, and THEN apex picking on the final geometry.
    This unifies the manual and automated workflows.
    """
    apex_id_for_resampling = -1

    # --- Step 1: Prepare Surface Geometry (Cut or Separate) ---
    if args.closed_surface:
        generator.load_element_tags(csv_filepath=args.tag_csv)
        generator.separate_epi_endo(tagged_volume_mesh_path=str(paths.initial_mesh), atrium=args.atrium)
        paths._update_stage('epi_separated', base_path=str(paths.closed_surface_epi_mesh))

    elif args.open_orifices:
        orifice_func = open_orifices_with_curvature if args.use_curvature_to_open else open_orifices_manually
        print(f"Calling {orifice_func.__name__} for mesh='{args.mesh}', atrium='{args.atrium}'...")

        orifice_params = {'meshpath': str(paths.initial_mesh),
                          'atrium': args.atrium,
                          'MRI': args.MRI,
                          'scale': args.scale,
                          'min_cutting_radius': getattr(args, 'min_cutting_radius', 7.5),
                          'max_cutting_radius': getattr(args, 'max_cutting_radius', 17.5),
                          'debug': args.debug}

        # Add orifice coordinates file for manual function if provided
        if hasattr(args, 'orifice_file') and args.orifice_file and not args.use_curvature_to_open:
            orifice_params['orifice_coordinates_file'] = args.orifice_file

        # Call the orifice function. Note: open_orifices_with_curvature might return a valid apex ID,
        # but open_orifices_manually now returns -1 as placeholder.
        cut_path, returned_apex_id = orifice_func(**orifice_params)

        if cut_path is None or not Path(cut_path).exists():
            raise FileNotFoundError(f"{orifice_func.__name__} failed: Invalid cut_path")

        paths._update_stage('cut', base_path=str(Path(cut_path).with_suffix('')))
        print(f"Mesh after orifice cutting: {cut_path}")

        # If the function was curvature-based, it found the apex for us.
        if returned_apex_id != -1:
            apex_id_for_resampling = returned_apex_id

    # --- Step 2: Determine Apex on the FINAL Geometry ---
    # This block now runs AFTER the mesh has been cut or separated.
    # It handles both automated (file-based) and manual apex picking.

    # Only perform apex picking if the orifice function didn't already provide it.
    if apex_id_for_resampling == -1:
        print("--- Determining Apex on Final Mesh ---")
        # Load the final mesh for apex determination
        final_mesh_path = paths.active_mesh_base.with_suffix('.vtk')
        if not final_mesh_path.exists():
            raise FileNotFoundError(f"Final mesh not found at {final_mesh_path}")

        final_mesh_polydata = apply_vtk_geom_filter(smart_reader(str(final_mesh_path)))
        final_mesh_pv = pv.PolyData(final_mesh_polydata)

        if args.apex_file:
            print(f"Loading apex ID from file: {args.apex_file}")
            apex_ids_from_file = _load_apex_ids_from_file(args.apex_file)

            # Here we use the apex ID from the file, but we assume it is valid for the final cut mesh.
            # However, since the mesh has been cut, the ID might not be valid. We need to ensure the ID is within bounds.
            if args.atrium == "LA" and "LAA" in apex_ids_from_file:
                proposed_apex_id = apex_ids_from_file["LAA"]
                # Check if the proposed apex ID is within the range of the final mesh
                if proposed_apex_id < final_mesh_polydata.GetNumberOfPoints():
                    apex_id_for_resampling = proposed_apex_id
                else:
                    print(
                        f"WARNING: Apex ID {proposed_apex_id} is out of bounds for the final mesh. Falling back to interactive picking.")
                    # Fall back to interactive picking
                    apex_coord = pick_point(final_mesh_pv, "appendage apex")
                    if apex_coord is None:
                        raise RuntimeError("Apex picking cancelled or failed.")
                    apex_id_for_resampling = int(final_mesh_pv.find_closest_point(apex_coord))
            elif args.atrium == "RA" and "RAA" in apex_ids_from_file:
                proposed_apex_id = apex_ids_from_file["RAA"]
                if proposed_apex_id < final_mesh_polydata.GetNumberOfPoints():
                    apex_id_for_resampling = proposed_apex_id
                else:
                    print(
                        f"WARNING: Apex ID {proposed_apex_id} is out of bounds for the final mesh. Falling back to interactive picking.")
                    apex_coord = pick_point(final_mesh_pv, "appendage apex")
                    if apex_coord is None:
                        raise RuntimeError("Apex picking cancelled or failed.")
                    apex_id_for_resampling = int(final_mesh_pv.find_closest_point(apex_coord))
            else:
                raise ValueError(f"Apex file does not contain ID for atrium {args.atrium}")

            print(f"Using apex ID {apex_id_for_resampling} from file on the final cut mesh.")

        else:  # Manual interactive picking
            print("No apex file provided. Starting interactive point picking on final mesh...")
            apex_coord = pick_point(final_mesh_pv, "appendage apex")
            if apex_coord is None:
                raise RuntimeError("Apex picking cancelled or failed.")

            apex_id_for_resampling = int(final_mesh_pv.find_closest_point(apex_coord))

    # --- Step 3: Update State and Save Apex Info ---
    if apex_id_for_resampling == -1:
        raise ValueError("Failed to determine a valid apex ID.")

    if args.atrium == "LA":
        generator.la_apex = apex_id_for_resampling
    elif args.atrium == "RA":
        generator.ra_apex = apex_id_for_resampling

    # Save the determined apex ID for subsequent steps
    csv_data_to_save = {}
    if generator.la_apex is not None:
        csv_data_to_save["LAA_id"] = [generator.la_apex]
    if generator.ra_apex is not None:
        csv_data_to_save["RAA_id"] = [generator.ra_apex]

    if csv_data_to_save:
        _save_apex_ids(str(paths.active_mesh_base), csv_data_to_save)
        print(f"Saved final apex ID to {paths.active_mesh_base}_mesh_data.csv")

    return apex_id_for_resampling

def _ensure_ssm_base_landmarks_exist(args: Any) -> None:
    """
    Check for SSM base landmarks and generate them if missing.

    :param args: Arguments object with attributes:
        - SSM_basename: Base path for the SSM model
        - debug: Debug flag (optional)
    :return: None
    """
    ssm_base_landmarks_file = Path(f"{args.SSM_basename}_surf") / "landmarks.json"
    if ssm_base_landmarks_file.is_file():
        print(f"Found existing SSM base landmarks: {ssm_base_landmarks_file}")
        return

    print("SSM base landmarks not found. Generating...")
    ssm_basename_obj = _ensure_obj_available(args.SSM_basename,
                                             original_extension=".vtk")

    try:
        ssm_base_generator = AtrialBoundaryGenerator(mesh_path=str(ssm_basename_obj),
                                                     la_apex=LAA_APEX_ID_FOR_SSM,
                                                     ra_apex=RAA_APEX_ID_FOR_SSM,
                                                     debug=getattr(args, "debug", False))

        ssm_base_generator.extract_rings(surface_mesh_path=str(ssm_basename_obj),
                                         output_dir=str(ssm_base_landmarks_file.parent))

        get_landmarks(args.SSM_basename, 0, 1)
        print(f"SSM base landmarks generated in {ssm_base_landmarks_file.parent}/")
    except Exception as e:
        raise RuntimeError(f"Error during ring extraction for SSM base '{ssm_basename_obj}': {e}")


def _generate_target_mesh_landmarks(paths: WorkflowPaths, args: Any) -> None:
    """
    Pre-align and generate landmarks for the target mesh.

    :param paths: WorkflowPaths tracking file paths for each stage
    :param args:  Arguments object with attributes:
                  - SSM_basename: Base name of the SSM model
                  - atrium:       'LA', 'RA', or 'LA_RA'
    :return:       None
    """
    # Step 1: Pre-align the cut mesh to the SSM base
    try:
        print(f"Pre-aligning target '{paths.cut_mesh}' to SSM base '{args.SSM_basename}'.")
        prealign_meshes(str(paths.cut_mesh), args.SSM_basename, args.atrium, 0)
    except Exception as e:
        raise RuntimeError(f"Pre-alignment failed for '{paths.cut_mesh}': {e}")

    # Step 2: Generate landmarks on the pre-aligned mesh
    try:
        print(f"Generating landmarks for target '{paths.cut_mesh}'.")
        get_landmarks(str(paths.cut_mesh), prealigned=1, scale=1)
    except Exception as e:
        raise RuntimeError(f"Landmark generation failed for '{paths.cut_mesh}': {e}")


def _create_ssm_registration_script(paths: WorkflowPaths, args: Any) -> None:
    """
    Create the SSM fitting algorithm script from a template.

    :param paths: WorkflowPaths tracking file paths for each stage
    :param args: Arguments object with attributes:
        - SSM_file: Path to the SSM H5 file
        - SSM_basename: Base name for the SSM model directory
    :return: None
    """
    os.makedirs(str(paths.ssm_target_dir), exist_ok=True)

    template_path = 'template/Registration_ICP_GP_template.txt'

    # Read the template file
    try:
        with open(template_path, 'r') as f:
            template_string = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Required template file missing: {template_path}")

    # Substitute variables into the template
    try:
        template = Template(template_string)
        script_content = template.substitute(
            SSM_file=args.SSM_file,
            SSM_dir=f"{args.SSM_basename}_surf",
            target_dir=str(paths.ssm_target_dir)
        )
    except Exception as e:
        raise RuntimeError(f"Template substitution failed: {e}")

    # Write the filled-in script to disk
    script_path = paths.ssm_target_dir / 'Registration_ICP_GP.txt'
    try:
        with open(script_path, 'w') as f:
            f.write(script_content)
        print(f"SSM Fitting script written to '{script_path}'.")
    except Exception as e:
        raise RuntimeError(f"Failed to write registration script '{script_path}': {e}")


def _resample_ssm_output_if_needed(args: Any, paths: WorkflowPaths, apex_id_for_resampling: int) -> None:
    """
    If requested, resamples the newly fitted SSM mesh.

    :param args:                   Arguments object with resampling flags and parameters
    :param paths:                  WorkflowPaths tracking mesh file paths
    :param apex_id_for_resampling: Apex ID to use for resampling
    :return:                       None
    """
    if not getattr(args, 'resample_input', False):
        return

    try:
        resample_surf_mesh(
            meshname=str(paths.active_mesh_base),  # The active mesh is now the 'fit' mesh
            target_mesh_resolution=args.target_mesh_resolution,
            find_apex_with_curv=1,
            scale=args.scale,
            apex_id=apex_id_for_resampling,
            atrium=args.atrium
        )
        paths._update_stage('resampled', base_path=str(paths.resampled_mesh))
    except Exception as e:
        raise RuntimeError(f"SSM output resampling failed for '{paths.active_mesh_base}': {e}")


def _create_ssm_instance_from_coeffs(paths: WorkflowPaths, args: Any) -> None:
    """
    Creates a new mesh instance from the SSM using a coefficients file.

    :param paths: WorkflowPaths tracking file paths for each stage
    :param args: Arguments object with attributes:
        - SSM_file: Base path to the SSM H5 file
    :return: None
    """
    coeffs_file = paths.ssm_target_dir / "coefficients.txt"
    if not coeffs_file.is_file():
        raise FileNotFoundError(
            f"Coefficients file not found: {coeffs_file}. Run the external SSM fitting process first.")

    try:
        create_SSM_instance(
            f"{args.SSM_file}.h5",
            str(coeffs_file),
            str(paths.fit_mesh.with_suffix(".obj"))
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create SSM instance from coefficients: {e}")


def _run_final_labeling_and_fibers_for_ssm(args: Any, paths: WorkflowPaths, generator: AtrialBoundaryGenerator,
                                           n_cpu: int) -> None:
    """
    Handle the final steps of labeling and fiber generation for the SSM output.

    :param args:       Arguments object with flags and parameters
    :param paths:      WorkflowPaths tracking file paths for each stage
    :param generator:  AtrialBoundaryGenerator for ring extraction
    :param n_cpu:      Number of CPU cores for fiber scripts
    :return:           None
    """
    mesh_base = paths.active_mesh_base
    laa_id, raa_id = _load_apex_ids(str(mesh_base))
    generator.la_apex = laa_id
    generator.ra_apex = raa_id
    print(f"Generator apex IDs for final processing: LAA={generator.la_apex}, RAA={generator.ra_apex}")

    if getattr(args, "resample_input", False):
        source_ext = ".ply"
    else:
        source_ext = ".obj"
    path_for_labeling_obj = _ensure_obj_available(str(mesh_base), source_ext)

    try:
        generator.extract_rings(surface_mesh_path=path_for_labeling_obj, output_dir=str(paths.surf_dir))
        print(f"Final ring extraction on SSM result '{mesh_base.name}' complete.")
    except Exception as e:
        raise RuntimeError(f"Final ring extraction failed: {e}")

    fiber_main = la_main if args.atrium == "LA" else ra_main
    try:
        fiber_main.run([
            "--mesh", str(mesh_base),
            "--np", str(n_cpu),
            "--normals_outside", str(args.normals_outside),
            "--ofmt", args.ofmt,
            "--debug", str(args.debug),
            "--overwrite-behaviour", "append"
        ])
    except Exception as e:
        raise RuntimeError(f"Fiber generation script failed: {e}")


def _run_ssm_fitting(paths: WorkflowPaths, generator: AtrialBoundaryGenerator, args: Any, apex_id_for_resampling: int,
                     n_cpu: int) -> None:
    """
    Orchestrate the SSM fitting workflow by calling modular sub-steps.

    :param paths: WorkflowPaths tracking file paths for each stage
    :param generator: AtrialBoundaryGenerator used for ring extraction
    :param args: Arguments object with flags and parameters
    :param apex_id_for_resampling: Apex ID to use if resampling is required
    :param n_cpu: Number of CPU cores for parallel tasks
    :return: None
    """

    # Stage 1: Prepare landmarks
    _ensure_ssm_base_landmarks_exist(args)
    _generate_target_mesh_landmarks(paths, args)

    # Stage 2: Create the registration script for the external fitting tool
    _create_ssm_registration_script(paths, args)

    # Stage 3: Instantiate mesh from the fitting coefficients
    _create_ssm_instance_from_coeffs(paths, args)
    paths._update_stage('fit', base_path=str(paths.fit_mesh))

    # Stage 4: Resample the new instance if requested
    _resample_ssm_output_if_needed(args, paths, apex_id_for_resampling)

    # Stage 5: Run final labeling and fiber generation on the result
    _run_final_labeling_and_fibers_for_ssm(args, paths, generator, n_cpu)


def _resample_mesh_if_needed(args: Any, paths: WorkflowPaths):
    """
    If requested, resamples the mesh and updates the pipeline path.

    :param args: Workflow arguments containing resample flags and parameters
    :param paths: WorkflowPaths object tracking mesh file paths
    :return: None
    """
    if not args.resample_input:
        print('INFO: No resampling requested (resample_input=0).')
        return

    # Check if apex information is available from any source
    apex_available = False
    apex_source = "unknown"

    # Check if apex was provided via file
    if args.apex_file:
        apex_available = True
        apex_source = "apex file"

    # Check if interactive selection is enabled
    elif args.find_appendage:
        apex_available = True
        apex_source = "interactive selection"

    # Check if apex IDs were already saved from a previous step
    else:
        laa_id, raa_id = _load_apex_ids(str(paths.active_mesh_base))
        if laa_id is not None or raa_id is not None:
            apex_available = True
            apex_source = "previously saved IDs"

    if not apex_available:
        print('WARNING: Resampling requested but no apex information available.')
        print('Provide apex via --apex-file, enable --find_appendage=1, or ensure apex IDs were saved in a previous step.')
        return

    print(f'INFO: Resampling requested and apex information available from {apex_source}.')

    laa_id, raa_id = _load_apex_ids(str(paths.active_mesh_base))
    apex_id_to_pass = -1
    if args.atrium == "LA" and laa_id is not None:
        apex_id_to_pass = laa_id
    elif args.atrium == "RA" and raa_id is not None:
        apex_id_to_pass = raa_id

    # This is a safeguard. In your test, this condition should not be met.
    if apex_id_to_pass == -1 and not args.find_appendage:
        raise ValueError("Could not load a valid apex ID for automated resampling.")

    mesh_base = paths.active_mesh_base
    print(f"INFO: Resampling mesh: '{mesh_base.name}'")

    # Ensure an OBJ exists for meshtool, converting from the original extension if needed
    source_ext = mesh_base.suffix or paths.initial_mesh_ext
    _ensure_obj_available(str(mesh_base), original_extension=source_ext)

    # Perform the resampling operation
    try:
        resample_surf_mesh(meshname=str(paths.active_mesh_base),
                           target_mesh_resolution=args.target_mesh_resolution,
                           find_apex_with_curv=0,
                           scale=args.scale,
                           apex_id=apex_id_to_pass,
                           # Use the loaded apex id instead of -1 if apex_file enabled(-1 means pick them by hand)
                           atrium=args.atrium)
    except Exception as e:
        raise RuntimeError(f"Mesh resampling failed for '{paths.active_mesh_base}': {e}")

    try:
        paths._update_stage(stage_name='resampled', base_path=str(paths.resampled_mesh))
    except Exception as e:
        raise RuntimeError(f"Failed to update workflow path after resampling: {e}")

    print(f'INFO: Resampling complete. Active mesh is now: {paths.active_mesh_base.name}')


def _update_generator_with_apex_ids(paths: WorkflowPaths, generator: AtrialBoundaryGenerator) -> None:
    """
    Load the correct apex IDs for the active mesh and set them on the generator.

    :param paths: WorkflowPaths tracking file paths for each pipeline stage
    :param generator: AtrialBoundaryGenerator instance to update with apex IDs
    :return: None
    """
    laa_id, raa_id = _load_apex_ids(str(paths.active_mesh_base))

    if laa_id is not None:
        generator.la_apex = laa_id
    if raa_id is not None:
        generator.ra_apex = raa_id

    print(f"INFO: Final apex IDs for '{paths.active_mesh_base.name}': LAA={generator.la_apex}, RAA={generator.ra_apex}")


def _execute_labeling_and_fiber_generation(args: Any, paths: WorkflowPaths, generator: AtrialBoundaryGenerator,
                                           n_cpu: int) -> None:
    """
    Run the final ring extraction and execute the external fiber scripts.

    :param args: Command-line arguments or config object
    :param paths: WorkflowPaths tracking input/output mesh paths
    :param generator: AtrialBoundaryGenerator for ring extraction
    :param n_cpu: Number of CPU cores for fiber scripts
    :return: None
    """
    # Ensure the mesh is in OBJ format for labeling
    path_for_labeling_obj = _ensure_obj_available(str(paths.active_mesh_base), paths.active_mesh_base.suffix)
    print(f"INFO: Ensuring OBJ for final labeling exists at: {path_for_labeling_obj}")

    try:
        if args.atrium == "LA_RA":
            try:
                # Ring extraction for combined LA and RA
                generator.extract_rings(surface_mesh_path=path_for_labeling_obj, output_dir=str(paths.surf_dir))
                print(f"INFO: Ring extraction for LA_RA on {path_for_labeling_obj} complete.")
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Mesh file not found for LA_RA ring extraction: {path_for_labeling_obj}") from e
            except PermissionError as e:
                raise PermissionError(f"Cannot write LA_RA rings to: {paths.surf_dir}") from e
            except Exception as e:
                raise RuntimeError(f"LA_RA ring extraction failed: {e}") from e

            # Run fiber generation for both atria
            print(f"INFO: Running LA fibers for LA_RA on mesh: {str(paths.active_mesh_base)}")
            args.atrium = "LA"
            la_main.run(
                ["--mesh", str(paths.active_mesh_base),
                 "--np", str(n_cpu),
                 "--normals_outside", str(args.normals_outside),
                 "--ofmt", args.ofmt,
                 "--debug", str(args.debug),
                 "--overwrite-behaviour",
                 "append"]
            )

            print(f"INFO: Running RA fibers for LA_RA on mesh: {str(paths.active_mesh_base)}")
            args.atrium = "RA"
            ra_main.run(
                ["--mesh", str(paths.active_mesh_base),
                 "--np", str(n_cpu),
                 "--normals_outside", str(args.normals_outside),
                 "--ofmt", args.ofmt,
                 "--debug", str(args.debug),
                 "--overwrite-behaviour",
                 "append"]
            )

            # Scale and convert the final LA_RA output
            args.atrium = "LA_RA"
            scale_val = 1000 * float(args.scale)
            input_mesh_carp_txt = str(paths.final_bilayer_mesh('').with_suffix(''))  # Get base path without extension
            output_mesh_carp_txt_um = f"{input_mesh_carp_txt}_um"
            cmd1 = (f"meshtool convert "
                    f"-imsh={input_mesh_carp_txt} "
                    f"-ifmt=carp_txt "
                    f"-omsh={output_mesh_carp_txt_um} "
                    f"-ofmt=carp_txt "
                    f"-scale={scale_val}")

            cmd2 = (f"meshtool convert "
                    f"-imsh={output_mesh_carp_txt_um} "
                    f"-ifmt=carp_txt "
                    f"-omsh={output_mesh_carp_txt_um} "
                    f"-ofmt=vtk")

            os.system(cmd1)
            os.system(cmd2)

        elif args.atrium == "LA":
            print(f"INFO: LA path (non-SSM). Labeling and preparing fibers for: {path_for_labeling_obj}")
            print(f"INFO: Using LAA: {generator.la_apex} for labeling.")

            try:
                generator.extract_rings(surface_mesh_path=path_for_labeling_obj,
                                        output_dir=str(paths.surf_dir))
                print(f"INFO: Ring extraction for LA_RA on {path_for_labeling_obj} complete.")
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Mesh file not found for LA_RA ring extraction: {path_for_labeling_obj}") from e
            except PermissionError as e:
                raise PermissionError(f"Cannot write LA_RA rings to: {paths.surf_dir}") from e
            except Exception as e:
                raise RuntimeError(f"LA_RA ring extraction failed: {e}") from e

            if args.closed_surface:
                # Generate volumetric mesh from combined wall
                combined_wall_path = paths.initial_mesh_base.with_name(
                    f"{paths.initial_mesh_base.name}_{args.atrium}.vtk")
                generator.generate_mesh(input_surface_path=str(combined_wall_path))

                volumetric_mesh_path_vtk = str(
                    paths.closed_surface_vol_mesh.with_suffix('.vtk'))  # Output of generate_mesh

                print(f"INFO: Surface ID generation for LA volumetric mesh: {volumetric_mesh_path_vtk}")
                generator.generate_surf_id(
                    volumetric_mesh_path=volumetric_mesh_path_vtk,
                    atrium=args.atrium,
                    resampled=args.resample_input
                )

                # The active mesh for fiber generation is now the volumetric one.
                paths._update_stage(stage_name='volumetric', base_path=str(paths.closed_surface_vol_mesh))

                # This logic for mapping IDs from a resampled surf dir to the new vol surf dir is preserved.
                resampled_suffix_for_map = "_res" if args.resample_input else ""
                old_surf_dir = paths.active_mesh_base.with_name(
                    f"{paths.closed_surface_epi_mesh.name}{resampled_suffix_for_map}_surf")
                new_surf_dir = paths.surf_dir  # The surf_dir for the volumetric mesh

                la_main.run(
                    ["--mesh", str(paths.active_mesh_base),
                     "--np", str(n_cpu),
                     "--normals_outside", str(0),
                     "--mesh_type", "vol",
                     "--ofmt", args.ofmt,
                     "--debug", str(args.debug),
                     "--overwrite-behaviour",
                     "append"]
                )

            else:
                la_main.run(
                    ["--mesh", str(paths.active_mesh_base),
                     "--np", str(n_cpu),
                     "--normals_outside", str(args.normals_outside),
                     "--ofmt", args.ofmt,
                     "--debug", str(args.debug),
                     "--overwrite-behaviour",
                     "append"]
                )

                scale_val_la = 1000 * float(args.scale)
                input_mesh_carp_txt_LA = str(paths.final_bilayer_mesh('').with_suffix(''))
                output_mesh_carp_txt_um_LA = f"{input_mesh_carp_txt_LA}_um"

                cmd1_la = (f"meshtool convert "
                           f"-imsh={input_mesh_carp_txt_LA} "
                           f"-ifmt=carp_txt "
                           f"-omsh={output_mesh_carp_txt_um_LA} "
                           f"-ofmt=carp_txt "
                           f"-scale={scale_val_la}")

                cmd2_la = (f"meshtool convert "
                           f"-imsh={output_mesh_carp_txt_um_LA} "
                           f"-ifmt=carp_txt "
                           f"-omsh={output_mesh_carp_txt_um_LA} "
                           f"-ofmt=vtk")

                os.system(cmd1_la)
                os.system(cmd2_la)

        elif args.atrium == "RA":
            try:
                if args.closed_surface:
                    generator.extract_rings_top_epi_endo(
                        surface_mesh_path=str(paths.active_mesh_base),
                        output_dir=str(paths.surf_dir)
                    )
                else:
                    generator.extract_rings(
                        surface_mesh_path=str(paths.active_mesh_base),
                        output_dir=str(paths.surf_dir)
                    )
            except Exception as e:
                raise RuntimeError(f"RA ring extraction failed: {e}") from e

            if args.closed_surface:
                combined_wall_path = paths.initial_mesh_base.with_name(
                    f"{paths.initial_mesh_base.name}_{args.atrium}.vtk")
                generator.generate_mesh(input_surface_path=str(combined_wall_path))

                volumetric_mesh_path_vtk = str(paths.closed_surface_vol_mesh.with_suffix('.vtk'))
                generator.generate_surf_id(
                    volumetric_mesh_path=volumetric_mesh_path_vtk,
                    atrium=args.atrium,
                    resampled=False  # `resampled` is always False for RA in original code
                )

                paths._update_stage(stage_name='volumetric', base_path=str(paths.closed_surface_vol_mesh))

                ra_main.run(
                    ["--mesh", str(paths.active_mesh_base),
                     "--np", str(n_cpu),
                     "--normals_outside", "0",
                     "--mesh_type", "vol",
                     "--ofmt", args.ofmt,
                     "--debug", str(args.debug),
                     "--overwrite-behaviour", "append"]
                )
            else:  # RA, not closed_surface
                ra_main.run(
                    ["--mesh", str(paths.active_mesh_base),
                     "--np", str(n_cpu),
                     "--normals_outside", str(args.normals_outside),
                     "--ofmt", args.ofmt,
                     "--debug", str(args.debug),
                     "--overwrite-behaviour", "append"]
                )

        else:
            raise ValueError(f"Unknown atrium type: {args.atrium}")

    except Exception as e:
        raise RuntimeError(f"Labeling and fiber generation failed: {e}") from e


def _run_fiber_generation(paths: WorkflowPaths, generator: AtrialBoundaryGenerator, args: Any, n_cpu: int) -> None:
    """
    Orchestrates the non-SSM fiber generation workflow by calling modular sub-steps.

    :param paths: WorkflowPaths object tracking mesh file paths for each stage
    :param generator: AtrialBoundaryGenerator instance for ring extraction and labeling
    :param args: Command-line arguments or config with flags and parameters
    :param n_cpu: Number of CPU cores to use for parallel fiber generation
    :return: None
    """
    # Step 1: Resample the surface if requested by the user.
    _resample_mesh_if_needed(args, paths)

    # Step 2: Load the apex IDs for the current mesh (which may have been resampled).
    _update_generator_with_apex_ids(paths, generator)

    # Step 3: Run the final labeling, ring extraction, and external fiber scripts.
    _execute_labeling_and_fiber_generation(args, paths, generator, n_cpu)


def _plot_debug_results(paths: WorkflowPaths, args) -> None:
    """
    Handle final debug plotting of the resulting mesh with fibers.

    :param paths: WorkflowPaths tracking file paths for each stage
    :param args:  Command-line arguments including flags and formats
    :return:      None
    """
    # Determine if the final active mesh was a volumetric one.
    is_volumetric_plot = "_vol" in paths.active_mesh_base.name

    # The `paths` methods will correctly build the final file path based on the active stage
    # and will automatically handle the special subdirectory logic for the LA_RA case.
    if is_volumetric_plot:
        path_to_plot_file = paths.final_volumetric_mesh(args.ofmt)
    else:
        # In LA_RA case, fibers for RA part were in result_RA, and args.atrium was LA_RA for path construction
        path_to_plot_file = paths.final_bilayer_mesh(args.ofmt)

    print(f"INFO: Debug plotting. Reading file: {path_to_plot_file}")

    if not path_to_plot_file.exists():
        print(f"ERROR: Cannot plot. File not found: {path_to_plot_file}")
        # Exit the function cleanly if the expected output file doesn't exist
        return

    try:
        bil = pv.read(path_to_plot_file)

        tag_data_array = None
        if 'elemTag' in bil.point_data:
            tag_data_array = bil.point_data['elemTag']
        elif 'elemTag' in bil.cell_data:
            bil = bil.cell_data_to_point_data(pass_cell_data=True)
            if 'elemTag' in bil.point_data:
                tag_data_array = bil.point_data['elemTag']

        if tag_data_array is not None:
            processed_tags_for_viz = tag_data_array.copy()

            mask = processed_tags_for_viz > 99
            processed_tags_for_viz[mask] = 0

            mask = processed_tags_for_viz > 80
            processed_tags_for_viz[mask] = 20

            mask_gt10 = processed_tags_for_viz > 10
            processed_tags_for_viz[mask_gt10] -= 10

            mask_gt50 = processed_tags_for_viz > 50
            processed_tags_for_viz[mask_gt50] -= 50

            bil.point_data['elemTag_visualized'] = processed_tags_for_viz
            scalar_viz_key = 'elemTag_visualized'
        else:
            scalar_viz_key = None
            print("WARNING: 'elemTag' not found in mesh for plotting.")

        p = pv.Plotter(notebook=False)
        if not args.closed_surface:
            if 'fiber' in bil.point_data:
                geom = pv.Line()

                if scalar_viz_key and scalar_viz_key in bil.point_data:
                    scale_glyph_by = scalar_viz_key
                else:
                    scale_glyph_by = None

                try:
                    fibers = bil.glyph(orient="fiber", factor=0.5, geom=geom, scale=scale_glyph_by)
                    p.add_mesh(fibers,
                               show_scalar_bar=False,
                               cmap='tab20',
                               line_width=10,
                               render_lines_as_tubes=True)

                except Exception as e_glyph:
                    print(f"WARNING: Could not create fiber glyphs: {e_glyph}")
            else:
                print("WARNING: 'fiber' data not found for glyphing.")

            p.add_mesh(bil, scalars=scalar_viz_key, show_scalar_bar=False, cmap='tab20')
            if not args.no_plot:
                p.show()
            else:
                # Optionally, save a screenshot for debugging in headless mode
                output_plot_path = Path(args.mesh).parent / "debug_plot.png"
                p.screenshot(str(output_plot_path))
                print(f"INFO: Headless mode: saved debug plot to {output_plot_path}")

    except Exception as e:
        print(f"ERROR during debug plotting: {e}")


def AugmentA(args):
    try:
        print("\n--- Initializing Pipeline ---")
        paths, generator = _setup(args)
        paths.log_current_stage()

        print("\n--- Preparing Surface ---")
        apex_id = _prepare_surface(paths=paths, generator=generator, args=args)
        paths.log_current_stage()

        if args.SSM_fitting and not args.closed_surface:
            print("\n--- Running SSM Fitting ---")
            _run_ssm_fitting(paths=paths, generator=generator, args=args, apex_id_for_resampling=apex_id)
            paths.log_current_stage()
        elif not args.SSM_fitting:
            print("\n--- Running Fiber Generation ---")
            _run_fiber_generation(paths, generator, args, n_cpu)
            paths.log_current_stage()

        if args.debug:
            print("\n--- Plotting Debug Results ---")
            _plot_debug_results(paths, args)

        print("\n--- Pipeline Finished ---")

    except Exception as e:
        print("\n" + "=" * 80, file=sys.stderr)
        print("FATAL ERROR: Pipeline failed with full traceback:", file=sys.stderr)
        print("=" * 80, file=sys.stderr)

        import traceback
        traceback.print_exc(file=sys.stderr)

        print("=" * 80, file=sys.stderr)
        print(f"Error summary: {type(e).__name__}: {e}", file=sys.stderr)
        print("=" * 80, file=sys.stderr)

        tr = traceback.extract_tb(e.__traceback__)
        print(f"FATAL ERROR: Pipeline failed — {e} in {tr[-1].name} at line {tr[-1].lineno}", file=sys.stderr)

        sys.exit(1)
