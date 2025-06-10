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

# TODO: Enable or disable steps by choice so that we resample first for X amount of meshes, then the user can pick the apex point.
# TODO: Allow appendage point to be provided from a text file instead of manual picking.

import os
import sys
import subprocess
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

from vtk_opencarp_helper_methods.AugmentA_methods.point_selection import pick_point, pick_point_with_preselection
from vtk_opencarp_helper_methods.vtk_methods.filters import apply_vtk_geom_filter
from vtk_opencarp_helper_methods.vtk_methods.mapper import mapp_ids_for_folder
from vtk_opencarp_helper_methods.vtk_methods.normal_orientation import are_normals_outside
from vtk_opencarp_helper_methods.vtk_methods.reader import smart_reader

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


def _prepare_surface(paths: WorkflowPaths, generator: AtrialBoundaryGenerator, args) -> int:
    """
    Handles all surface preparation steps: epi/endo separation, orifice opening,
    or manual apex picking. Updates the paths and generator objects accordingly.
    Returns the apex_id for potential resampling.
    """
    apex_id_for_resampling = -1

    if args.closed_surface:
        generator.load_element_tags(csv_filepath=args.tag_csv)
        generator.separate_epi_endo(tagged_volume_mesh_path=str(paths.initial_mesh), atrium=args.atrium)

        # After this stage, the "active" mesh for subsequent steps is the epi mesh.
        paths._update_stage('epi_separated', base_path=str(paths.closed_surface_epi_mesh))

    else:
        if open_orifices_manually is None or open_orifices_with_curvature is None:
            raise RuntimeError("Orifice opening scripts not available")

        if args.open_orifices:
            orifice_func = open_orifices_with_curvature if args.use_curvature_to_open else open_orifices_manually
            print(f"Calling {orifice_func.__name__} for mesh='{args.mesh}', atrium='{args.atrium}'...")

            # cut_path: path to the final cut and cleaned mesh
            cut_path, apex_id = orifice_func(meshpath=str(paths.initial_mesh),
                                             atrium=args.atrium,
                                             MRI=args.MRI,
                                             scale=args.scale,
                                             min_cutting_radius=getattr(args, 'min_cutting_radius', 7.5),
                                             max_cutting_radius=getattr(args, 'max_cutting_radius', 17.5),
                                             debug=args.debug)

            if cut_path is None or not Path(cut_path).exists():
                raise FileNotFoundError(f"{orifice_func.__name__} failed: Invalid cut_path")
            if apex_id is None or apex_id < 0:
                raise ValueError(f"{orifice_func.__name__} failed: Invalid apex_id")
            print(f"Mesh after orifice cutting: {cut_path}\nApex ID picked: {apex_id}")

            # Carrying the apex_id for resampling
            apex_id_for_resampling = apex_id

            # Register the 'cut' stage as complete. The `paths` object now knows the active mesh is the one that was
            # just created.
            paths._update_stage('cut', base_path=str(Path(cut_path).with_suffix('')))

            if args.atrium == "LA":
                generator.la_apex = apex_id
                print(f"Updating generator.la_apex from {generator.la_apex} to {apex_id}.")
            elif args.atrium == "RA":
                generator.ra_apex = apex_id
                print(f"Updating generator.ra_apex from {generator.ra_apex} to {apex_id}.")

            try:
                generator.extract_rings(
                    surface_mesh_path=str(paths.active_mesh_base.with_suffix('.vtk')),
                    output_dir=str(paths.surf_dir)
                )
            except Exception as e:
                raise RuntimeError(
                    f"Error in extract_rings('{paths.active_mesh_base.with_suffix('.vtk')}'): {e}") from e

            print(f"INFO: Ring extraction complete. Outputs saved in: {paths.surf_dir}")

        else:  # not args.open_orifices:
            if args.find_appendage and not args.resample_input:
                polydata = apply_vtk_geom_filter(smart_reader(str(paths.initial_mesh)))
                if polydata is None:
                    raise FileNotFoundError(f"Could not read mesh: {paths.initial_mesh}")
                if polydata.GetNumberOfPoints() == 0:
                    raise ValueError(f"Mesh is empty (no points): {paths.initial_mesh}")

                pv_mesh = pv.PolyData(polydata)
                # Ensure points are double for cKDTree
                points_for_tree = pv_mesh.points.astype(np.double)
                tree = cKDTree(points_for_tree)

                picked_apex_data_for_csv: Dict[str, Any] = {}
                initial_apex = pick_point(pv_mesh, "appendage apex")
                if initial_apex is None:
                    raise RuntimeError("Initial 'appendage apex' picking cancelled or failed")

                _, initial_apex_id = tree.query(initial_apex)
                print(f"Initial 'appendage apex' picked: ID={initial_apex_id}")

                if args.atrium == "LA":
                    generator.la_apex = initial_apex_id
                    picked_apex_data_for_csv["LAA_id"] = [initial_apex_id]
                elif args.atrium == "RA":
                    generator.ra_apex = initial_apex_id
                    picked_apex_data_for_csv["RAA_id"] = [initial_apex_id]
                elif args.atrium == "LA_RA":
                    generator.la_apex = initial_apex_id
                    picked_apex_data_for_csv["LAA_id"] = [initial_apex_id]
                    raa_apex = pick_point_with_preselection(pv_mesh, "RA appendage apex", initial_apex)

                    pv_mesh = pv.PolyData(polydata)
                    points_for_tree = pv_mesh.points.astype(np.double)
                    tree = cKDTree(points_for_tree)
                    _, raa_apex_id = tree.query(raa_apex)
                    generator.ra_apex = raa_apex_id
                    picked_apex_data_for_csv["RAA_id"] = [raa_apex_id]

                else:
                    raise ValueError(f"Unknown atrium value '{args.atrium}'. Aborting...")

                _save_apex_ids(str(paths.initial_mesh_base), picked_apex_data_for_csv)
                print(f"Apex IDs saved to {paths.initial_mesh_base}_mesh_data.csv")

    return apex_id_for_resampling


def _run_ssm_fitting(paths: WorkflowPaths, generator: AtrialBoundaryGenerator, args, apex_id_for_resampling: int):
    """Handles the entire SSM fitting, resampling, and fiber generation workflow."""
    print(f"Using target base for SSM operations: '{paths.cut_mesh}'")

    # Generate landmarks for the SSM mean shape if they don't exist.
    ssm_base_landmarks_file = Path(f"{args.SSM_basename}_surf") / "landmarks.json"

    if not ssm_base_landmarks_file.is_file():
        print(f"SSM base landmarks not found ({ssm_base_landmarks_file}). Generating...")
        ssm_basename_obj = Path(args.SSM_basename).with_suffix('.obj')

        if not ssm_basename_obj.exists():
            # Attempt to convert from VTK if OBJ is missing.
            ssm_basename_vtk = ssm_basename_obj.with_suffix('.vtk')

            if ssm_basename_vtk.exists():
                print(f"Converting {ssm_basename_vtk} to {ssm_basename_obj} for landmark generation.")
                pv.save_meshio(str(ssm_basename_obj), pv.read(ssm_basename_vtk))
            else:
                raise ValueError(f"ERROR: SSM base mesh {ssm_basename_obj} (or .vtk) not found. Aborting...")

        try:
            # Generate landmarks for the base mesh
            print(f"Instantiating temporary AtrialBoundaryGenerator for SSM base: {ssm_basename_obj}")
            ssm_base_generator = AtrialBoundaryGenerator(mesh_path=str(ssm_basename_obj),
                                                         la_apex=LAA_APEX_ID_FOR_SSM,
                                                         ra_apex=RAA_APEX_ID_FOR_SSM,
                                                         debug=args.debug)

            ssm_base_generator.extract_rings(
                surface_mesh_path=str(ssm_basename_obj),
                output_dir=str(ssm_base_landmarks_file.parent)
            )
        except Exception as e:
            raise ValueError(f"Error during ring extraction for SSM base '{ssm_basename_obj}': {e}")

        get_landmarks(args.SSM_basename, 0, 1)
        print(f"SSM base landmarks generated in {ssm_base_landmarks_file.parent}/")
    else:
        print(f"Found existing SSM base landmarks: {ssm_base_landmarks_file}")

    # Pre-align and generate landmarks for our target mesh
    print(f"Pre-aligning target '{paths.cut_mesh}' to SSM base '{args.SSM_basename}'.")
    prealign_meshes(str(paths.cut_mesh), args.SSM_basename, args.atrium, 0)

    print(f"Generating landmarks for target '{paths.cut_mesh}'.")
    get_landmarks(str(paths.cut_mesh), 1, 1)

    # Create and run the SSM fitting script
    os.makedirs(str(paths.ssm_target_dir), exist_ok=True)
    try:
        with open('template/Registration_ICP_GP_template.txt', 'r') as f:
            tmpl_str = f.read()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required template file missing: {e.filename}") from e

    temp_obj = Template(tmpl_str)
    ssm_fit_script_content = temp_obj.substitute(SSM_file=args.SSM_file,
                                                 SSM_dir=f"{args.SSM_basename}_surf",
                                                 target_dir=str(paths.ssm_target_dir))

    ssm_fit_script_path: Path = paths.ssm_target_dir / 'Registration_ICP_GP.txt'
    with open(ssm_fit_script_path, 'w') as f:
        f.write(ssm_fit_script_content)
    print(f"SSM Fitting script written to '{ssm_fit_script_path}'.")

    # Create the patient-specific SSM instance
    coeffs_file_path = paths.ssm_target_dir / 'coefficients.txt'
    if not coeffs_file_path.is_file():
        raise FileNotFoundError(f"Coefficients file not found: {coeffs_file_path}. Run SSM first.")

    create_SSM_instance(f"{args.SSM_file}.h5",
                        str(coeffs_file_path),
                        str(paths.fit_mesh.with_suffix('.obj')))

    # The 'fit' stage is now complete. Update the state.
    paths._update_stage('fit', base_path=str(paths.fit_mesh))

    if args.resample_input:
        try:
            resample_surf_mesh(
                meshname=str(paths.active_mesh_base),
                target_mesh_resolution=args.target_mesh_resolution,
                find_apex_with_curv=1,
                scale=args.scale,
                apex_id=apex_id_for_resampling,
                atrium=args.atrium
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Mesh file not found for resampling: {paths.active_mesh_base}") from e
        except ValueError as e:
            raise ValueError(f"Invalid parameters for mesh resampling: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Mesh resampling failed for '{paths.active_mesh_base}': {e}")

        # The 'resampled' stage is now complete. Update the state.
        paths._update_stage('resampled', base_path=str(paths.resampled_mesh))

        updated_laa_ssm, updated_raa_ssm = _load_apex_ids(str(paths.resampled_mesh))
        if updated_laa_ssm is not None:
            if args.atrium == "LA" or args.atrium == "LA_RA":
                generator.la_apex = updated_laa_ssm
        if updated_raa_ssm is not None:
            if args.atrium == "RA" or args.atrium == "LA_RA":
                generator.ra_apex = updated_raa_ssm
        print(f"Generator apex IDs after SSM resampling: LAA={generator.la_apex}, RAA={generator.ra_apex}")

    # Update generator apex IDs from the mesh_data.csv of the FIT mesh
    laa_fit_csv, raa_fit_csv = _load_apex_ids(str(paths.fit_mesh))
    original_gen_laa, original_gen_raa = generator.la_apex, generator.ra_apex
    generator.la_apex = laa_fit_csv if args.atrium in ["LA", "LA_RA"] else None
    generator.ra_apex = raa_fit_csv if args.atrium in ["RA", "LA_RA"] else None

    source_ext_for_ssm_final_rings = ".ply" if args.resample_input else ".obj"
    path_for_final_ssm_rings = _ensure_obj_available(
        str(paths.active_mesh_base),
        original_extension=source_ext_for_ssm_final_rings
    )

    try:
        generator.extract_rings(surface_mesh_path=path_for_final_ssm_rings,
                                output_dir=str(paths.surf_dir))

        print(f"Final ring extraction on SSM result '{paths.active_mesh_base.name}' complete.")
        print(f"DEBUG: LAA={generator.la_apex}, RAA={generator.ra_apex}")

    except Exception as e:
        raise RuntimeError(f"Final ring extraction failed after SSM workflow: {e}") from e

    # Restore original apex IDs for the generator instance if needed elsewhere.
    generator.la_apex, generator.ra_apex = original_gen_laa, original_gen_raa

    # Run fiber generation
    fiber_main = la_main if args.atrium == "LA" else ra_main
    fiber_main.run(["--mesh", str(paths.active_mesh_base),
                    "--np", str(n_cpu),
                    "--normals_outside", str(args.normals_outside),
                    "--ofmt", args.ofmt,
                    "--debug", str(args.debug),
                    "--overwrite-behaviour", "append"])

    print(f"INFO: SSM path complete. Final active mesh is: {paths.active_mesh_base.name}")


def _run_fiber_generation(paths: WorkflowPaths, generator: AtrialBoundaryGenerator, args):
    """Handles non-SSM fiber generation, including resampling and volumetric processing."""
    if args.resample_input and args.find_appendage:
        print(f"INFO: Resampling mesh: '{paths.active_mesh_base.name}'")

        # Determine the correct file extension of the mesh we are resampling
        source_ext = paths.active_mesh_base.suffix
        # If base path has no extension, check common ones
        if not source_ext:
            if Path(str(paths.active_mesh_base) + '.vtk').exists():
                source_ext = '.vtk'
            elif Path(str(paths.active_mesh_base) + '.obj').exists():
                source_ext = '.obj'
            else:
                source_ext = paths.initial_mesh_ext  # Fallback to original

        _ensure_obj_available(str(paths.active_mesh_base), original_extension=source_ext)

        try:
            resample_surf_mesh(meshname=str(paths.active_mesh_base),
                               target_mesh_resolution=args.target_mesh_resolution,
                               find_apex_with_curv=0,
                               scale=args.scale,
                               apex_id=-1,  # Always find new apex in this workflow
                               atrium=args.atrium)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Mesh file not found for resampling: {paths.active_mesh_base}") from e
        except ValueError as e:
            raise ValueError(f"Invalid parameters for mesh resampling: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Mesh resampling failed for '{paths.active_mesh_base}': {e}")

        # The 'resampled' stage is now complete. Update the state.
        paths._update_stage(stage_name='resampled', base_path=str(paths.resampled_mesh))
        print(f'INFO: Resampling complete. Active mesh is now: {paths.active_mesh_base.name}')

        # Update apex IDs from the output of the resampling script.
        laa_from_resampled_csv, raa_from_resampled_csv = _load_apex_ids(str(paths.resampled_mesh))
        if laa_from_resampled_csv is not None:
            if args.atrium == "LA" or args.atrium == "LA_RA":
                generator.la_apex = laa_from_resampled_csv
        if raa_from_resampled_csv is not None:
            if args.atrium == "RA" or args.atrium == "LA_RA":
                generator.ra_apex = raa_from_resampled_csv
        print(
            f'INFO: Generator apex IDs updated from resampled data: LAA={generator.la_apex}, RAA={generator.ra_apex}')
    else:
        print(f'INFO: No resampling of original mesh in this specific non-SSM branch.')

    # Update apex IDs from CSV if it exists, for the current active mesh
    laa_from_pm_csv, raa_from_pm_csv = _load_apex_ids(str(paths.active_mesh_base))
    if laa_from_pm_csv is not None:
        generator.la_apex = laa_from_pm_csv
    if raa_from_pm_csv is not None:
        generator.ra_apex = raa_from_pm_csv
    print(
        f"INFO: Final labeling on '{paths.active_mesh_base.name}': LAA={generator.la_apex}, RAA={generator.ra_apex}")

    # Finalize ring extraction on the current active mesh (which may be initial, cut, or resampled)
    path_for_labeling_obj = _ensure_obj_available(str(paths.active_mesh_base), paths.active_mesh_base.suffix)
    print(f"INFO: Ensuring OBJ for final labeling exists at: {path_for_labeling_obj}")

    if args.atrium == "LA_RA":
        try:
            generator.extract_rings(surface_mesh_path=path_for_labeling_obj,
                                    output_dir=str(paths.surf_dir))
            print(f"INFO: Ring extraction for LA_RA on {path_for_labeling_obj} complete.")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Mesh file not found for LA_RA ring extraction: {path_for_labeling_obj}") from e
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

        subprocess.run(cmd1, shell=True, check=True)
        subprocess.run(cmd2, shell=True, check=True)

    if args.atrium == "LA":
        print(f"INFO: LA path (non-SSM). Labeling and preparing fibers for: {path_for_labeling_obj}")
        print(f"INFO: Using LAA: {generator.la_apex} for labeling.")

        try:
            generator.extract_rings(surface_mesh_path=path_for_labeling_obj,
                                    output_dir=str(paths.surf_dir))
            print(f"INFO: Ring extraction for LA_RA on {path_for_labeling_obj} complete.")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Mesh file not found for LA_RA ring extraction: {path_for_labeling_obj}") from e
        except PermissionError as e:
            raise PermissionError(f"Cannot write LA_RA rings to: {paths.surf_dir}") from e
        except Exception as e:
            raise RuntimeError(f"LA_RA ring extraction failed: {e}") from e

        if args.closed_surface:
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

            subprocess.run(cmd1_la, shell=True, check=True)
            subprocess.run(cmd2_la, shell=True, check=True)

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


def _plot_debug_results(paths: WorkflowPaths, args):
    """Handles the final debug plotting of the resulting mesh with fibers."""
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
            p.show()

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
            _run_fiber_generation(paths, generator, args)
            paths.log_current_stage()

        if args.debug:
            print("\n--- Plotting Debug Results ---")
            _plot_debug_results(paths, args)

        print("\n--- Pipeline Finished ---")

    except Exception as e:
        print(f"\nFATAL ERROR: Pipeline failed — {e}", file=sys.stderr)
        sys.exit(1)


