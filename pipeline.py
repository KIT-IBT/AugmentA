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
import os
from string import Template
from typing import Dict, Tuple, Any
import sys

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
from Atrial_LDRBM.Generate_Boundaries.file_manager import write_vtx_file
EXAMPLE_DESCRIPTIVE_NAME = 'AugmentA: Patient-specific Augmented Atrial model Generation Tool'
EXAMPLE_AUTHOR = 'Luca Azzolin <luca.azzolin@kit.edu>'

pv.set_plot_theme('dark')
n_cpu = os.cpu_count()
if not n_cpu % 2:
    n_cpu = int(n_cpu / 2)

def _mesh_base_and_dir(path: str) -> Tuple[str, str]:
    """Return (mesh_base_without_ext, dir_path)."""
    mesh_dir = os.path.dirname(path)
    base = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(mesh_dir, base), mesh_dir


def _load_apex_ids(csv_base: str) -> Tuple[int | None, int | None]:
    """Load LAA/RAA IDs from '<csv_base>_mesh_data.csv' if present."""
    csv_path = f"{csv_base}_mesh_data.csv"
    if not os.path.exists(csv_path):
        return None, None

    try:
        df = pd.read_csv(csv_path)
        laa = int(df["LAA_id"][0]) if "LAA_id" in df.columns and pd.notna(df["LAA_id"][0]) else None
        raa = int(df["RAA_id"][0]) if "RAA_id" in df.columns and pd.notna(df["RAA_id"][0]) else None
        return laa, raa
    except Exception:
        return None, None

def _save_apex_ids(csv_base: str, ids: Dict[str, int]) -> None:
    """Write apex ID CSV in the same format used by the legacy script."""
    df = pd.DataFrame(ids)
    df.to_csv(f"{csv_base}_mesh_data.csv", float_format="%.2f", index=False)


def _ensure_obj_available(base_path_no_ext: str, original_extension: str = ".vtk") -> str:
    """
    Ensures a .obj file exists for the given base path.
    Converts from .vtk or .ply if .obj is missing.
    Returns the path to the .obj file.
    """
    obj_path = base_path_no_ext + ".obj"
    if os.path.exists(obj_path):
        return obj_path

    source_vtk = base_path_no_ext + ".vtk"
    source_ply = base_path_no_ext + ".ply"
    source_original = base_path_no_ext + original_extension

    if os.path.exists(source_original) and source_original != obj_path:
        print(f"Converting {source_original} to {obj_path}")
        pv_mesh = pv.read(source_original)
        pv.save_meshio(obj_path, pv_mesh, file_format="obj")
    elif os.path.exists(source_vtk) and source_vtk != obj_path:
        print(f"Converting {source_vtk} to {obj_path}")
        pv_mesh = pv.read(source_vtk)
        pv.save_meshio(obj_path, pv_mesh, file_format="obj")
    elif os.path.exists(source_ply) and source_ply != obj_path:
        print(f"Converting {source_ply} to {obj_path}")
        pv_mesh = pv.read(source_ply)
        pv.save_meshio(obj_path, pv_mesh, file_format="obj")
    else:
        raise FileNotFoundError(
            f"Cannot ensure OBJ file: {obj_path}. No suitable source found for base '{base_path_no_ext}'.")


def AugmentA(args):
    # TODO: Add a variable that is not scope limited for keeping id
    # global apex_id
    apex_id_for_resampling: int = -1


    args.SSM_file = os.path.abspath(args.SSM_file)
    args.SSM_basename = os.path.abspath(args.SSM_basename)
    args.mesh = os.path.abspath(args.mesh)

    mesh_filename = os.path.basename(args.mesh)
    mesh_base, mesh_ext = os.path.splitext(mesh_filename)
    mesh_dir = os.path.dirname(args.mesh)
    meshname = os.path.join(mesh_dir, mesh_base)

    laa_csv_id, raa_csv_id = _load_apex_ids(mesh_base)

    if args.normals_outside < 0:
        args.normals_outside = int(are_normals_outside(smart_reader(args.mesh)))

    generator = AtrialBoundaryGenerator(mesh_path=args.mesh,
                                        la_apex=getattr(args, 'LAA', None),
                                        ra_apex=getattr(args, 'RAA', None),
                                        la_base=getattr(args, 'LAA_base', None),
                                        ra_base=getattr(args, 'RAA_base', None),
                                        debug=bool(args.debug))

    meshname_old = str(meshname)
    processed_mesh: str = str(meshname)

    if args.closed_surface:
        generator.load_element_tags(csv_filepath=args.tag_csv)

        try:
            generator.separate_epi_endo(tagged_volume_mesh_path=args.mesh, atrium=args.atrium)
        except Exception as e:
            print(f"ERROR during epi/endo separation: {e}")
            sys.exit(1)

        meshname = f"{meshname_old}_{args.atrium}_epi" # e.g. {orig_base}_{atrium}_epi
        processed_mesh = meshname

    else:
        if open_orifices_manually is None or open_orifices_with_curvature is None:
            print("Error: Orifice opening scripts not available.")
            sys.exit(1)

        if args.open_orifices:
            if args.use_curvature_to_open:
                orifice_func = open_orifices_with_curvature
            else:
                orifice_func = open_orifices_manually

            print(f"Calling {orifice_func.__name__} for mesh='{args.mesh}', atrium='{args.atrium}'...")

            # cut_path = path to the final cut and cleaned mesh
            cut_path, apex_id = orifice_func(meshpath=args.mesh,
                                             atrium=args.atrium,
                                             MRI=args.MRI,
                                             scale=args.scale,
                                             min_cutting_radius=getattr(args, 'min_cutting_radius', 7.5),
                                             max_cutting_radius=getattr(args, 'max_cutting_radius', 17.5),
                                             debug=args.debug)
            """
            procedural -> meshname = {mesh_dir}{atrium}_cutted
            refactored -> processed_mesh = {mesh_dir}{atrium}_cutted (=os.path.splitext(cut_path)[0])
            """

            # TODO: Use this as global apex_id later on in the program
            apex_id_for_resampling = apex_id

            if cut_path is None or not os.path.exists(cut_path) or apex_id is None or apex_id < 0:
                print(f"Error: {orifice_func.__name__} for atrium '{args.atrium}' failed. Returned path: '{cut_path}', picked_apex_id: {apex_id}. Aborting.")
                sys.exit(1)

            current_mesh_file_path = cut_path
            processed_mesh = os.path.splitext(current_mesh_file_path)[0]

            print(f"Mesh after orifice cutting by {orifice_func.__name__}: {current_mesh_file_path}")
            print(f"Apex ID picked by {orifice_func.__name__} for '{args.atrium} appendage apex ID picked': {apex_id}")

            if args.atrium == "LA":
                if generator.la_apex != apex_id:
                    print(f"Updating generator.la_apex from {generator.la_apex} to {apex_id}.")
                    generator.la_apex = apex_id
            elif args.atrium == "RA":
                if generator.ra_apex != apex_id:
                    print(f"Updating generator.ra_apex from {generator.ra_apex} to {apex_id}.")
                    generator.ra_apex = apex_id

            print(f"Calling generator.extract_rings on VTK: {current_mesh_file_path}")
            print(f"Using LAA apex ID={generator.la_apex}, RAA ID={generator.ra_apex}")

            try:
                processed_mesh_base_after_cut = os.path.splitext(current_mesh_file_path)[0]
                path_for_rings_obj_after_cut = _ensure_obj_available(processed_mesh_base_after_cut, ".vtk") # e.g. /headless/data/LA_cutted.obj
                generator.extract_rings(surface_mesh_path=path_for_rings_obj_after_cut)
                # generator.extract_rings(surface_mesh_path=current_mesh_file_path)
                print(f"INFO: Ring extraction complete. Outputs expected in directory: '{processed_mesh_base_after_cut}_surf/'.")
                processed_mesh = processed_mesh_base_after_cut
            except Exception as e:
                print(f"Error in extract_rings('{current_mesh_file_path}'): {e}")
                sys.exit(1)

            current_mesh_base = os.path.join(mesh_dir, f"{args.atrium}_cutted")
            if os.path.splitext(current_mesh_file_path)[0] != current_mesh_base:
                print(f"Path Sanity Check: Base of actual cut file '{os.path.splitext(current_mesh_file_path)[0]}' vs expected procedural base '{current_mesh_base}'. Ensure consistency if other scripts rely on exact procedural 'meshname'.")
                current_mesh_base = os.path.splitext(current_mesh_file_path)[0]

            processed_mesh = current_mesh_base

        else: # not args.open_orifices:
            if args.find_appendage and not args.resample_input:
                picked_apex_data_for_csv: Dict[str, Any] = {}
                polydata = apply_vtk_geom_filter(smart_reader(args.mesh))

                if polydata is None or polydata.GetNumberOfPoints() == 0:
                    print(f"Error: Could not read or mesh is empty: {args.mesh}. Aborting.")
                    sys.exit(1)

                pv_mesh = pv.PolyData(polydata)
                # Ensure points are double for cKDTree
                points_for_tree = pv_mesh.points.astype(np.double)
                initial_apex = pick_point(pv_mesh, "appendage apex")
                tree = cKDTree(points_for_tree)
                if initial_apex is None:
                    print("Error: Initial 'appendage apex' picking cancelled or failed. Aborting.")
                    sys.exit(1)

                _, initial_apex_id = tree.query(initial_apex)
                print(f"Initial 'appendage apex' picked: ID={initial_apex_id}")

                if args.atrium == "LA":
                    generator.la_apex = initial_apex_id
                    picked_apex_data_for_csv["LAA_id"] = generator.la_apex
                    print(f"LA apex (from initial pick) set in generator: ID={generator.la_apex}")

                elif args.atrium == "RA":
                    generator.ra_apex = initial_apex_id
                    picked_apex_data_for_csv["RAA_id"] = initial_apex_id
                    print(f"RA apex (from initial pick) set in generator: ID={generator.ra_apex}")

                elif args.atrium == "LA_RA":
                    generator.la_apex = initial_apex_id
                    picked_apex_data_for_csv["LAA_id"] = initial_apex_id
                    print(f"LAA (from initial pick) set in generator: ID={generator.la_apex}")

                    raa_apex = pick_point_with_preselection(pv_mesh, "RA appendage apex", initial_apex)

                    pv_mesh = pv.PolyData(polydata)
                    # Ensure points are double for cKDTree
                    points_for_tree = pv_mesh.points.astype(np.double)
                    tree = cKDTree(points_for_tree)

                    _, raa_apex_id = tree.query(raa_apex)

                    generator.ra_apex = raa_apex_id
                    picked_apex_data_for_csv["RAA_id"] = raa_apex_id
                    print(f"RAA (from second specific pick) set in generator: ID={generator.ra_apex}")

                else:
                    print(f"Error: Unknown args.atrium value '{args.atrium}'. Aborting.")
                    sys.exit(1)

                _save_apex_ids(meshname, picked_apex_data_for_csv)
                print(f"Apex IDs saved to {mesh_base}_mesh_data.csv")


    if args.SSM_fitting and not args.closed_surface:
        target_base_for_ssm_ops = os.path.join(mesh_dir, f"{args.atrium}_cutted")
        print(f"Using target base for SSM operations: '{target_base_for_ssm_ops}'")

        ssm_base_landmarks_file = os.path.join(args.SSM_basename + "_surf", "landmarks.json")
        if not os.path.isfile(ssm_base_landmarks_file):
            print(f"SSM base landmarks not found ({ssm_base_landmarks_file}). Generating...")
            ssm_basename_obj = args.SSM_basename + ".obj"

            if not os.path.exists(ssm_basename_obj):
                ssm_basename_vtk = args.SSM_basename + ".vtk"

                if os.path.exists(ssm_basename_vtk):
                    print(f"Converting {ssm_basename_vtk} to {ssm_basename_obj} for landmark generation.")
                    pv.save_meshio(ssm_basename_obj, pv.read(ssm_basename_vtk))
                else:
                    print(f"ERROR: SSM base mesh {ssm_basename_obj} (or .vtk) not found. Aborting.")
                    sys.exit(1)

            try:
                print(f"Instantiating temporary AtrialBoundaryGenerator for SSM base: {ssm_basename_obj}")
                # 6329 LAA apex id and 21685 RAA apex id in meanshape from Nagel et al. 2020
                hard_coded_laa_apex_id = 6329
                hard_coded_raa_apex_id = 21685
                ssm_base_generator = AtrialBoundaryGenerator(mesh_path=ssm_basename_obj,
                                                             la_apex=hard_coded_laa_apex_id,
                                                             ra_apex=hard_coded_raa_apex_id,
                                                             debug=args.debug)

                ssm_base_generator.extract_rings(surface_mesh_path=ssm_basename_obj)
            except Exception as e:
                print(f"Error during refactored ring extraction for SSM base '{ssm_basename_obj}': {e}")
                sys.exit(1)

            get_landmarks(args.SSM_basename, 0, 1)
            print(f"SSM base landmarks generated in {args.SSM_basename}_surf/")
        else:
            print(f"Found existing SSM base landmarks: {ssm_base_landmarks_file}")

        print(f"Pre-aligning target '{target_base_for_ssm_ops}' to SSM base '{args.SSM_basename}'.")
        target_mesh_for_ssm_vtk = target_base_for_ssm_ops + ".vtk"
        target_mesh_for_ssm_obj = target_base_for_ssm_ops + ".obj"
        if not (os.path.exists(target_mesh_for_ssm_vtk) or os.path.exists(target_mesh_for_ssm_obj)):
            print(f"Warning: Target mesh for prealignment ('{target_mesh_for_ssm_vtk}' or '{target_mesh_for_ssm_obj}') not found. Prealignment might fail.")
        prealign_meshes(target_base_for_ssm_ops, args.SSM_basename, args.atrium, 0)

        print(f"Generating landmarks for target '{target_base_for_ssm_ops}'.")
        try:
            get_landmarks(target_base_for_ssm_ops, 1, 1)
        except Exception as e:
            print(f"ERROR during get_landmarks for target '{target_base_for_ssm_ops}'.")
            print(f"Ensure ring files (e.g., rings_centroids.csv) exist in '{target_base_for_ssm_ops}_surf/'. Error: {e}")
            sys.exit(1)

        target_surf_dir_for_ssm = target_base_for_ssm_ops + "_surf"
        os.makedirs(target_surf_dir_for_ssm, exist_ok=True)

        try:
            with open('template/Registration_ICP_GP_template.txt', 'r') as f:
                tmpl_str = f.read()
        except FileNotFoundError:
            print("ERROR: 'template/Registration_ICP_GP_template.txt' not found. Aborting.")
            sys.exit(1)

        temp_obj = Template(tmpl_str)
        ssm_fit_script_content = temp_obj.substitute(SSM_file=args.SSM_file,
                                                     SSM_dir=args.SSM_basename + "_surf",
                                                     target_dir=target_surf_dir_for_ssm)

        ssm_fit_script_path = os.path.join(target_surf_dir_for_ssm, 'Registration_ICP_GP.txt')
        with open(ssm_fit_script_path, 'w') as f:
            f.write(ssm_fit_script_content)

        print(f"SSM Fitting script written to '{ssm_fit_script_path}'.")

        coeffs_file_path = os.path.join(target_surf_dir_for_ssm, 'coefficients.txt')
        if not os.path.isfile(coeffs_file_path):
            print(f"ERROR: Coefficients file not found: {coeffs_file_path}. Run Scalismo first. Aborting.");
            sys.exit(1)

        fitted_mesh_obj_filename = f"{args.atrium}_fit.obj"
        fitted_mesh_obj_path = os.path.join(target_surf_dir_for_ssm, fitted_mesh_obj_filename)
        create_SSM_instance(args.SSM_file + ".h5",
                            coeffs_file_path,
                            fitted_mesh_obj_path)

        fitted_mesh_base = os.path.splitext(fitted_mesh_obj_path)[0] # Base name, e.g., ..._cutted_surf/LA_fit

        # This variable will store the base name of the mesh after fitting and optional resampling.
        current_processed_mesh_base_in_ssm = ""
        if args.resample_input:
            try:
                resample_surf_mesh(meshname=fitted_mesh_base,
                                   target_mesh_resolution=args.target_mesh_resolution,
                                   find_apex_with_curv=1,
                                   scale=args.scale,
                                   apex_id=apex_id_for_resampling,
                                   atrium=args.atrium)
            except Exception as e:
                print(f"Error during resample_surf_mesh for '{fitted_mesh_base}': {e}");
                sys.exit(1)

            current_processed_mesh_base_in_ssm = fitted_mesh_base + "_res"

            updated_laa_ssm, updated_raa_ssm = _load_apex_ids(current_processed_mesh_base_in_ssm)
            if args.atrium == "LA" or args.atrium == "LA_RA":
                if updated_laa_ssm is not None:
                    generator.la_apex = updated_laa_ssm
            if args.atrium == "RA" or args.atrium == "LA_RA":
                if updated_raa_ssm is not None:
                    generator.ra_apex = updated_raa_ssm
            print(f"Generator apex IDs after SSM resampling: LAA={generator.la_apex}, RAA={generator.ra_apex}")

        else: # No resampling
            current_processed_mesh_base_in_ssm = fitted_mesh_base
            print(f"No resampling of SSM instance. Using '{current_processed_mesh_base_in_ssm}'.")

        csv_base_for_specific_ssm_labeling = fitted_mesh_base
        laa_fit_csv, raa_fit_csv = _load_apex_ids(csv_base_for_specific_ssm_labeling)

        original_gen_laa, original_gen_raa = generator.la_apex, generator.ra_apex

        generator.la_apex = laa_fit_csv if args.atrium in ["LA", "LA_RA"] else None
        generator.ra_apex = raa_fit_csv if args.atrium in ["RA", "LA_RA"] else None

        try:
            source_ext_for_ssm_final_rings = ".ply" if args.resample_input else ".obj"
            path_for_final_ssm_rings_obj = _ensure_obj_available(current_processed_mesh_base_in_ssm, source_ext_for_ssm_final_rings)
            generator.extract_rings(surface_mesh_path=path_for_final_ssm_rings_obj)
            print(f"Final ring extraction on SSM result: {current_processed_mesh_base_in_ssm}, LAA={generator.la_apex}, RAA={generator.ra_apex}")
            print("Final ring extraction after SSM processing complete.")
        except Exception as e:
            print(f"Error during final generator.extract_rings after SSM: {e}");
            sys.exit(1)

        generator.la_apex, generator.ra_apex = original_gen_laa, original_gen_raa

        if args.atrium == "LA":
            la_main.run(["--mesh", current_processed_mesh_base_in_ssm,
                         "--np", str(n_cpu),
                         "--normals_outside", str(args.normals_outside),
                         "--ofmt", args.ofmt,
                         "--debug", str(args.debug),
                         "--overwrite-behaviour",
                         "append"])

        elif args.atrium == "RA":
            ra_main.run(["--mesh", current_processed_mesh_base_in_ssm,
                         "--np", str(n_cpu),
                         "--normals_outside", str(args.normals_outside),
                         "--ofmt", args.ofmt,
                         "--debug", str(args.debug),
                         "--overwrite-behaviour",
                         "append"])

        processed_mesh = current_processed_mesh_base_in_ssm  # Update main tracker
        print(f"INFO: SSM path complete. Main 'processed_mesh' updated to: {processed_mesh}")

    elif not args.SSM_fitting:
        fiber_mesh_base = processed_mesh

        if args.resample_input and args.find_appendage:
            print(f"INFO: Conditional resampling active: Preparing to resample original mesh '{meshname_old}'.")

            _ensure_obj_available(meshname_old, mesh_ext)
            apex_id_for_this_resample = -1

            source_ext_for_current_pm = mesh_ext  # Default
            if processed_mesh == f"{meshname_old}_{args.atrium}_epi":
                source_ext_for_current_pm = ".vtk"
            elif "_cutted" in os.path.basename(processed_mesh):
                source_ext_for_current_pm = ".vtk"

            _ensure_obj_available(processed_mesh, source_ext_for_current_pm)

            _ensure_obj_available(processed_mesh, source_ext_for_current_pm)

            resample_surf_mesh(meshname=processed_mesh,
                               target_mesh_resolution=args.target_mesh_resolution,
                               find_apex_with_curv=0,
                               scale=args.scale,
                               apex_id=apex_id_for_this_resample,
                               atrium=args.atrium)

            processed_mesh = f"{processed_mesh}_res"
            fiber_mesh_base = processed_mesh
            print(f"INFO: Original mesh resampled. 'processed_mesh' is now: {processed_mesh}")

            _ensure_obj_available(processed_mesh, original_extension=".ply")
            print(f"INFO: Ensured {processed_mesh}.obj is available.")

            laa_from_resampled_csv, raa_from_resampled_csv = _load_apex_ids(processed_mesh)
            if args.atrium == "LA" or args.atrium == "LA_RA":
                generator.la_apex = laa_from_resampled_csv
            if args.atrium == "RA" or args.atrium == "LA_RA":
                generator.ra_apex = raa_from_resampled_csv
            print(f"INFO: Generator apex IDs updated from '{processed_mesh}_mesh_data.csv': LAA={generator.la_apex}, RAA={generator.ra_apex}")
        else:
            print(f"INFO: No resampling of original mesh in this specific non-SSM branch. "
                  f"'processed_mesh' ({processed_mesh}) and 'fiber_mesh_base' ({fiber_mesh_base}) remain from prior steps.")

        current_mesh_actual_extension = mesh_ext
        if processed_mesh == f"{meshname_old}_{args.atrium}_epi":
            # This mesh was created by generator.separate_epi_endo, which writes both .vtk and .obj.
            # We prefer .vtk as a source for conversion if .obj was somehow missing.
            current_mesh_actual_extension = ".vtk"
        elif "_cutted" in os.path.basename(processed_mesh):
            # This mesh was created by open_orifices_manually (refactored), which saves .vtk for the cut mesh.
            current_mesh_actual_extension = ".vtk"
        elif "_res" in os.path.basename(processed_mesh):
            # This mesh was created by resample_surf_mesh, which saves .ply.
            current_mesh_actual_extension = ".ply"
        else:
            current_mesh_actual_extension = mesh_ext

        path_for_labeling_obj = _ensure_obj_available(processed_mesh, current_mesh_actual_extension)
        print(f"INFO: Ensured OBJ file for ring extraction for all non-SSM paths: {path_for_labeling_obj}")

        # TODO: Check if procedural code does it here
        print(f"INFO: Finalizing apex IDs for labeling from: {processed_mesh}_mesh_data.csv")
        laa_from_pm_csv, raa_from_pm_csv = _load_apex_ids(processed_mesh)
        if laa_from_pm_csv is not None:
            if generator.la_apex != laa_from_pm_csv:
                print(f"INFO: Updating generator.la_apex from {generator.la_apex} to {laa_from_pm_csv} (from {processed_mesh}_mesh_data.csv)")
            generator.la_apex = laa_from_pm_csv

        if raa_from_pm_csv is not None:
            if generator.ra_apex != raa_from_pm_csv:
                print(f"INFO: Updating generator.ra_apex from {generator.ra_apex} to {raa_from_pm_csv} (from {processed_mesh}_mesh_data.csv)")
            generator.ra_apex = raa_from_pm_csv

        print(f"INFO: Apex IDs now set in generator for labeling '{processed_mesh}': LAA={generator.la_apex}, RAA={generator.ra_apex}")

        if args.atrium == "LA_RA":
            try:
                generator.extract_rings(surface_mesh_path=path_for_labeling_obj)
                print(f"INFO: Ring extraction for LA_RA on {path_for_labeling_obj} complete.")
            except Exception as e:
                print(f"ERROR during LA_RA extract_rings: {e}")
                sys.exit(1)

            print(f"INFO: Running LA fibers for LA_RA on mesh: {fiber_mesh_base}")
            args.atrium = "LA"

            la_main.run(
                ["--mesh", fiber_mesh_base,
                 "--np", str(n_cpu),
                 "--normals_outside", str(args.normals_outside),
                 "--ofmt", args.ofmt,
                 "--debug", str(args.debug),
                 "--overwrite-behaviour",
                 "append"]
            )

            print(f"INFO: Running RA fibers for LA_RA on mesh: {fiber_mesh_base}")
            args.atrium = "RA"
            ra_main.run(
                ["--mesh", fiber_mesh_base,
                 "--np", str(n_cpu),
                 "--normals_outside", str(args.normals_outside),
                 "--ofmt", args.ofmt,
                 "--debug", str(args.debug),
                 "--overwrite-behaviour",
                 "append"]
            )

            args.atrium = "LA_RA"
            scale_val = 1000 * float(args.scale)
            input_mesh_carp_txt = f"{fiber_mesh_base}_fibers/result_RA/{args.atrium}_bilayer_with_fiber"
            output_mesh_carp_txt_um = f"{input_mesh_carp_txt}_um"
            cmd1 = (f"meshtool convert "
                    f"-imsh={input_mesh_carp_txt} "
                    f"-ifmt=carp_txt "
                    f"-omsh={output_mesh_carp_txt_um} "
                    f"-ofmt=carp_txt "
                    f"-scale={scale_val}")

            input_mesh_for_vtk_conversion = output_mesh_carp_txt_um
            output_mesh_vtk_um = output_mesh_carp_txt_um
            cmd2 = (f"meshtool convert "
                    f"-imsh={input_mesh_for_vtk_conversion} "
                    f"-ifmt=carp_txt "
                    f"-omsh={output_mesh_vtk_um} -ofmt=vtk")

            os.system(cmd1)
            os.system(cmd2)

        if args.atrium == "LA":
            print(f"INFO: LA path (non-SSM). Labeling and preparing fibers for: {processed_mesh}")
            print(f"INFO: Using LAA: {generator.la_apex} for labeling.")

            try:
                generator.extract_rings(surface_mesh_path=path_for_labeling_obj)  # Uses generator.la_apex
                print(f"INFO: Ring extraction for LA on {path_for_labeling_obj} complete.")
            except Exception as e:
                print(f"ERROR during LA extract_rings: {e}")
                sys.exit(1)

            if args.closed_surface:
                combined_wall_base_LA = f"{meshname_old}_{args.atrium}"  # e.g., path/to/XYZ_LA
                combined_wall_obj_path_LA = _ensure_obj_available(combined_wall_base_LA, ".vtk")

                generator.generate_mesh(input_surface_path=combined_wall_obj_path_LA)
                volumetric_mesh_path_LA_vtk = f"{combined_wall_base_LA}_vol.vtk"  # Output of generate_mesh

                print(f"INFO: Surface ID generation for LA volumetric mesh: {volumetric_mesh_path_LA_vtk}")
                generator.generate_surf_id(
                    volumetric_mesh_path=volumetric_mesh_path_LA_vtk,
                    atrium=args.atrium,
                    resampled=args.resample_input
                )

                fiber_mesh_base = f"{combined_wall_base_LA}_vol"  # e.g., path/to/XYZ_LA_vol
                print(f"INFO: LA volumetric processing complete. 'fiber_mesh_base' is now: {fiber_mesh_base}")

                old_folder_map_base_LA = processed_mesh
                resampled_suffix_for_map_LA = "_res" if args.resample_input else ""

                old_folder_for_map_LA = old_folder_map_base_LA + resampled_suffix_for_map_LA + "_surf"

                new_folder_for_map_LA = fiber_mesh_base + "_surf"  # e.g., path/to/XYZ_LA_vol_surf

                origin_mesh_vtk_path_map_LA = os.path.join(old_folder_for_map_LA, f"{args.atrium}.vtk")  # e.g. LA.vtk
                volumetric_surf_vtk_path_map_LA = os.path.join(new_folder_for_map_LA, f"{args.atrium}.vtk")
                la_main.run(
                    ["--mesh", fiber_mesh_base,
                     "--np", str(n_cpu),
                     "--normals_outside", str(0),
                     "--mesh_type", "vol"
                     "--ofmt", args.ofmt,
                     "--debug", str(args.debug),
                     "--overwrite-behaviour",
                     "append"]
                )

            else:
                la_main.run(
                    ["--mesh", fiber_mesh_base,
                     "--np", str(n_cpu),
                     "--normals_outside", str(args.normals_outside),
                     "--ofmt", args.ofmt,
                     "--debug", str(args.debug),
                     "--overwrite-behaviour",
                     "append"]
                )

                scale_val_la = 1000 * float(args.scale)
                input_mesh_carp_txt_LA = f"{fiber_mesh_base}_fibers/result_{args.atrium}/{args.atrium}_bilayer_with_fiber"
                output_mesh_carp_txt_um_LA = f"{input_mesh_carp_txt_LA}_um"

                cmd1_la = (f"meshtool convert "
                           f"-imsh={input_mesh_carp_txt_LA} -ifmt=carp_txt "
                           f"-omsh={output_mesh_carp_txt_um_LA} -ofmt=carp_txt "
                           f"-scale={scale_val_la}")

                input_mesh_for_vtk_conversion_LA = output_mesh_carp_txt_um_LA
                output_mesh_vtk_um_LA = output_mesh_carp_txt_um_LA
                cmd2_la = (f"meshtool convert "
                           f"-imsh={input_mesh_for_vtk_conversion_LA} -ifmt=carp_txt "
                           f"-omsh={output_mesh_vtk_um_LA} -ofmt=vtk")

                os.system(cmd1_la)
                os.system(cmd2_la)

        elif args.atrium == "RA":
            print(f"INFO: RA path (non-SSM). Current 'processed_mesh' for labeling: {processed_mesh}")
            try:
                if args.closed_surface:
                    # processed_mesh = 'meshname_old_RA_epi'
                    generator.extract_rings_top_epi_endo(surface_mesh_path=processed_mesh)
                else:
                    generator.extract_rings(surface_mesh_path=processed_mesh)
            except Exception as e:
                print(f"ERROR during RA ring extraction: {e}")
                sys.exit(1)

            if args.closed_surface:
                # 'processed_mesh' is currently '{meshname_old}_RA_epi' (the epicardial mesh where rings were just labeled).
                # 'meshname_old' refers to the original input mesh base name (e.g., path/to/XYZ).
                combined_wall_base_RA = f"{meshname_old}_{args.atrium}"
                combined_wall_obj_path_RA = _ensure_obj_available(combined_wall_base_RA, ".vtk")

                print(f"INFO: Volumetric mesh generation for RA from combined wall: {combined_wall_obj_path_RA}")
                generator.generate_mesh(input_surface_path=combined_wall_obj_path_RA)

                # The output of generate_mesh is, e.g., 'meshname_old_RA_vol.vtk'
                volumetric_mesh_path_RA_vtk = f"{combined_wall_base_RA}_vol.vtk"

                print(f"INFO: Surface ID generation for RA volumetric mesh: {volumetric_mesh_path_RA_vtk}")
                generator.generate_surf_id(volumetric_mesh_path=volumetric_mesh_path_RA_vtk,
                                           atrium=args.atrium,
                                           resampled=False)

                fiber_mesh_base = f"{combined_wall_base_RA}_vol"  # e.g. path/to/XYZ_RA_vol

                ra_main.run(
                    ["--mesh",
                     fiber_mesh_base,
                     "--np",
                     str(n_cpu),
                     "--normals_outside",
                     "0",
                     "--mesh_type",
                     "vol",
                     "--ofmt",
                     args.ofmt,
                     "--debug",
                     str(args.debug),
                     "--overwrite-behaviour",
                     "append"]
                )
            else:  # RA, not closed_surface
                ra_main.run(
                    ["--mesh",
                     fiber_mesh_base,
                     "--np",
                     str(n_cpu),
                     "--normals_outside",
                     str(args.normals_outside),
                     "--ofmt",
                     args.ofmt,
                     "--debug",
                     str(args.debug),
                     "--overwrite-behaviour",
                     "append"]
                )

    final_mesh_base_for_plot = ""
    if args.SSM_fitting and not args.closed_surface:
        final_mesh_base_for_plot = current_processed_mesh_base_in_ssm
    elif not args.SSM_fitting:
        final_mesh_base_for_plot = fiber_mesh_base

    if args.debug:
        print(f"INFO: Debug plotting. Using mesh base: {processed_mesh}")

        final_mesh_input_to_fibers = ""
        is_volumetric_plot = "_vol" in os.path.basename(final_mesh_base_for_plot) and args.closed_surface

        if is_volumetric_plot:
            path_to_plot_file = f"{processed_mesh}_fibers/result_{args.atrium}/{args.atrium}_vol_with_fiber.{args.ofmt}"
        else:
            # For LA_RA, the procedural script specifically constructed path using result_RA,
            # even if args.atrium was restored to "LA_RA". We need to respect this.
            # args.atrium at the point of plotting in procedural was its final state.
            if args.atrium == 'LA_RA':
                # In LA_RA case, fibers for RA part were in result_RA, and args.atrium was LA_RA for path construction.
                path_to_plot_file = f"{processed_mesh}_fibers/result_RA/{args.atrium}_bilayer_with_fiber.{args.ofmt}"
            else:
                path_to_plot_file = f"{final_mesh_base_for_plot}_fibers/result_{args.atrium}/{args.atrium}_bilayer_with_fiber.{args.ofmt}"

        try:
            bil = pv.read(path_to_plot_file)

            scalar_viz_key = None
            tag_data_array = None
            if 'elemTag' in bil.point_data:
                tag_data_array = bil.point_data['elemTag']
            elif 'elemTag' in bil.cell_data:
                bil = bil.cell_data_to_point_data(pass_cell_data=True)
                if 'elemTag' in bil.point_data:
                    tag_data_array = bil.point_data['elemTag']


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


            p = pv.Plotter(notebook=False)
            if not args.closed_surface:
                if 'fiber' in bil.point_data:
                    geom = pv.Line()
                    scale_glyph_by = scalar_viz_key if scalar_viz_key and scalar_viz_key in bil.point_data else None

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

    elif args.debug and not processed_mesh:  # If processed_mesh ended up empty/None
        print("WARNING: Debug plotting enabled, but final mesh base for fiber results was not determined. Skipping plot.")
