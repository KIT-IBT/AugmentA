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
import sys
import warnings
from string import Template
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import cKDTree

from Atrial_LDRBM.LDRBM.Fiber_LA import la_main
from Atrial_LDRBM.LDRBM.Fiber_RA import ra_main
from vtk_opencarp_helper_methods.AugmentA_methods.point_selection import pick_point, pick_point_with_preselection
from vtk_opencarp_helper_methods.vtk_methods.filters import apply_vtk_geom_filter
from vtk_opencarp_helper_methods.vtk_methods.mapper import mapp_ids_for_folder
from vtk_opencarp_helper_methods.vtk_methods.normal_orientation import are_normals_outside
from vtk_opencarp_helper_methods.vtk_methods.reader import smart_reader

EXAMPLE_DESCRIPTIVE_NAME = 'AugmentA: Patient-specific Augmented Atrial model Generation Tool'
EXAMPLE_AUTHOR = 'Luca Azzolin <luca.azzolin@kit.edu>'

from standalones.open_orifices_with_curvature import open_orifices_with_curvature
from standalones.open_orifices_manually import open_orifices_manually

from standalones.prealign_meshes import prealign_meshes
from standalones.getmarks import get_landmarks
from standalones.create_SSM_instance import create_SSM_instance
from standalones.resample_surf_mesh import resample_surf_mesh

from Atrial_LDRBM.Generate_Boundaries.atrial_boundary_generator import AtrialBoundaryGenerator


pv.set_plot_theme('dark')
try:
    n_cpu = os.cpu_count()
    if n_cpu and n_cpu > 1 :
        n_cpu = int(n_cpu / 2) if n_cpu % 2 == 0 else int((n_cpu - 1) / 2)
        if n_cpu == 0: n_cpu = 1
    elif n_cpu == 1:
        n_cpu = 1
    else:
        n_cpu = 1
except NotImplementedError:
    n_cpu = 1
print(f"Using {n_cpu} CPUs for parallel tasks (where applicable).")

# -----------------------------------------------------------
# helper: read apex IDs if the *_mesh_data.csv file exists
# -----------------------------------------------------------
def _load_apex_ids(csv_base: str) -> tuple[int | None, int | None]:
    """
    Load LAA and RAA apex IDs from a CSV named '{csv_base}_mesh_data.csv'.

    Returns:
        A tuple (laa_id, raa_id). Each is an int if present and valid,
        otherwise None. Returns (None, None) if the file is missing
        or any error occurs.
    """
    csv_path = f"{csv_base}_mesh_data.csv"

    # If the file doesn't exist, nothing to load
    if not os.path.exists(csv_path):
        return None, None

    try:
        df = pd.read_csv(csv_path)

        # Parse left atrial appendage ID
        laa_id = None
        if "LAA_id" in df.columns:
            raw_la = df["LAA_id"][0]
            if pd.notna(raw_la):
                laa_id = int(raw_la)

        # Parse right atrial appendage ID
        raa_id = None
        if "RAA_id" in df.columns:
            raw_ra = df["RAA_id"][0]
            if pd.notna(raw_ra):
                raa_id = int(raw_ra)

        return laa_id, raa_id

    except Exception:
        return None, None



def get_atria_list(atrium_arg: str) -> list:
    """Converts 'LA', 'RA', 'LA_RA' argument to a list ['LA'], ['RA'], or ['LA', 'RA']."""
    if atrium_arg == 'LA_RA':
        return ['LA', 'RA']
    elif atrium_arg in ['LA', 'RA']:
        return [atrium_arg]
    else:
        raise ValueError(f"Invalid --atrium argument provided: {atrium_arg}")

def AugmentA(args):
    args.SSM_file = os.path.abspath(args.SSM_file)
    args.SSM_basename = os.path.abspath(args.SSM_basename)
    args.mesh = os.path.abspath(args.mesh)

    mesh_dir = os.path.dirname(args.mesh)
    mesh_filename = os.path.basename(args.mesh)
    meshname, mesh_ext = os.path.splitext(mesh_filename)
    meshname = os.path.join(mesh_dir, meshname) # Full base path without extension

    print(f"Processing mesh: {args.mesh}")
    print(f"Base meshname: {meshname}")

    if getattr(args, 'normals_outside', -1) < 0:
        print("Auto-detecting normals...")
        try:
            args.normals_outside = int(are_normals_outside(smart_reader(args.mesh)))
            print(f"Normals outside detected: {args.normals_outside}")
        except Exception as e:
            print(f"Warning: Failed to auto-detect normals ({e}). Defaulting to 0.")
            args.normals_outside = 0

    generator = AtrialBoundaryGenerator(mesh_path=args.mesh,
                                        la_apex=getattr(args, 'LAA', None),
                                        ra_apex=getattr(args, 'RAA', None),
                                        la_base=getattr(args, 'LAA_base', None),
                                        ra_base=getattr(args, 'RAA_base', None),
                                        debug=(getattr(args, 'debug', 0) > 0))

    processed_mesh = meshname # Tracks the base name, starts as initial base
    meshname_old = str(meshname) # Store original base name
    apex_id = None # Store apex ID determined during preprocessing if applicable
    LAA = generator.la_apex if generator.la_apex is not None else ""
    RAA = generator.ra_apex if generator.ra_apex is not None else ""

    if getattr(args, 'closed_surface', False):
        meshname = meshname_old
        processed_mesh = meshname_old + f"_{args.atrium}_epi"
        meshname_epi_base = processed_mesh

        try:
            generator.load_element_tags(csv_filepath=getattr(args, 'tag_csv', 'Atrial_LDRBM/element_tag.csv'))
            generator.separate_epi_endo(tagged_volume_mesh_path=args.mesh, atrium=args.atrium)
        except Exception as e:
            print(f"ERROR during OOP Epi/Endo Separation: {e}")
            sys.exit(1)

    else:
        if getattr(args, 'open_orifices', False):
            if open_orifices_manually is None or open_orifices_with_curvature is None:
                 print("Error: Orifice opening scripts not available.")
                 sys.exit(1)

            try:
                orifice_func = open_orifices_with_curvature if getattr(args, 'use_curvature_to_open', True) else open_orifices_manually
                print(f"Calling standalone: {orifice_func.__name__}")

                if args.atrium == "LA_RA":
                    print("  Performing orifice opening for: LA")
                    la_cut_path, la_determined_apex_id = orifice_func(
                        meshpath=args.mesh,
                        atrium="LA",
                        MRI=getattr(args, 'MRI', 0),
                        scale=getattr(args, 'scale', 1.0),
                        size=getattr(args, 'size', 30),
                        min_cutting_radius=getattr(args, 'min_cutting_radius', 7.5),
                        max_cutting_radius=getattr(args, 'max_cutting_radius', 17.5),
                        LAA=str(getattr(args, 'LAA', '')),
                        RAA=str(getattr(args, 'RAA', '')),
                        debug=getattr(args, 'debug', 0)
                    )

                    if not la_cut_path or not os.path.exists(la_cut_path) or la_determined_apex_id is None or la_determined_apex_id < 0:
                        raise RuntimeError("Orifice opening script failed for LA.")

                    LAA = int(la_determined_apex_id)
                    generator.la_apex = LAA
                    print(f"LA cut mesh: {la_cut_path}, LA Apex ID: {LAA}")

                    ra_cut_path, ra_determined_apex_id = orifice_func(
                        meshpath=args.mesh,
                        atrium="RA",
                        MRI=getattr(args, 'MRI', 0),
                        scale=getattr(args, 'scale', 1.0),
                        size=getattr(args, 'size', 30),
                        min_cutting_radius=getattr(args, 'min_cutting_radius', 7.5),
                        max_cutting_radius=getattr(args, 'max_cutting_radius', 17.5),
                        LAA=str(LAA), RAA=str(getattr(args, 'RAA', '')),
                        debug=getattr(args, 'debug', 0))

                    RAA = int(ra_determined_apex_id)
                    generator.ra_apex = RAA
                    print(f"RA cut mesh: {ra_cut_path}, RA Apex ID: {RAA}")

                    cut_mesh_path = ra_cut_path
                    apex_id = RAA

                else:
                    cut_mesh_path, determined_apex_id = orifice_func(
                        meshpath=args.mesh,
                        atrium=args.atrium,
                        MRI=getattr(args, 'MRI', 0),
                        scale=getattr(args, 'scale', 1.0),
                        size=getattr(args, 'size', 30),
                        min_cutting_radius=getattr(args, 'min_cutting_radius', 7.5),
                        max_cutting_radius=getattr(args, 'max_cutting_radius', 17.5),
                        LAA=str(getattr(args, 'LAA', '')),
                        RAA=str(getattr(args, 'RAA', '')),
                        debug=getattr(args, 'debug', 0)
                    )
                    apex_id = int(determined_apex_id)
                    if args.atrium == "LA":
                        generator.la_apex = apex_id
                        LAA = apex_id

                    elif args.atrium == "RA":
                        generator.ra_apex = apex_id
                        RAA = apex_id

                meshname = os.path.splitext(cut_mesh_path)[0]
                processed_mesh = meshname

            except Exception as e:
                print(f"ERROR during orifice opening call: {e}")
                sys.exit(1)

        else:
            if not getattr(args, 'resample_input', False) and getattr(args, 'find_appendage', False):
                # Handles cases where the input mesh is open, not being resampled here, but the apex needs manual identification.
                print("Workflow: Find Appendage Manually...")
                try:
                    mesh_data = dict()
                    polydata = apply_vtk_geom_filter(smart_reader(args.mesh))

                    if polydata is None or polydata.GetNumberOfPoints() == 0:
                        raise ValueError("Mesh empty")

                    mesh_from_vtk = pv.PolyData(polydata)
                    tree = cKDTree(mesh_from_vtk.points.astype(np.double))

                    LAA = ""
                    RAA = ""

                    if args.atrium == "LA":
                        apex_coord = pick_point(mesh_from_vtk, "LA appendage apex")

                        if apex_coord is None:
                            raise RuntimeError("Apex selection cancelled.")

                        _, apex_idx = tree.query(apex_coord)
                        generator.la_apex = int(apex_idx)
                        LAA = generator.la_apex
                        mesh_data[f"{args.atrium}A_id"] = [LAA]

                    elif args.atrium == "RA":
                        apex_coord = pick_point(mesh_from_vtk, "RA appendage apex")

                        if apex_coord is None:
                            raise RuntimeError("Apex selection cancelled.")

                        _, apex_idx = tree.query(apex_coord)
                        generator.ra_apex = int(apex_idx); RAA = generator.ra_apex
                        mesh_data[f"{args.atrium}A_id"] = [RAA]
                    elif args.atrium == "LA_RA":
                        apex_coord_la = pick_point(mesh_from_vtk, "LA appendage apex")

                        if apex_coord_la is None:
                            raise RuntimeError("Apex selection cancelled.")

                        _, apex_idx_la = tree.query(apex_coord_la)
                        generator.la_apex = int(apex_idx_la); LAA = generator.la_apex
                        mesh_data["LAA_id"] = [LAA]
                        apex_coord_ra = pick_point_with_preselection(mesh_from_vtk, "RA appendage apex", apex_coord_la)

                        if apex_coord_ra is None:
                            raise RuntimeError("Apex selection cancelled.")
                        _, apex_idx_ra = tree.query(apex_coord_ra)

                        generator.ra_apex = int(apex_idx_ra)
                        RAA = generator.ra_apex
                        mesh_data["RAA_id"] = [RAA]

                    fname = f'{meshname}_mesh_data.csv'
                    df = pd.DataFrame(mesh_data)
                    df.to_csv(fname, float_format="%.2f", index=False)
                    processed_mesh = meshname

                except Exception as e:
                    print(f"ERROR during manual apex selection: {e}")
                    sys.exit(1)
            else:
                print("Workflow: Using p rovided Apex IDs (or defaults).")
                processed_mesh = meshname

                LAA = generator.la_apex if generator.la_apex is not None else ""
                RAA = generator.ra_apex if generator.ra_apex is not None else ""


    # --- SSM Fitting Logic ---
    if getattr(args, 'SSM_fitting', False) and not getattr(args, 'closed_surface', False):
        print("\n--- Running SSM Fitting Workflow ---")

        try:
            ssm_abs_basename = os.path.abspath(args.SSM_basename)
            ssm_abs_file = os.path.abspath(args.SSM_file)
            ssm_surf_dir = ssm_abs_basename + '_surf'
            os.makedirs(ssm_surf_dir, exist_ok=True)

            # Generate SSM landmarks if not present
            ssm_landmarks_path = os.path.join(ssm_surf_dir, 'landmarks.json')
            if not os.path.isfile(ssm_landmarks_path):
                print(f"  Generating landmarks for SSM base shape: {ssm_abs_basename}")
                ssm_laa_id = 6329
                ssm_raa_id = 21685

                ssm_base_mesh_path = ssm_abs_basename + ".vtk"
                if not os.path.exists(ssm_base_mesh_path):
                    ssm_base_mesh_path = ssm_abs_basename + ".obj"

                if not os.path.exists(ssm_base_mesh_path):
                    raise FileNotFoundError(f"SSM base mesh not found: {ssm_abs_basename}.vtk/obj")

                print(f"Running ring detection on SSM base mesh...")
                ssm_ring_generator = AtrialBoundaryGenerator(mesh_path=ssm_base_mesh_path, la_apex=ssm_laa_id, ra_apex=ssm_raa_id, debug=(args.debug > 0))

                ssm_ring_generator.extract_rings(surface_mesh_path=ssm_base_mesh_path)
                print(f"Ring detection complete for SSM base.")

                get_landmarks(ssm_abs_basename, 0, 1)
                print(f"Landmarks generated for SSM base.")
            else:
                print(f"Using existing landmarks for SSM base shape: {ssm_landmarks_path}")


            target_mesh_base_for_ssm = processed_mesh
            target_surf_dir = target_mesh_base_for_ssm + '_surf'
            os.makedirs(target_surf_dir, exist_ok=True)
            target_landmarks_path = os.path.join(target_surf_dir, 'landmarks.json')

            # Generate landmarks for target mesh if needed
            if not os.path.isfile(target_landmarks_path):
                 print(f"Generating landmarks for target mesh: {target_mesh_base_for_ssm}")

                 target_surface_path = processed_mesh + ".vtk" # Assume vtk first

                 if not os.path.exists(target_surface_path):
                     target_surface_path = processed_mesh + ".obj"

                 if not os.path.exists(target_surface_path):
                     target_surface_path = processed_mesh + ".ply"

                 if not os.path.exists(target_surface_path):
                     raise FileNotFoundError(f"Cannot find target surface mesh: {target_mesh_base_for_ssm}.vtk/obj/ply")

                 print(f"Running ring detection on target mesh: {target_surface_path}")

                 # Use the main generator instance (update apex IDs from local vars)
                 generator.la_apex = LAA if LAA else None
                 generator.ra_apex = RAA if RAA else None

                 generator.extract_rings(surface_mesh_path=target_surface_path)
                 print(f"Ring detection complete for target.")

                 get_landmarks(target_mesh_base_for_ssm, 1, 1)
                 print(f"Landmarks generated for target.")
            else:
                 print(f"Using existing landmarks for target mesh: {target_landmarks_path}")

            print(f"Prealigning target: {target_mesh_base_for_ssm}...")
            prealign_meshes(target_mesh_base_for_ssm, ssm_abs_basename, args.atrium, 0)

            print(f"Setting up registration file...")
            reg_template_path = 'template/Registration_ICP_GP_template.txt'
            if not os.path.exists(reg_template_path):
                raise FileNotFoundError(f"Reg template not found: {reg_template_path}")

            with open(reg_template_path, 'r') as f:
                lines = ''.join(f.readlines())

            temp_obj = Template(lines)
            SSM_fit_file = temp_obj.substitute(SSM_file=ssm_abs_file,
                                               SSM_dir=os.path.abspath(ssm_surf_dir),
                                               target_dir=os.path.abspath(target_surf_dir))

            reg_output_path = os.path.join(target_surf_dir, 'Registration_ICP_GP.txt')

            with open(reg_output_path, 'w') as f:
                f.write(SSM_fit_file)

            print(f"Registration file written: {reg_output_path}. Run registration externally.")

            # Create SSM instance
            coeffs_path = os.path.join(target_surf_dir, 'coefficients.txt')

            if not os.path.isfile(coeffs_path):
                raise ValueError(f"SSM Coefficients file not found: {coeffs_path}. Run registration first.")

            ssm_instance_base = os.path.join(target_surf_dir, f"{args.atrium}_fit")
            ssm_instance_obj = ssm_instance_base + ".obj"
            print(f"Creating SSM instance: {ssm_instance_obj}...")
            create_SSM_instance(ssm_abs_file, coeffs_path, ssm_instance_obj)

            # Update processed_mesh base name (Kept from original)
            processed_mesh = ssm_instance_base
            print(f"SSM instance created. Updated processed mesh base: {processed_mesh}")

            # Optional Resampling after fitting (Kept from original)
            if getattr(args, 'resample_input', False):
                print(f"Resampling SSM instance...")
                apex_id_for_resample = apex_id if apex_id is not None else -1 # Use ID if available from orifice opening
                print(f"Using apex_id={apex_id_for_resample} for resampling.")

                resample_surf_mesh(
                    processed_mesh,
                    target_mesh_resolution=args.target_mesh_resolution,
                    find_apex_with_curv=1,
                    scale=args.scale,
                    apex_id=apex_id_for_resample, # Pass determined/fallback apex_id
                    atrium=args.atrium
                )
                processed_mesh = processed_mesh + '_res' # Update base name

                laa_csv, raa_csv = _load_apex_ids(processed_mesh)
                if laa_csv is not None:
                    generator.la_apex = laa_csv
                if raa_csv is not None:
                    generator.ra_apex = raa_csv

                print(f"Resampling complete. Updated processed mesh base: {processed_mesh}")


            # --- Ring detection ON FITTED/RESAMPLED MESH ---
            print(f"  Running OOP Ring Detection on final SSM mesh: {processed_mesh}")

            # Read final apex IDs from CSV generated by fitting/resampling
            final_apex_csv_path = f"{processed_mesh}_mesh_data.csv" # Assumes resample_surf_mesh creates this
            final_laa_id = None
            final_raa_id = None
            if os.path.exists(final_apex_csv_path):
                try:
                    df_final_apex = pd.read_csv(final_apex_csv_path)

                    # Use LAA_id/RAA_id keys consistently
                    if 'LAA_id' in df_final_apex.columns and pd.notna(df_final_apex['LAA_id'][0]):
                        final_laa_id = int(df_final_apex['LAA_id'][0])

                    if 'RAA_id' in df_final_apex.columns and pd.notna(df_final_apex['RAA_id'][0]):
                        final_raa_id = int(df_final_apex['RAA_id'][0])

                    print(f"Read final apex IDs: LAA={final_laa_id}, RAA={final_raa_id}")

                except Exception as read_err:
                    print(f"Warning: Could not read final apex IDs from {final_apex_csv_path}: {read_err}")
            else:
                print(f"Warning: Final apex ID CSV not found: {final_apex_csv_path}. Using IDs from before resampling if available.")

                # Fallback to IDs held by generator before this step
                final_laa_id = generator.la_apex
                final_raa_id = generator.ra_apex

            # Update generator state just before the call
            generator.la_apex = final_laa_id
            generator.ra_apex = final_raa_id

            # Determine final surface path (OBJ preferred by original label_atrial_orifices)
            final_surface_path_obj = processed_mesh + ".obj"
            final_surface_path_ply = processed_mesh + ".ply"
            final_surface_path_vtk = processed_mesh + ".vtk"

            if not os.path.exists(final_surface_path_obj):
                 if os.path.exists(final_surface_path_ply):
                     print(f"Converting {final_surface_path_ply} to OBJ for ring detection...")
                     pv.save_meshio(final_surface_path_obj, pv.read(final_surface_path_ply), "obj")
                 elif os.path.exists(final_surface_path_vtk):
                     print(f"Converting {final_surface_path_vtk} to OBJ for ring detection...")
                     pv.save_meshio(final_surface_path_obj, pv.read(final_surface_path_vtk), "obj")
                 else:
                     raise FileNotFoundError(f"Cannot find final fitted/resampled mesh: {processed_mesh}.obj/vtk/ply")

            generator.extract_rings(surface_mesh_path=final_surface_path_obj)

            print(f"OOP Ring detection complete for fitted mesh.")

        except Exception as e:
            print(f"ERROR during SSM fitting workflow: {e}")
            sys.exit(1)
    # --- END SSM Fitting ---

    elif not getattr(args, 'SSM_fitting', False):
        # --- Workflow: Not SSM Fitting ---
        print("\n--- Running Non-SSM Workflow ---")

        if getattr(args, 'resample_input', False) and getattr(args, 'find_appendage', False):
            print("Workflow Action: Resampling mesh...")
            if resample_surf_mesh is None:
                 print("Resample script not available, skipping.")

            else:
                try:
                    resample_input_base = meshname # Use original base name
                    resample_input_path_vtk = f'{resample_input_base}.vtk'
                    resample_input_path_obj = f'{resample_input_base}.obj'

                    if not os.path.exists(resample_input_path_obj):
                        if not os.path.exists(resample_input_path_vtk):
                             raise FileNotFoundError(f"Cannot find {resample_input_path_vtk} or {resample_input_path_obj} for resampling.")

                        meshin = pv.read(resample_input_path_vtk)
                        pv.save_meshio(resample_input_path_obj, meshin, "obj")
                        # todo: just overwrite regardless

                    # Determine apex ID for resampling (use the one found manually earlier)
                    apex_id_for_resample = -1

                    print(f"Calling resample_surf_mesh on base: {resample_input_base}, Apex ID: {apex_id_for_resample}")
                    resample_surf_mesh(
                        resample_input_base,
                        target_mesh_resolution=args.target_mesh_resolution,
                        find_apex_with_curv=0,
                        scale=args.scale,
                        apex_id=apex_id_for_resample,
                        atrium=args.atrium
                    )
                    processed_mesh = f'{resample_input_base}_res'

                    laa_csv, raa_csv = _load_apex_ids(processed_mesh)
                    if laa_csv is not None:
                        generator.la_apex = laa_csv
                    if raa_csv is not None:
                        generator.ra_apex = raa_csv

                    print(f"Resampling complete. Updated processed mesh base: {processed_mesh}")

                    # Convert PLY output to OBJ (matching original behavior which used OBJ later)
                    resampled_ply_path = processed_mesh + ".ply"
                    resampled_obj_path = processed_mesh + ".obj"

                    if os.path.exists(resampled_ply_path) and not os.path.exists(resampled_obj_path):
                        meshin = pv.read(resampled_ply_path)
                        pv.save_meshio(resampled_obj_path, meshin, "obj")

                    elif not os.path.exists(resampled_ply_path):
                        print(f"Warning: Resampled PLY file not found: {resampled_ply_path}")

                except Exception as e:
                    print(f"Error during resampling: {e}")
                    sys.exit(1)
        else:
            print("Skipping resampling for non-SSM case based on args.")


        # --- Ring Detection (Replaces old label_atrial_orifices) ---
        print(f"Running OOP Ring Detection on: {processed_mesh}")

        # Read final apex IDs from the CSV file created earlier
        final_apex_csv_path = f"{processed_mesh}_mesh_data.csv"
        final_laa_id = generator.la_apex
        final_raa_id = generator.ra_apex

        print(f"Attempting to read apex IDs from: {final_apex_csv_path}")

        if os.path.exists(final_apex_csv_path):
            try:
                df_final_apex = pd.read_csv(final_apex_csv_path)
                if args.atrium == "LA":
                    key = f"{args.atrium}A_id"
                    if key in df_final_apex.columns and pd.notna(df_final_apex[key][0]): final_laa_id = int(df_final_apex[key][0])
                elif args.atrium == "RA":
                    key = f"{args.atrium}A_id"
                    if key in df_final_apex.columns and pd.notna(df_final_apex[key][0]): final_raa_id = int(df_final_apex[key][0])
                elif args.atrium == "LA_RA":
                    if "LAA_id" in df_final_apex.columns and pd.notna(df_final_apex["LAA_id"][0]):
                        final_laa_id = int(df_final_apex["LAA_id"][0])

                    if "RAA_id" in df_final_apex.columns and pd.notna(df_final_apex["RAA_id"][0]):
                        final_raa_id = int(df_final_apex["RAA_id"][0])

                print(f"Read apex IDs from CSV: LAA={final_laa_id}, RAA={final_raa_id}")

            except Exception as read_err:
                print(f"Warning: Could not read apex IDs from {final_apex_csv_path}: {read_err}")
        else:
             # If CSV doesn't exist, rely on generator state
             print(f"  Apex ID CSV not found ({final_apex_csv_path}). Using IDs from generator state: LAA={final_laa_id}, RAA={final_raa_id}")

        # Update generator state just before calling extract_rings
        generator.la_apex = final_laa_id
        generator.ra_apex = final_raa_id

        # Determine final surface path
        final_surface_path = processed_mesh + ".obj"

        if not os.path.exists(final_surface_path):
             # Convert VTK/PLY if OBJ not found
             vtk_path = processed_mesh + ".vtk"
             ply_path = processed_mesh + ".ply"

             if os.path.exists(vtk_path):
                 print(f"  Converting {vtk_path} to OBJ for ring detection...")
                 pv.save_meshio(final_surface_path, pv.read(vtk_path), "obj")

             elif os.path.exists(ply_path):
                 # Keep PLY->VTK->OBJ conversion for robustness
                 temp_vtk_path = processed_mesh + "_temp.vtk"

                 print(f"  Converting {ply_path} to VTK ({temp_vtk_path})...")

                 ply_mesh = pv.read(ply_path)
                 ply_mesh.save(temp_vtk_path, binary=True)

                 print(f"  Converting {temp_vtk_path} to OBJ ({final_surface_path})...")
                 pv.save_meshio(final_surface_path, pv.read(temp_vtk_path), "obj")

                 if os.path.exists(temp_vtk_path):
                     os.remove(temp_vtk_path) # Clean up temp VTK
             else:
                 raise FileNotFoundError(f"Cannot find final surface mesh: {processed_mesh}.obj/vtk/ply")

        try:
            if getattr(args, 'closed_surface', False) and args.atrium == "RA":
                 print("Using 'top_epi_endo' ring workflow (Closed Surface RA, Non-SSM path)...")

                 expected_endo_path = f"{meshname_old}_RA_endo.obj" # Specific to RA
                 print(f"Expecting endo mesh at: {expected_endo_path}")

                 if not os.path.exists(expected_endo_path):
                      print(f"  Warning: Required endo mesh '{expected_endo_path}' not found.")

                 generator.extract_rings_top_epi_endo(surface_mesh_path=final_surface_path)
            else:
                # Standard case for open surfaces OR LA closed surface
                generator.extract_rings(surface_mesh_path = final_surface_path)

            print(f"  OOP Ring detection complete for non-SSM mesh.")
        except Exception as e:
            print(f"ERROR during OOP ring detection for non-SSM mesh: {e}")
            sys.exit(1)

    else:
        mesh_type_for_ldrbm = "surf" # Default
        mesh_for_ldrbm_base = processed_mesh # Use the base name determined by previous steps

        # Handle the closed_surface case where volume mesh is generated and used
        if getattr(args, 'closed_surface', False):
            mesh_type_for_ldrbm = "vol"
            print(f"  LDRBM on Volume Mesh (Closed Surface Workflow)...")
            # --- Generate Volume Mesh (OOP Replacement) ---
            print(f"    Generating volume mesh for {args.atrium}...")
            # Input surface is the separated epi mesh base name from earlier
            # Original logic used meshname_old + f'_{args.atrium}' as base for generate_mesh
            input_surf_base_for_vol = meshname_old + f'_{args.atrium}'
            input_surf_path_for_vol = input_surf_base_for_vol + ".obj" # Assume obj needed

            # CHANGE 6: Guard conversion with existence check
            if not os.path.exists(input_surf_path_for_vol):
                 input_surf_path_vtk = input_surf_base_for_vol + "_epi.vtk" # Epi surface vtk
                 if not os.path.exists(input_surf_path_vtk):
                     raise FileNotFoundError(f"Cannot find separated epi surface {input_surf_base_for_vol}_epi.obj/vtk for volume generation.")
                 print(f"      Converting {input_surf_path_vtk} to {input_surf_path_for_vol}...")
                 meshin = pv.read(input_surf_path_vtk)
                 pv.save_meshio(input_surf_path_for_vol, meshin, "obj")


            generator.generate_mesh(input_surface_path=input_surf_path_for_vol)

            volumetric_mesh_path = input_surf_base_for_vol + "_vol.vtk"
            if not os.path.exists(volumetric_mesh_path):
                 raise RuntimeError(f"Volume mesh generation failed: {volumetric_mesh_path} not found.")
            print(f"    Volume mesh generated: {volumetric_mesh_path}")

            # --- Generate Surface IDs (OOP Replacement) ---
            print(f"    Generating surface IDs for {args.atrium}...")
            # REPLACEMENT for old generate_surf_id(meshname_old, args.atrium, ...)
            # Original passed meshname_old as the base name argument
            generator.generate_surf_id(
                volumetric_mesh_path=volumetric_mesh_path, # Path to the volume mesh just created
                atrium=args.atrium,
                # CHANGE 2-3: Pass resample flag. Internal logic of generate_surf_id needs to handle old_folder construction.
                resampled=getattr(args, 'resample_input', False)
            )
            print(f"    Surface ID generation complete.")

            # Update base name for LDRBM call to the volume mesh base (as per original)
            mesh_for_ldrbm_base = meshname_old + f"_{args.atrium}_vol"


            if args.atrium == "LA":
                 print(f"    Mapping IDs from surface to volume...")
                 resampled_suffix = "_res" if getattr(args, 'resample_input', False) else ""
                 # CHANGE 2-3: old_folder construction matches original logic's use of 'meshname' variable state
                 old_folder = meshname + resampled_suffix + "_surf" # meshname is meshname_old + '_LA_epi' here
                 new_folder = mesh_for_ldrbm_base + "_surf"
                 origin_mesh_path = meshname + ".vtk" # Path to *_epi.vtk
                 volumetric_mesh_read_path = os.path.join(new_folder, f"{args.atrium}.vtk")

                 if os.path.exists(old_folder) and os.path.exists(origin_mesh_path) and os.path.exists(volumetric_mesh_read_path):
                     try:
                         origin_mesh_vtk = smart_reader(origin_mesh_path)
                         volumetric_mesh_vtk = smart_reader(volumetric_mesh_read_path)
                         mapp_ids_for_folder(old_folder, new_folder, origin_mesh_vtk, volumetric_mesh_vtk)
                         print(f"    ID Mapping complete.")
                     except Exception as map_err:
                         print(f"    Warning: Failed to map IDs: {map_err}")
                 else:
                     print(f"    Warning: Skipping ID mapping due to missing files/folders.")
                     print(f"      Need: {old_folder}, {origin_mesh_path}, {volumetric_mesh_read_path}")

        # --- LDRBM Call Setup (Common) ---
        print(f"  Input mesh base for LDRBM: {mesh_for_ldrbm_base}")
        print(f"  Mesh type for LDRBM: {mesh_type_for_ldrbm}")

        # Auto-detect normals (using appropriate mesh path)
        ldrbm_input_path = f"{mesh_for_ldrbm_base}.vtk" # Assume VTK is primary format
        if not os.path.exists(ldrbm_input_path): ldrbm_input_path = f"{mesh_for_ldrbm_base}.vtu"
        if not os.path.exists(ldrbm_input_path): ldrbm_input_path = f"{mesh_for_ldrbm_base}.obj"
        if not os.path.exists(ldrbm_input_path): ldrbm_input_path = f"{mesh_for_ldrbm_base}.ply"

        current_normals_outside = getattr(args, 'normals_outside', -1) # Use value potentially set earlier
        if current_normals_outside == -1: # Only detect if not already set
             print(f"  Auto-detecting normals on: {ldrbm_input_path}")
             if not os.path.exists(ldrbm_input_path):
                  print(f"Warning: Mesh file for normal detection not found: {ldrbm_input_path}. Defaulting normals_outside=0.")
                  current_normals_outside = 0
             else:
                 try:
                     current_normals_outside = int(are_normals_outside(smart_reader(ldrbm_input_path)))
                     print(f"  Auto-detected normals_outside: {current_normals_outside}")
                 except Exception as e:
                     print(f"  Warning: Failed to auto-detect normals ({e}). Defaulting to 0.")
                     current_normals_outside = 0

        # Construct LDRBM arguments
        ldrbm_common_args = ["--mesh", mesh_for_ldrbm_base,
                             "--np", str(n_cpu),
                             "--normals_outside", str(current_normals_outside),
                             "--ofmt", args.ofmt,
                             "--debug", str(getattr(args, 'debug', 0)),
                             "--overwrite-behaviour", "append"]
        if mesh_type_for_ldrbm == "vol":
            ldrbm_common_args.extend(["--mesh_type", "vol"])

        # --- Execute LDRBM ---
        try:
            original_atrium_arg = args.atrium # Store original arg for LA_RA case
            if args.atrium == 'LA_RA':
                # LA First
                print("  Running LA LDRBM (LA_RA case)...")
                args.atrium = "LA" # LDRBM scripts might use this
                if la_main: la_main.run(ldrbm_common_args[:])
                else: print("   Error: la_main not imported.")
                print("  LA LDRBM finished.")
                # Then RA
                print("  Running RA LDRBM (LA_RA case)...")
                args.atrium = "RA"
                if ra_main: ra_main.run(ldrbm_common_args[:])
                else: print("   Error: ra_main not imported.")
                print("  RA LDRBM finished.")
                args.atrium = original_atrium_arg # Restore

                # Meshtool conversions (kept identical to original)
                print("  Running meshtool conversions for LA_RA case...")
                scale_val = 1000 * getattr(args, 'scale', 1.0)
                ra_fiber_base = f'{mesh_for_ldrbm_base}_fibers/result_RA/RA_bilayer_with_fiber'
                ra_um_base = f'{ra_fiber_base}_um'
                input_fiber_file = f"{ra_fiber_base}.carp_txt"
                if os.path.exists(input_fiber_file):
                     cmd1 = f"meshtool convert -imsh={ra_fiber_base} -ifmt=carp_txt -omsh={ra_um_base} -ofmt=carp_txt -scale={scale_val}"
                     cmd2 = f"meshtool convert -imsh={ra_um_base} -ifmt=carp_txt -omsh={ra_um_base} -ofmt=vtk"
                     print(f"    Executing: {cmd1}"); os.system(cmd1)
                     print(f"    Executing: {cmd2}"); os.system(cmd2)
                else: print(f"    Warning: Input for RA meshtool conversion not found: {input_fiber_file}")

            elif args.atrium == "LA":
                print("  Running LA LDRBM...")
                if la_main: la_main.run(ldrbm_common_args[:])
                else: print("   Error: la_main not imported.")
                print("  LA LDRBM finished.")
                # Meshtool conversions for LA surface case (kept identical)
                if mesh_type_for_ldrbm != "vol":
                     print("  Running meshtool conversions for LA surface case...")
                     scale_val = 1000 * getattr(args, 'scale', 1.0)
                     la_fiber_base = f'{mesh_for_ldrbm_base}_fibers/result_LA/LA_bilayer_with_fiber'
                     la_um_base = f'{la_fiber_base}_um'
                     input_fiber_file = f"{la_fiber_base}.carp_txt"
                     if os.path.exists(input_fiber_file):
                         cmd1 = f"meshtool convert -imsh={la_fiber_base} -ifmt=carp_txt -omsh={la_um_base} -ofmt=carp_txt -scale={scale_val}"
                         cmd2 = f"meshtool convert -imsh={la_um_base} -ifmt=carp_txt -omsh={la_um_base} -ofmt=vtk"
                         print(f"    Executing: {cmd1}"); os.system(cmd1)
                         print(f"    Executing: {cmd2}"); os.system(cmd2)
                     else: print(f"    Warning: Input for LA meshtool conversion not found: {input_fiber_file}")

            elif args.atrium == "RA":
                print("  Running RA LDRBM...")
                if ra_main: ra_main.run(ldrbm_common_args[:])
                else: print("   Error: ra_main not imported.")
                print("  RA LDRBM finished.")
                # No specific RA conversions in original non-closed surface case

        except Exception as e:
            print(f"Error during LDRBM execution: {e}")
            sys.exit(1)


    # --- Final Debug Plotting (Identical to Original) ---
    if getattr(args, 'debug', 0):
        print("\n--- Generating Debug Plot ---")
        try:
            final_mesh_for_plot = None
            # CHANGE 6: Use corrected plot_base logic
            if getattr(args, "closed_surface", False):
                plot_base = mesh_for_ldrbm_base  # This is the *_vol base name
            else:
                plot_base = processed_mesh # This is the final surface base name

            # Construct path based on plot_base and original logic
            if getattr(args, 'closed_surface', False):
                 plot_path = f'{plot_base}_fibers/result_{args.atrium}/{args.atrium}_vol_with_fiber.{args.ofmt}'
            else: # Open surface
                plot_atrium = 'RA' if args.atrium == 'LA_RA' else args.atrium
                plot_path = f'{plot_base}_fibers/result_{plot_atrium}/{plot_atrium}_bilayer_with_fiber.{args.ofmt}'

            if os.path.exists(plot_path):
                final_mesh_for_plot = plot_path
                print(f"  Reading final mesh for plotting: {final_mesh_for_plot}")
                bil = pv.read(final_mesh_for_plot)
                geom = pv.Line()
                # Tag manipulation (identical to original)
                mask = bil['elemTag'] > 99; bil['elemTag'][mask] = 0
                mask = bil['elemTag'] > 80; bil['elemTag'][mask] = 20
                mask = bil['elemTag'] > 10; bil['elemTag'][mask] = bil['elemTag'][mask] - 10
                mask = bil['elemTag'] > 50; bil['elemTag'][mask] = bil['elemTag'][mask] - 50

                p = pv.Plotter(notebook=False)
                if not getattr(args, 'closed_surface', False):
                    # Add fibers glyph only for surface meshes
                    fibers = bil.glyph(orient="fiber", factor=0.5, geom=geom, scale="elemTag")
                    p.add_mesh(fibers, show_scalar_bar=False, cmap='tab20', line_width=10, render_lines_as_tubes=True)
                p.add_mesh(bil, scalars="elemTag", show_scalar_bar=False, cmap='tab20')
                print("  Displaying plot window...")
                p.show()
                p.close()
            else:
                print(f"  Warning: Could not find final LDRBM output mesh for debug plotting: {plot_path}")
        except Exception as plot_err:
            print(f"  Warning: Failed to generate debug plot: {plot_err}")


    print("\n--- AugmentA Pipeline Finished ---")