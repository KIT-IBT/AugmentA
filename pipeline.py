#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised AugmentA Pipeline using Refactored AtrialBoundaryGenerator.

This version aims to replicate the original procedural workflow EXACTLY,
without added control flags. Steps like LDRBM and Surface ID generation
will run unconditionally if possible within the determined workflow.
Ring workflow choice is inferred from args.closed_surface.

Requires corrected 'epi_endo_separator.py' and MODIFIED standalone
orifice opening scripts (must RETURN path and apex_id).

FIX 1: Corrected access to args.LAA/args.RAA when calling standalone
       orifice opening scripts using getattr.
FIX 2: Added conversion from PLY to VTK after resampling to avoid
       'ReadAllScalarsOn' error during ring detection.
"""

import argparse
import os
import sys
import shutil
import warnings
from string import Template
from typing import Union, Tuple, Any

import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import cKDTree

from Atrial_LDRBM.Generate_Boundaries.atrial_boundary_generator import AtrialBoundaryGenerator

from standalones.open_orifices_with_curvature import open_orifices_with_curvature
from standalones.open_orifices_manually import open_orifices_manually
STANDALONE_ORIFICE_OPENING_AVAILABLE = True

from vtk_opencarp_helper_methods.vtk_methods.reader import smart_reader
from vtk_opencarp_helper_methods.vtk_methods.filters import apply_vtk_geom_filter
from vtk_opencarp_helper_methods.vtk_methods.normal_orientation import are_normals_outside
from vtk_opencarp_helper_methods.vtk_methods.finder import find_closest_point
from vtk_opencarp_helper_methods.AugmentA_methods.point_selection import pick_point

try:
    from Atrial_LDRBM.LDRBM.Fiber_LA import la_main
    from Atrial_LDRBM.LDRBM.Fiber_RA import ra_main
    LDRBM_AVAILABLE = True
except ImportError:
    warnings.warn("Could not import LDRBM modules (Fiber_LA/Fiber_RA). Fiber generation step may fail.")
    la_main = None
    ra_main = None
    LDRBM_AVAILABLE = False # Track availability

try:
    from standalones.prealign_meshes import prealign_meshes
    from standalones.getmarks import get_landmarks
    from standalones.create_SSM_instance import create_SSM_instance
    from standalones.resample_surf_mesh import resample_surf_mesh
    SSM_TOOLS_AVAILABLE = True
except ImportError:
    warnings.warn("Could not import standalone SSM fitting scripts. --SSM_fitting option may not work fully.")
    prealign_meshes = None
    get_landmarks = None
    create_SSM_instance = None
    resample_surf_mesh = None
    SSM_TOOLS_AVAILABLE = False # Track availability

# --- Setup ---
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

# NOTE: Argument parsing happens SOLELY in main.py
# The AugmentA function receives the already parsed 'args' object.

def get_atria_list(atrium_arg: str) -> list:
    """Converts 'LA', 'RA', 'LA_RA' argument to a list ['LA'], ['RA'], or ['LA', 'RA']."""
    if atrium_arg == 'LA_RA':
        return ['LA', 'RA']
    elif atrium_arg in ['LA', 'RA']:
        return [atrium_arg]
    else:
        raise ValueError(f"Invalid --atrium argument provided: {atrium_arg}")

def AugmentA(args):
    """Main AugmentA workflow function using the refactored OOP approach."""

    print("--- Starting AugmentA Pipeline (OOP, Replicating Procedural Flow) ---")
    print(f"Run Args: {vars(args)}") # Print received args

    # --- Input Validation & Setup ---
    # Basic checks based on originally defined arguments
    if not hasattr(args, 'mesh') or not args.mesh or not os.path.exists(args.mesh):
        print(f"Error: Input mesh file not found or not specified: {getattr(args, 'mesh', 'Not Specified')}")
        sys.exit(1)
    # Use getattr for optional args to avoid AttributeError if not defined in main.py parser
    if getattr(args, 'closed_surface', False) and (not hasattr(args, 'tag_csv') or not getattr(args, 'tag_csv') or not os.path.exists(getattr(args, 'tag_csv', ''))):
        print(f"Error: Tag CSV file not found (--closed_surface requires --tag_csv): {getattr(args, 'tag_csv', 'Not Specified')}")
        sys.exit(1)

    # Determine atria to process using helper
    try:
        atria_to_process = get_atria_list(args.atrium)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Initial Apex ID Check - rely only on args passed from main.py's parser
    # These checks ensure that if we don't run orifice opening or manual finding, the IDs are provided
    # Use getattr to safely access LAA/RAA, defaulting to None if not present
    if not getattr(args, 'open_orifices', False) and not getattr(args, 'find_appendage', False):
        if 'LA' in atria_to_process and getattr(args, 'LAA', None) is None:
             print("Error: --LAA must be provided if processing LA without --open_orifices or --find_appendage.")
             sys.exit(1)
        if 'RA' in atria_to_process and getattr(args, 'RAA', None) is None:
             print("Error: --RAA must be provided if processing RA without --open_orifices or --find_appendage.")
             sys.exit(1)


    # --- Path and Variable Setup ---
    mesh_abs_path = os.path.abspath(args.mesh)
    mesh_dir = os.path.dirname(mesh_abs_path)
    mesh_filename = os.path.basename(mesh_abs_path)
    mesh_base_name_orig, mesh_ext = os.path.splitext(mesh_filename)
    mesh_base_path_orig = os.path.join(mesh_dir, mesh_base_name_orig)

    processed_mesh_base = mesh_base_path_orig
    surface_mesh_path_for_rings = mesh_abs_path
    volumetric_mesh_path = None # Will be set if input is closed or volume is generated

    # Use original args directly for apex IDs, may be updated
    # Use getattr for safe access, defaulting to None
    current_LAA = getattr(args, 'LAA', None)
    current_RAA = getattr(args, 'RAA', None)
    current_LAA_base = getattr(args, 'LAA_base', None)
    current_RAA_base = getattr(args, 'RAA_base', None)

    # Determine if volume generation might be needed later
    needs_volume_mesh_generation = False # Flag to check later

    # --- Instantiate Generator ---
    print("\nInstantiating AtrialBoundaryGenerator...")
    generator = AtrialBoundaryGenerator(
        mesh_path=mesh_abs_path,
        la_apex=current_LAA,
        ra_apex=current_RAA,
        la_base=current_LAA_base,
        ra_base=current_RAA_base,
        debug=(getattr(args, 'debug', 0) > 0) # Use getattr for safety
    )
    print("Generator instantiated.")

    # --- Step 1: Input Preprocessing ---
    print("\n--- Step 1: Input Preprocessing ---")
    if getattr(args, 'closed_surface', False):
        # --- Workflow: Closed Surface -> Separate Epi/Endo ---
        print("Workflow Action: Running Epi/Endo Separation.")
        volumetric_mesh_path = mesh_abs_path # Input IS the volume mesh
        # Derive base name expected after separation (strip _vol)
        if mesh_base_name_orig.endswith('_vol'):
            processed_mesh_base = os.path.join(mesh_dir, mesh_base_name_orig[:-4])
        else:
             processed_mesh_base = mesh_base_path_orig
             print(f"Warning: --closed_surface specified, but input file '{mesh_filename}' "
                   f"does not end with '_vol'. Using '{processed_mesh_base}' as base for outputs.")

        tag_csv_path = getattr(args, 'tag_csv', 'Atrial_LDRBM/element_tag.csv') # Use default if not present
        if not os.path.exists(tag_csv_path):
             print(f"Error: Tag CSV file not found: {tag_csv_path}")
             sys.exit(1)
        try:
            print("  Loading Element Tags...")
            generator.load_element_tags(tag_csv_path)
            print(f"  Tags loaded from {tag_csv_path}")

            for atrium_tag in atria_to_process:
                print(f"  Separating {atrium_tag}...")
                generator.separate_epi_endo(tagged_volume_mesh_path=mesh_abs_path, atrium=atrium_tag)
                print(f"  {atrium_tag} separation complete.")
            print("Epi/Endo separation finished.")

            # Determine the surface for subsequent ring detection (use epi)
            atrium_for_surf = 'LA' if 'LA' in atria_to_process else 'RA'
            potential_surf_path = f"{processed_mesh_base}_{atrium_for_surf}_epi.vtk"
            if not os.path.exists(potential_surf_path):
                 potential_surf_path = f"{processed_mesh_base}_{atrium_for_surf}_epi.obj"
            if not os.path.exists(potential_surf_path):
                 print(f"Error: Separated epicardial surface not found after separation step: {potential_surf_path}")
                 sys.exit(1)
            surface_mesh_path_for_rings = potential_surf_path
            print(f"  Surface for subsequent ring detection set to: {surface_mesh_path_for_rings}")

        except Exception as e:
            print(f"Error during closed surface preprocessing: {e}")
            sys.exit(1)

    elif getattr(args, 'open_orifices', False):
        # --- Workflow: Open Orifices ---
        print("Workflow Action: Running Standalone Orifice Opening script...")
        if not STANDALONE_ORIFICE_OPENING_AVAILABLE:
            print("Error: Cannot run orifice opening - required scripts not imported or failed.")
            sys.exit(1)
        try:
            orifice_func = open_orifices_with_curvature if getattr(args, 'use_curvature_to_open', True) else open_orifices_manually
            print(f"  Calling: {orifice_func.__name__}")
            atrium_for_opening = atria_to_process[0]
            print(f"  Performing orifice opening for: {atrium_for_opening}")

            # Use getattr to safely access LAA/RAA from args
            laa_arg = getattr(args, 'LAA', None)
            raa_arg = getattr(args, 'RAA', None)

            # **MODIFIED CALL** - Expecting return values (cut_mesh_path, apex_id)
            # Ensure standalone script is modified to return these
            cut_mesh_path, apex_id = orifice_func(
                meshpath=mesh_abs_path,
                atrium=atrium_for_opening,
                MRI=getattr(args, 'MRI', 0),
                scale=getattr(args, 'scale', 1.0),
                size=getattr(args, 'size', 30),
                min_cutting_radius=getattr(args, 'min_cutting_radius', 7.5),
                max_cutting_radius=getattr(args, 'max_cutting_radius', 17.5),
                # Pass LAA/RAA as strings if not None, else empty string
                LAA=str(laa_arg) if laa_arg is not None else "",
                RAA=str(raa_arg) if raa_arg is not None else "",
                debug=getattr(args, 'debug', 0)
            )

            if not cut_mesh_path or not os.path.exists(cut_mesh_path):
                 raise FileNotFoundError(f"Orifice opening script did not return a valid output file path: {cut_mesh_path}")
            if apex_id is None or apex_id < 0:
                 raise ValueError("Orifice opening script did not return a valid apex ID.")

            print(f"  Orifice opening complete. Output mesh: {cut_mesh_path}")
            print(f"  Determined Apex ID for {atrium_for_opening}: {apex_id}")

            surface_mesh_path_for_rings = cut_mesh_path
            processed_mesh_base = os.path.splitext(cut_mesh_path)[0]

            if atrium_for_opening == 'LA': current_LAA = int(apex_id)
            if atrium_for_opening == 'RA': current_RAA = int(apex_id)
            generator.la_apex = current_LAA
            generator.ra_apex = current_RAA
            print(f"  Generator apex IDs updated: LAA={generator.la_apex}, RAA={generator.ra_apex}")
            needs_volume_mesh_generation = True

        except Exception as e:
            print(f"Error during orifice opening call: {e}")
            sys.exit(1)

    elif getattr(args, 'find_appendage', False) and not getattr(args, 'resample_input', False):
        # --- Workflow: Find Appendage Manually ---
        print("Workflow Action: Manual Apex Selection...")
        try:
            print("  Loading mesh for apex selection...")
            polydata = apply_vtk_geom_filter(smart_reader(mesh_abs_path))
            if polydata is None or polydata.GetNumberOfPoints() == 0:
                raise ValueError("Input mesh for apex selection is empty or invalid.")

            mesh_pv = pv.PolyData(polydata)
            print(f"  Loaded {mesh_pv.n_points} points.")
            tree = cKDTree(mesh_pv.points.astype(np.double))

            if 'LA' in atria_to_process:
                print("  Please select the LEFT Atrial Appendage (LAA) apex in the popup window...")
                picked_la = pick_point(mesh_pv, "LEFT Atrial Appendage (LAA) apex")
                if picked_la is None: raise RuntimeError("LAA apex selection cancelled or failed.")
                _, laa_idx = tree.query(picked_la)
                current_LAA = int(laa_idx)
                print(f"  LAA Apex ID selected: {current_LAA}")

            if 'RA' in atria_to_process:
                print("  Please select the RIGHT Atrial Appendage (RAA) apex in the popup window...")
                picked_ra = pick_point(mesh_pv, "RIGHT Atrial Appendage (RAA) apex")
                if picked_ra is None: raise RuntimeError("RAA apex selection cancelled or failed.")
                _, raa_idx = tree.query(picked_ra)
                current_RAA = int(raa_idx)
                print(f"  RAA Apex ID selected: {current_RAA}")

            generator.la_apex = current_LAA
            generator.ra_apex = current_RAA
            print(f"  Generator apex IDs updated: LAA={generator.la_apex}, RAA={generator.ra_apex}")

            # Save to CSV like original script
            apex_id_data = {}
            laa_id_to_save = getattr(generator, 'la_apex', None)
            raa_id_to_save = getattr(generator, 'ra_apex', None)
            # Determine column names based on args.atrium for compatibility
            if args.atrium == "LA" and laa_id_to_save is not None:
                apex_id_data['LAA_id'] = [laa_id_to_save] # Use specific key
            elif args.atrium == "RA" and raa_id_to_save is not None:
                apex_id_data['RAA_id'] = [raa_id_to_save] # Use specific key
            elif args.atrium == "LA_RA":
                 if laa_id_to_save is not None: apex_id_data['LAA_id'] = [laa_id_to_save]
                 if raa_id_to_save is not None: apex_id_data['RAA_id'] = [raa_id_to_save]

            if apex_id_data:
                 apex_csv_path = f'{processed_mesh_base}_mesh_data.csv'
                 df_apex = pd.DataFrame(apex_id_data)
                 df_apex.to_csv(apex_csv_path, index=False, float_format="%.2f")
                 print(f"  Apex IDs saved to {apex_csv_path}")

        except Exception as e:
            print(f"Error during manual apex selection: {e}")
            sys.exit(1)
        surface_mesh_path_for_rings = mesh_abs_path
        processed_mesh_base = mesh_base_path_orig
        needs_volume_mesh_generation = True

    else:
         # --- Workflow: Standard Open Surface ---
        print("Workflow Action: Using provided --mesh directly (standard open surface).")
        # Ensure apex IDs were provided via args or updated previously
        if 'LA' in atria_to_process and current_LAA is None:
             print("Error: --LAA must be provided if using standard open surface workflow for LA.")
             sys.exit(1)
        if 'RA' in atria_to_process and current_RAA is None:
             print("Error: --RAA must be provided if using standard open surface workflow for RA.")
             sys.exit(1)
        # Ensure generator has the correct IDs
        generator.la_apex = current_LAA
        generator.ra_apex = current_RAA
        surface_mesh_path_for_rings = mesh_abs_path
        processed_mesh_base = mesh_base_path_orig
        needs_volume_mesh_generation = True


    # --- Step 2: SSM Fitting (Optional, Procedural Standalones) ---
    print("\n--- Step 2: SSM Fitting (Optional) ---")
    if getattr(args, 'SSM_fitting', False):
        if not SSM_TOOLS_AVAILABLE:
             print("SSM Tools not available, skipping SSM fitting.")
        elif getattr(args, 'closed_surface', False):
             print("Warning: SSM fitting is typically performed on open surfaces. Skipping for --closed_surface.")
        else:
            print("Workflow Action: Running procedural SSM fitting steps...")
            # This block remains largely procedural, calling standalone scripts
            try:
                # Use getattr for safe access to SSM args
                ssm_file_arg = getattr(args, 'SSM_file', 'data/SSM/SSM.h5')
                ssm_basename_arg = getattr(args, 'SSM_basename', 'data/SSM/mean')
                ssm_abs_basename = os.path.abspath(ssm_basename_arg)
                ssm_abs_file = os.path.abspath(ssm_file_arg)
                ssm_surf_dir = ssm_abs_basename + '_surf'
                os.makedirs(ssm_surf_dir, exist_ok=True)

                # --- Generate landmarks for base SSM shape ---
                ssm_landmarks_path = os.path.join(ssm_surf_dir, 'landmarks.json')
                if not os.path.isfile(ssm_landmarks_path):
                    print(f"  Generating landmarks for SSM base shape: {ssm_abs_basename}")
                    ssm_laa_id = 6329; ssm_raa_id = 21685 # Hardcoded, verify!
                    ssm_base_mesh_path = ssm_abs_basename + ".vtk"
                    if not os.path.exists(ssm_base_mesh_path): ssm_base_mesh_path = ssm_abs_basename + ".obj"
                    if not os.path.exists(ssm_base_mesh_path): raise FileNotFoundError(f"SSM base mesh not found: {ssm_abs_basename}.vtk/obj")

                    print(f"  Using SSM Base Mesh: {ssm_base_mesh_path}, LAA={ssm_laa_id}, RAA={ssm_raa_id}")
                    ssm_ring_generator = AtrialBoundaryGenerator(mesh_path=ssm_base_mesh_path, la_apex=ssm_laa_id, ra_apex=ssm_raa_id, debug=(args.debug > 0))
                    ssm_ring_generator.extract_rings(surface_mesh_path=ssm_base_mesh_path) # Standard rings for landmarks
                    print(f"  Standard ring extraction complete for SSM base.")
                    get_landmarks(ssm_abs_basename, 0, 1)
                    print(f"  Landmarks generated for SSM base.")
                else:
                    print(f"  Using existing landmarks for SSM base shape: {ssm_landmarks_path}")

                # --- Prepare Target Mesh ---
                target_mesh_base_for_ssm = processed_mesh_base
                target_surf_dir = target_mesh_base_for_ssm + '_surf'
                os.makedirs(target_surf_dir, exist_ok=True)

                if not os.path.exists(os.path.join(target_surf_dir, 'landmarks.json')):
                     print(f"  Running ring detection for TARGET mesh to generate landmarks: {target_mesh_base_for_ssm}")
                     if not os.path.exists(surface_mesh_path_for_rings): raise FileNotFoundError(f"Target surface mesh for SSM landmark generation not found: {surface_mesh_path_for_rings}")
                     generator.la_apex = current_LAA; generator.ra_apex = current_RAA # Ensure current IDs
                     generator.extract_rings(surface_mesh_path=surface_mesh_path_for_rings) # Standard rings
                     print(f"  Target ring extraction complete.")

                # --- Prealign target mesh ---
                print(f"  Prealigning target mesh: {target_mesh_base_for_ssm}...")
                prealign_meshes(target_mesh_base_for_ssm, ssm_abs_basename, args.atrium, 0)

                # --- Get landmarks for (prealigned) target mesh ---
                print(f"  Generating landmarks for (prealigned) target mesh: {target_mesh_base_for_ssm}...")
                get_landmarks(target_mesh_base_for_ssm, 1, 1)

                # --- Create registration file ---
                print(f"  Setting up registration ({target_surf_dir} vs {ssm_surf_dir})...")
                reg_template_path = 'template/Registration_ICP_GP_template.txt'
                if not os.path.exists(reg_template_path): raise FileNotFoundError(f"Reg template not found: {reg_template_path}")
                with open(reg_template_path, 'r') as f: lines = ''.join(f.readlines())
                temp_obj = Template(lines)
                SSM_fit_file = temp_obj.substitute( SSM_file=ssm_abs_file, SSM_dir=os.path.abspath(ssm_surf_dir), target_dir=os.path.abspath(target_surf_dir))
                reg_output_path = os.path.join(target_surf_dir, 'Registration_ICP_GP.txt')
                with open(reg_output_path, 'w') as f: f.write(SSM_fit_file)
                print(f"  Registration file written: {reg_output_path}. Run registration externally.")

                # --- Check for coefficients file ---
                coeffs_path = os.path.join(target_surf_dir, 'coefficients.txt')
                if not os.path.isfile(coeffs_path): raise ValueError(f"SSM Coefficients file not found: {coeffs_path}. Run registration first.")
                print(f"  Found coefficients file: {coeffs_path}")

                # --- Create SSM instance ---
                ssm_instance_base = os.path.join(target_surf_dir, f"{args.atrium}_fit") # Naming based on original script
                ssm_instance_obj = ssm_instance_base + ".obj"
                print(f"  Creating SSM instance: {ssm_instance_obj}...")
                create_SSM_instance(ssm_abs_file, coeffs_path, ssm_instance_obj) # Assumes .h5 is handled internally

                processed_mesh_base = ssm_instance_base # Update base name
                surface_mesh_path_for_rings = ssm_instance_obj # Use this .obj for subsequent ring detection
                print(f"  SSM instance created. Updated processed mesh base: {processed_mesh_base}")

                # --- Optional Resampling after fitting ---
                if getattr(args, 'resample_input', False):
                    print(f"  Resampling SSM instance...")
                    resample_surf_mesh(
                        processed_mesh_base, # Base name like .../LA_fit
                        target_mesh_resolution=args.target_mesh_resolution,
                        find_apex_with_curv=1, # Let resampling find apex
                        scale=args.scale,
                        apex_id=-1, # Let resampling find
                        atrium=args.atrium # Pass correct atrium
                    )
                    processed_mesh_base = processed_mesh_base + '_res' # Update base name
                    resampled_ply_path = processed_mesh_base + ".ply" # Resample outputs ply
                    if not os.path.exists(resampled_ply_path): raise FileNotFoundError(f"Resampling output not found: {resampled_ply_path}")

                    # --- FIX: Convert PLY to VTK before ring detection ---
                    resampled_vtk_path = processed_mesh_base + ".vtk"
                    print(f"  Converting resampled PLY '{resampled_ply_path}' to VTK '{resampled_vtk_path}'...")
                    try:
                        mesh_ply = pv.read(resampled_ply_path)
                        mesh_ply.save(resampled_vtk_path, binary=True) # Save as VTK
                        surface_mesh_path_for_rings = resampled_vtk_path # Use VTK for rings
                        print(f"  Conversion complete. Using VTK for ring detection.")
                    except Exception as conv_err:
                         print(f"  Error converting PLY to VTK: {conv_err}. Ring detection might fail.")
                         surface_mesh_path_for_rings = resampled_ply_path # Fallback to PLY
                    # --- END FIX ---

                    print(f"  Resampling complete. Updated processed mesh base: {processed_mesh_base}")

                    # Read apex IDs potentially generated by resampling
                    resampled_apex_csv = f"{processed_mesh_base}_mesh_data.csv"
                    if os.path.exists(resampled_apex_csv):
                        print(f"  Reading apex IDs from resampled data CSV: {resampled_apex_csv}")
                        df_res_apex = pd.read_csv(resampled_apex_csv)
                        if 'LAA_id' in df_res_apex.columns and pd.notna(df_res_apex['LAA_id'][0]): current_LAA = int(df_res_apex['LAA_id'][0]); print(f"    Updated LAA: {current_LAA}")
                        if 'RAA_id' in df_res_apex.columns and pd.notna(df_res_apex['RAA_id'][0]): current_RAA = int(df_res_apex['RAA_id'][0]); print(f"    Updated RAA: {current_RAA}")
                        generator.la_apex = current_LAA; generator.ra_apex = current_RAA
                    else:
                         print("  Warning: No apex ID CSV found after resampling.")

            except Exception as e:
                print(f"Error during SSM fitting workflow: {e}")
                sys.exit(1)
        # END SSM Fitting block
    else:
        print("Workflow Action: No SSM Fitting.")
        # --- Optional Resampling if NOT doing SSM fitting ---
        if getattr(args, 'resample_input', False) and not getattr(args, 'closed_surface', False):
             if not SSM_TOOLS_AVAILABLE or resample_surf_mesh is None:
                 print("Resample script not available, skipping resampling.")
             else:
                print("Workflow Action: Resampling input mesh...")
                try:
                    resample_input_base = processed_mesh_base
                    resample_input_path = surface_mesh_path_for_rings
                    temp_obj_path = None
                    if not resample_input_path.lower().endswith(".obj"):
                         temp_obj_path = resample_input_base + "_temp_for_resample.obj"
                         print(f"  Converting {resample_input_path} to {temp_obj_path} for resampling...")
                         temp_mesh_pv = pv.read(resample_input_path)
                         pv.save_meshio(temp_obj_path, temp_mesh_pv, "obj")
                         resample_input_base = os.path.splitext(temp_obj_path)[0]

                    apex_id_for_resample = -1
                    atrium_arg_for_resample = args.atrium if args.atrium != 'LA_RA' else 'LA'
                    if atrium_arg_for_resample == 'LA' and current_LAA is not None: apex_id_for_resample = current_LAA
                    elif atrium_arg_for_resample == 'RA' and current_RAA is not None: apex_id_for_resample = current_RAA

                    print(f"  Calling resample_surf_mesh on base: {resample_input_base}, Apex ID: {apex_id_for_resample}")
                    resample_surf_mesh(
                        resample_input_base,
                        target_mesh_resolution=args.target_mesh_resolution,
                        find_apex_with_curv=0, # Use provided apex ID
                        scale=args.scale,
                        apex_id=apex_id_for_resample,
                        atrium=atrium_arg_for_resample
                    )
                    processed_mesh_base = resample_input_base + '_res'
                    resampled_ply_path = processed_mesh_base + ".ply"
                    if not os.path.exists(resampled_ply_path): raise FileNotFoundError(f"Resampling output not found: {resampled_ply_path}")

                    # --- FIX: Convert PLY to VTK before ring detection ---
                    resampled_vtk_path = processed_mesh_base + ".vtk"
                    print(f"  Converting resampled PLY '{resampled_ply_path}' to VTK '{resampled_vtk_path}'...")
                    try:
                        mesh_ply = pv.read(resampled_ply_path)
                        mesh_ply.save(resampled_vtk_path, binary=True) # Save as VTK
                        surface_mesh_path_for_rings = resampled_vtk_path # Use VTK for rings
                        print(f"  Conversion complete. Using VTK for ring detection.")
                    except Exception as conv_err:
                         print(f"  Error converting PLY to VTK: {conv_err}. Ring detection might fail.")
                         surface_mesh_path_for_rings = resampled_ply_path # Fallback
                    # --- END FIX ---

                    print(f"  Resampling complete. Updated processed mesh base: {processed_mesh_base}")

                    if temp_obj_path is not None and os.path.exists(temp_obj_path) and "_temp_for_resample" in temp_obj_path:
                         print(f"  Removing temporary file: {temp_obj_path}")
                         os.remove(temp_obj_path)

                    # Read apex IDs potentially generated by resampling
                    resampled_apex_csv = f"{processed_mesh_base}_mesh_data.csv"
                    if os.path.exists(resampled_apex_csv):
                        print(f"  Reading apex IDs from resampled data CSV: {resampled_apex_csv}")
                        df_res_apex = pd.read_csv(resampled_apex_csv)
                        if 'LAA_id' in df_res_apex.columns and pd.notna(df_res_apex['LAA_id'][0]): current_LAA = int(df_res_apex['LAA_id'][0]); print(f"    Updated LAA: {current_LAA}")
                        if 'RAA_id' in df_res_apex.columns and pd.notna(df_res_apex['RAA_id'][0]): current_RAA = int(df_res_apex['RAA_id'][0]); print(f"    Updated RAA: {current_RAA}")
                        generator.la_apex = current_LAA; generator.ra_apex = current_RAA
                    else:
                         print("  Warning: No apex ID CSV found after resampling.")

                except Exception as e:
                    print(f"Error during resampling: {e}")
                    sys.exit(1)
        # END Resampling block


    # --- Step 3: Ring Detection (OOP) ---
    print("\n--- Step 3: Ring Detection (OOP) ---")
    if not os.path.exists(surface_mesh_path_for_rings):
        print(f"Error: Cannot find surface mesh for ring detection: {surface_mesh_path_for_rings}")
        sys.exit(1)

    # Final check on apex IDs in the generator instance
    if 'LA' in atria_to_process and generator.la_apex is None:
         print("Error: LAA apex ID is missing in generator before ring detection step.")
         sys.exit(1)
    if 'RA' in atria_to_process and generator.ra_apex is None:
         print("Error: RAA apex ID is missing in generator before ring detection step.")
         sys.exit(1)
    print(f"Final Apex IDs for Ring Detection: LAA={generator.la_apex}, RAA={generator.ra_apex}")

    print(f"Running ring detection on: {surface_mesh_path_for_rings}")
    # Output: Creates {processed_mesh_base}_surf/ directory
    try:
        # Determine ring workflow based on original implicit logic:
        # Use top_epi_endo if closed_surface was true, otherwise standard.
        if getattr(args, 'closed_surface', False):
            print("  Using 'top_epi_endo' ring workflow (since --closed_surface was specified).")
            # Derive expected endo path based on the *processed_mesh_base* after separation
            atrium_for_endo = 'LA' if 'LA' in atria_to_process else 'RA' # Pick one
            expected_endo_path = f"{processed_mesh_base}_{atrium_for_endo}_endo.obj"
            print(f"  Expecting endo mesh at: {expected_endo_path}")
            if not os.path.exists(expected_endo_path):
                 if args.atrium == 'LA_RA':
                      atrium_for_endo = 'RA' if atrium_for_endo == 'LA' else 'LA'
                      expected_endo_path = f"{processed_mesh_base}_{atrium_for_endo}_endo.obj"
                      print(f"  Checking alternate endo path: {expected_endo_path}")
                 if not os.path.exists(expected_endo_path):
                      print(f"  Warning: Required endo mesh '{expected_endo_path}' not found for top_epi_endo workflow.")
            generator.extract_rings_top_epi_endo(surface_mesh_path=surface_mesh_path_for_rings)
        else:
            print("  Using 'standard' ring workflow.")
            generator.extract_rings(surface_mesh_path=surface_mesh_path_for_rings)

        print(f"Ring detection complete.")
        expected_ring_outdir = f"{processed_mesh_base}_surf"
        if not os.path.isdir(expected_ring_outdir):
            print(f"Warning: Expected ring output directory not found: {expected_ring_outdir}")

    except Exception as e:
        print(f"Error during ring detection step: {e}")
        sys.exit(1)

    # --- Step 4: Generate Volume Mesh (OOP) ---
    # Generate volume if needed implicitly (e.g., for surf_ids or vol LDRBM) and not already available
    print("\n--- Step 4: Generate Volume Mesh (Implicitly if needed) ---")
    volume_mesh_generated = False
    volume_mesh_base_path_for_surfids = None # Base path of volume mesh

    # Determine if volume mesh generation is needed
    # Needed if: Not closed surface AND (LDRBM will run on volume OR surf_ids will be generated)
    # Simplified: Generate if we started from an open surface.
    if volumetric_mesh_path is None and not getattr(args, 'closed_surface', False):
        needs_volume_mesh_generation = True

    if needs_volume_mesh_generation:
        if shutil.which("meshtool") is None:
            print("Error: Cannot generate volume mesh - 'meshtool' not found in PATH.")
            sys.exit(1)
        print("Workflow Action: Generating Volume Mesh (Implicit)...")
        surface_for_volume_gen_path = surface_mesh_path_for_rings # Use the mesh that went into ring detection
        if not surface_for_volume_gen_path.lower().endswith(".obj"):
             obj_path_for_vol = processed_mesh_base + ".obj" # Use current base name
             if not os.path.exists(obj_path_for_vol):
                  print(f"  Converting {surface_for_volume_gen_path} to {obj_path_for_vol} for meshtool...")
                  try:
                      temp_mesh_pv = pv.read(surface_for_volume_gen_path)
                      pv.save_meshio(obj_path_for_vol, temp_mesh_pv, "obj")
                  except Exception as conv_err:
                      print(f"Error converting surface to OBJ for meshtool: {conv_err}")
                      sys.exit(1)
             surface_for_volume_gen_path = obj_path_for_vol

        if surface_for_volume_gen_path and os.path.exists(surface_for_volume_gen_path):
            print(f"  Generating volume mesh from: {surface_for_volume_gen_path}")
            try:
                generator.generate_mesh(input_surface_path=surface_for_volume_gen_path)
                volumetric_mesh_path = processed_mesh_base + "_vol.vtk" # Expected output
                if not os.path.exists(volumetric_mesh_path):
                     raise RuntimeError(f"Meshtool did not produce expected output: {volumetric_mesh_path}")
                print(f"  Volume mesh generated: {volumetric_mesh_path}")
                volume_mesh_generated = True
            except Exception as e:
                print(f"Error during implicit volume mesh generation: {e}")
                sys.exit(1)
        else:
            print(f"Error: Could not determine or find input surface for implicit volume generation: {surface_for_volume_gen_path}")
            sys.exit(1)
    elif volumetric_mesh_path is not None:
         print("Volume mesh already available, skipping generation.")
    else:
         print("Volume mesh generation not required for this workflow.")

    # Set base path for volume mesh used by surf_ids
    if volumetric_mesh_path:
        volume_mesh_base_path_for_surfids = os.path.splitext(volumetric_mesh_path)[0]


    # --- Step 5: Generate Surface IDs (OOP) ---
    # Run unconditionally if volume mesh exists
    print("\n--- Step 5: Generate Surface IDs ---")
    if volumetric_mesh_path and os.path.exists(volumetric_mesh_path):
        print("Workflow Action: Generate Surface IDs -> Calling generator.generate_surf_id...")
        print(f"  Using Volume Mesh: {volumetric_mesh_path}")
        # Depends on outputs from separate_epi_endo ({processed_mesh_base}_{atrium}_[epi|endo].obj)
        # Depends on outputs from extract_rings ({processed_mesh_base}_surf/ids_*.vtx)
        try:
            for atrium_tag in atria_to_process:
                print(f"  Generating surface IDs for {atrium_tag}...")
                generator.generate_surf_id(
                    volumetric_mesh_path=volumetric_mesh_path,
                    atrium=atrium_tag,
                    resampled=(getattr(args, 'resample_input', False)) # Use resample_input flag
                )
                print(f"  {atrium_tag} surface ID generation complete.")
            print("Surface ID generation finished.")
            # Output dir: {volume_mesh_base}_vol_surf/

        except Exception as e:
            print(f"Error during surface ID generation: {e}")
            sys.exit(1)
    else:
        print("Skipped Surface ID Generation (No volume mesh available).")

    # --- Step 6: Run LDRBM Fiber Generation ---
    # Run unconditionally if module available
    print("\n--- Step 6: Run LDRBM Fiber Generation ---")
    if not LDRBM_AVAILABLE:
         print("Skipped LDRBM Fiber Generation (Modules not available).")
    else:
        print("Workflow Action: LDRBM Fiber Generation -> Calling la_main/ra_main...")
        # Determine final mesh path BASE and TYPE for LDRBM
        mesh_for_ldrbm_base = None
        mesh_type_for_ldrbm = "surf"
        final_mesh_path_used_by_ldrbm = None

        if volumetric_mesh_path is not None:
            mesh_for_ldrbm_base = os.path.splitext(volumetric_mesh_path)[0]
            mesh_type_for_ldrbm = "vol"
            final_mesh_path_used_by_ldrbm = volumetric_mesh_path
        elif surface_mesh_path_for_rings is not None:
            mesh_for_ldrbm_base = processed_mesh_base
            mesh_type_for_ldrbm = "surf"
            final_mesh_path_used_by_ldrbm = surface_mesh_path_for_rings
        else:
            print("Error: Could not determine final mesh path for LDRBM.")
            sys.exit(1)

        print(f"  Input mesh base for LDRBM: {mesh_for_ldrbm_base}")
        print(f"  Mesh type for LDRBM: {mesh_type_for_ldrbm}")
        print(f"  Actual LDRBM input file path: {final_mesh_path_used_by_ldrbm}")

        # Auto-detect normals if needed
        current_normals_outside = getattr(args, 'normals_outside', -1)
        if current_normals_outside == -1:
             print(f"  Auto-detecting normals on: {final_mesh_path_used_by_ldrbm}")
             if not os.path.exists(final_mesh_path_used_by_ldrbm):
                  print(f"Error: Mesh file for normal detection not found: {final_mesh_path_used_by_ldrbm}")
                  sys.exit(1)
             try:
                 current_normals_outside = int(are_normals_outside(smart_reader(final_mesh_path_used_by_ldrbm)))
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

        try:
            if 'LA' in atria_to_process:
                print("  Running LA LDRBM...")
                if la_main: la_main.run(ldrbm_common_args[:])
                else: print("   Error: la_main not imported.")
                print("  LA LDRBM finished.")
            if 'RA' in atria_to_process:
                print("  Running RA LDRBM...")
                if ra_main: ra_main.run(ldrbm_common_args[:])
                else: print("   Error: ra_main not imported.")
                print("  RA LDRBM finished.")

            # Optional conversion calls from original script
            # Add back if needed, ensuring paths match LDRBM output structure
            # Example for LA_RA case:
            # if args.atrium == 'LA_RA':
            #     print("  Running meshtool conversions for LA_RA case...")
            #     # ... os.system("meshtool convert ... ") calls ...


        except Exception as e:
            print(f"Error during LDRBM execution: {e}")
            sys.exit(1)


    print("\n--- AugmentA Pipeline Finished ---")

# Debug plotting section omitted.

