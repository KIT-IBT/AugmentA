# ---top_epi_endo.py---
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
top_epi_endo.py

This module implements the top epi/endo extraction logic for atrial boundaries.
It is a complete, self-contained replacement for the old procedural function.
"""

import os
from glob import glob
import numpy as np
import pandas as pd
import vtk
from vtk.numpy_interface import dataset_adapter as dsa

# Import MeshReader
from Atrial_LDRBM.Generate_Boundaries.mesh import MeshReader
from vtk_opencarp_helper_methods.vtk_methods.filters import apply_vtk_geom_filter, clean_polydata, generate_ids
from vtk_opencarp_helper_methods.vtk_methods.thresholding import get_threshold_between
from vtk_opencarp_helper_methods.vtk_methods.init_objects import init_connectivity_filter, ExtractionModes, initialize_plane_with_points
from vtk_opencarp_helper_methods.vtk_methods.finder import find_closest_point
from vtk_opencarp_helper_methods.vtk_methods.converters import vtk_to_numpy
from Atrial_LDRBM.Generate_Boundaries.file_manager import write_vtk, write_vtx_file, write_csv
from Atrial_LDRBM.Generate_Boundaries.ring_detector import RingDetector # Ensure this import is present


def label_atrial_orifices_TOP_epi_endo(mesh: str, LAA_id: str = "", RAA_id: str = "",
                                      LAA_base_id: str = "", RAA_base_id: str = "") -> None:
    """
    Extracts rings with top epi/endo logic for atrial boundaries.

    Steps:
      - Read the mesh using MeshReader and apply a geometry filter.
      - Remove any existing ID files from the output directory.
      - If both apex indices are provided, process both LA and RA regions using RingDetector.
      - Otherwise, process the single available region.
      - For RA processing, call the specialized method in RingDetector for TOP_EPI/ENDO.
      - Write the computed centroids to a CSV file.

    Parameters:
        mesh: Path to the mesh file (assumed epi or combined surface).
        LAA_id: LA apex point index as a string.
        RAA_id: RA apex point index as a string.
        LAA_base_id: LA base point index as a string.
        RAA_base_id: RA base point index as a string.
    """

    try:
        mesh_loader = MeshReader(mesh)
        mesh_surf = mesh_loader.get_polydata() # Already filtered by MeshReader
        if mesh_surf is None or mesh_surf.GetNumberOfPoints() == 0:
             raise ValueError("Input mesh is empty or failed to load.")
    except Exception as e:
         print(f"Error loading input mesh {mesh}: {e}")
         raise


    centroids = {}
    extension = mesh.split('.')[-1]
    mesh_root = mesh[:-(len(extension) + 1)] # e.g., /path/to/mesh_RA

    # Construct base name assuming input is like mesh_RA.vtk -> mesh
    base_name_parts = mesh_root.split('_')
    if len(base_name_parts) > 1 and base_name_parts[-1] in ['RA', 'LA', 'epi', 'endo']:
         mesh_base = '_'.join(base_name_parts[:-1]) # e.g. /path/to/mesh
    else:
         mesh_base = mesh_root # Assume input name doesn't have _RA/_LA suffix

    outdir = f"{mesh_base}_surf" # Output dir like /path/to/mesh_surf
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for file_path in glob(os.path.join(outdir, 'ids_*')):
        os.remove(file_path)

    # Find the corresponding endo mesh path - assuming consistent naming convention
    endo_mesh_path = f"{mesh_base}_endo.obj"
    print(f"Expecting endo mesh at: {endo_mesh_path}")
    if not os.path.exists(endo_mesh_path):
        print(f"WARNING: Corresponding endo mesh '{endo_mesh_path}' not found. Cannot perform TOP_ENDO analysis.")

    # Process biatrial case
    if LAA_id != "" and RAA_id != "":
        LA_ap_point = mesh_surf.GetPoint(int(LAA_id))
        RA_ap_point = mesh_surf.GetPoint(int(RAA_id))
        centroids["LAA"] = LA_ap_point
        centroids["RAA"] = RA_ap_point
        if LAA_base_id != "" and RAA_base_id != "":
            LA_bs_point = mesh_surf.GetPoint(int(LAA_base_id))
            RA_bs_point = mesh_surf.GetPoint(int(RAA_base_id))
            centroids["LAA_base"] = LA_bs_point
            centroids["RAA_base"] = RA_bs_point

        # Process LA region (standard)
        # Run connectivity on the loaded surface mesh
        mesh_conn = init_connectivity_filter(mesh_surf, ExtractionModes.ALL_REGIONS, True).GetOutput()
        arr = mesh_conn.GetPointData().GetArray("RegionId")
        if not arr: # Check if connectivity filter worked
            raise RuntimeError("Connectivity filter failed to produce RegionId array.")

        arr.SetName("RegionID")
        id_vector = vtk_to_numpy(arr)

        # Find closest point might fail if IDs change across filters
        new_LAA_id = find_closest_point(mesh_conn, LA_ap_point)

        if new_LAA_id < 0:
            raise ValueError("Could not find closest point for LAA apex.")

        LA_tag = id_vector[int(new_LAA_id)]

        la_thresh = get_threshold_between(mesh_conn, LA_tag, LA_tag,
                                          "vtkDataObject::FIELD_ASSOCIATION_POINTS", "RegionID")

        LA_poly_ug = la_thresh.GetOutput()
        LA_poly = apply_vtk_geom_filter(LA_poly_ug)

        # Ensure LA_poly is valid before proceeding
        if LA_poly is None or LA_poly.GetNumberOfPoints() == 0:
             raise ValueError("Thresholding LA region resulted in empty geometry.")

        LA_region = generate_ids(LA_poly, "Ids", "CellIds") # Add 'Ids' array
        if not LA_region.GetPointData().GetArray("Ids"):
             raise RuntimeError("Failed to generate 'Ids' array on LA region.")

        write_vtk(os.path.join(outdir, 'LA.vtk'), LA_region)

        adjusted_LAA = find_closest_point(LA_region, LA_ap_point)
        if adjusted_LAA < 0:
            raise ValueError("Could not adjust LAA apex on isolated LA region.")

        b_tag = np.zeros((LA_region.GetNumberOfPoints(),))

        detector_LA = RingDetector(LA_region, LA_ap_point, outdir)
        LA_rings = detector_LA.detect_rings()
        b_tag, centroids = detector_LA.mark_la_rings(adjusted_LAA, LA_rings, b_tag, centroids, LA_region)
        ds = dsa.WrapDataObject(LA_region)
        ds.PointData.append(b_tag, 'boundary_tag')

        write_vtk(os.path.join(outdir, 'LA_boundaries_tagged.vtk'), ds.VTKObject)

        # Re-use mesh_conn from LA processing
        new_RAA_id = find_closest_point(mesh_conn, RA_ap_point)
        if new_RAA_id < 0:
            raise ValueError("Could not find closest point for RAA apex.")

        RA_tag = id_vector[int(new_RAA_id)]

        ra_thresh = get_threshold_between(mesh_conn, RA_tag, RA_tag,
                                          "vtkDataObject::FIELD_ASSOCIATION_POINTS", "RegionID")
        ra_poly_ug = ra_thresh.GetOutput()
        ra_poly = apply_vtk_geom_filter(ra_poly_ug) # Ensure PolyData

        if ra_poly is None or ra_poly.GetNumberOfPoints() == 0:
             raise ValueError("Thresholding RA region resulted in empty geometry.")

        RA_region = generate_ids(ra_poly, "Ids", "CellIds")
        if not RA_region.GetPointData().GetArray("Ids"):
             raise RuntimeError("Failed to generate 'Ids' array on RA region.")

        write_vtk(os.path.join(outdir, 'RA.vtk'), RA_region)

        adjusted_RAA = find_closest_point(RA_region, RA_ap_point)
        if adjusted_RAA < 0:
            raise ValueError("Could not adjust RAA apex on isolated RA region.")

        b_tag_ra = np.zeros((RA_region.GetNumberOfPoints()))

        # Use RingDetector for RA ring marking
        detector_RA = RingDetector(RA_region, RA_ap_point, outdir)
        RA_rings = detector_RA.detect_rings()
        b_tag_ra, centroids, RA_rings = detector_RA.mark_ra_rings(adjusted_RAA, RA_rings, b_tag_ra, centroids)

        # Pass the isolated RA region polydata (RA_region) as the 'model_epi'
        detector_RA.perform_tv_split_and_find_top_epi_endo(RA_region, endo_mesh_path, RA_rings)

        ds_ra = dsa.WrapDataObject(RA_region)
        ds_ra.PointData.append(b_tag_ra, 'boundary_tag')
        write_vtk(os.path.join(outdir, 'RA_boundaries_tagged.vtk'), ds_ra.VTKObject)

    elif RAA_id == "": # Only LA provided
        write_vtk(os.path.join(outdir, 'LA.vtp'), mesh_surf, xml_format=True)

        LA_ap_point = mesh_surf.GetPoint(int(LAA_id))
        centroids["LAA"] = LA_ap_point

        if mesh_surf.GetPointData().GetArray("Ids") is None:
            print("Warning: Input LA mesh missing 'Ids'. Generating...")
            LA_region = generate_ids(mesh_surf, "Ids", "CellIds")
        else:
            LA_region = mesh_surf

        if not LA_region.GetPointData().GetArray("Ids"):
             raise RuntimeError("Failed to get/generate 'Ids' on LA mesh.")

        b_tag = np.zeros((LA_region.GetNumberOfPoints()))
        adjusted_LAA = find_closest_point(LA_region, LA_ap_point)
        if adjusted_LAA < 0:
            raise ValueError("Could not adjust LAA apex on LA mesh.")

        detector_LA = RingDetector(LA_region, LA_ap_point, outdir)
        LA_rings = detector_LA.detect_rings()
        b_tag, centroids = detector_LA.mark_la_rings(adjusted_LAA, LA_rings, b_tag, centroids, LA_region)
        ds = dsa.WrapDataObject(LA_region)
        ds.PointData.append(b_tag, 'boundary_tag')

        write_vtk(os.path.join(outdir, 'LA_boundaries_tagged.vtp'), ds.VTKObject, xml_format=True)

    elif LAA_id == "": # Only RA provided
        write_vtk(os.path.join(outdir, 'RA.vtp'), mesh_surf)
        RA_ap_point = mesh_surf.GetPoint(int(RAA_id))
        centroids["RAA"] = RA_ap_point

        if mesh_surf.GetPointData().GetArray("Ids") is None:
            print("Warning: Input RA mesh missing 'Ids'. Generating...")
            RA_region = generate_ids(mesh_surf, "Ids", "CellIds")
        else:
            RA_region = mesh_surf

        if not RA_region.GetPointData().GetArray("Ids"):
             raise RuntimeError("Failed to get/generate 'Ids' on RA mesh.")

        b_tag = np.zeros((RA_region.GetNumberOfPoints(),))
        adjusted_RAA = find_closest_point(RA_region, RA_ap_point)
        if adjusted_RAA < 0:
            raise ValueError("Could not adjust RAA apex on RA mesh.")

        detector_RA = RingDetector(RA_region, RA_ap_point, outdir)
        RA_rings = detector_RA.detect_rings()
        b_tag, centroids, RA_rings = detector_RA.mark_ra_rings(adjusted_RAA, RA_rings, b_tag, centroids)

        detector_RA.perform_tv_split_and_find_top_epi_endo(RA_region, endo_mesh_path, RA_rings)

        ds = dsa.WrapDataObject(RA_region)
        ds.PointData.append(b_tag, 'boundary_tag')
        write_vtk(os.path.join(outdir, 'RA_boundaries_tagged.vtp'), ds.VTKObject, xml_format=True) # VTP
    else:
        raise ValueError("At least one of LA or RA apex must be provided.")

    # Check if centroids dictionary is populated before creating DataFrame
    if centroids:
        df = pd.DataFrame(list(centroids.items()), columns=['RingName', 'Coordinates'])

        # Reformat if centroids are stored differently, e.g., {name: [x,y,z]}
        df = pd.DataFrame.from_dict(centroids, orient='index', columns=['X', 'Y', 'Z'])
        df.index.name = 'RingName'

    else:
        df = pd.DataFrame(columns=['RingName', 'X', 'Y', 'Z']).set_index('RingName')
        print("Warning: No centroids generated.")


    csv_path = os.path.join(outdir, "rings_centroids.csv")
    df.to_csv(csv_path, float_format="%.2f")

    print(f"Top epi/endo extraction complete. Centroids saved to {csv_path}")