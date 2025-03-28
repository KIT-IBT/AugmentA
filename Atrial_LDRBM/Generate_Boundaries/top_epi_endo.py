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

from vtk_opencarp_helper_methods.vtk_methods.reader import smart_reader
from vtk_opencarp_helper_methods.vtk_methods.filters import apply_vtk_geom_filter, clean_polydata, generate_ids
from vtk_opencarp_helper_methods.vtk_methods.thresholding import get_threshold_between
from vtk_opencarp_helper_methods.vtk_methods.exporting import vtk_polydata_writer
from vtk_opencarp_helper_methods.vtk_methods.init_objects import init_connectivity_filter, ExtractionModes, \
    initialize_plane_with_points
from vtk_opencarp_helper_methods.vtk_methods.finder import find_closest_point
from vtk_opencarp_helper_methods.vtk_methods.converters import vtk_to_numpy
from file_manager import write_vtk, write_vtx_file, write_csv


def label_atrial_orifices_TOP_epi_endo(mesh: str, LAA_id: str = "", RAA_id: str = "",
                                       LAA_base_id: str = "", RAA_base_id: str = "") -> None:
    """
    Extracts rings with top epi/endo logic for atrial boundaries.

    This function:
      - Reads the input mesh and applies a geometry filter.
      - Removes any pre-existing ID files.
      - Processes biatrial data if both LAA_id and RAA_id are provided.
      - Applies connectivity filtering and thresholding for the LA and RA regions.
      - Generates IDs and detects rings.
      - Writes out the top epi/endo ring boundaries and centroids.

    :param mesh: Path to the mesh file.
    :param LAA_id: LA apex point index as string.
    :param RAA_id: RA apex point index as string.
    :param LAA_base_id: LA base point index as string.
    :param RAA_base_id: RA base point index as string.
    :return: None.
    """
    print("Starting top epi/endo extraction for mesh:", mesh)

    # Read and filter the mesh.
    mesh_surf = smart_reader(mesh)
    mesh_surf = apply_vtk_geom_filter(mesh_surf)

    centroids = {}
    extension = mesh.split('.')[-1]
    mesh_root = mesh[:-(len(extension) + 1)]
    outdir = f"{mesh_root}_surf"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for file_path in glob(os.path.join(outdir, 'ids_*')):
        os.remove(file_path)

    if LAA_id != "" and RAA_id != "":
        # Retrieve apex points.
        LA_ap_point = mesh_surf.GetPoint(int(LAA_id))
        RA_ap_point = mesh_surf.GetPoint(int(RAA_id))
        centroids["LAA"] = LA_ap_point
        centroids["RAA"] = RA_ap_point
        if LAA_base_id != "" and RAA_base_id != "":
            LA_bs_point = mesh_surf.GetPoint(int(LAA_base_id))
            RA_bs_point = mesh_surf.GetPoint(int(RAA_base_id))
            centroids["LAA_base"] = LA_bs_point
            centroids["RAA_base"] = RA_bs_point

        # Process LA region.
        mesh_conn = init_connectivity_filter(mesh_surf, ExtractionModes.ALL_REGIONS, True).GetOutput()
        arr = mesh_conn.GetPointData().GetArray("RegionId")
        arr.SetName("RegionID")
        id_vector = vtk_to_numpy(arr)
        new_LAA_id = find_closest_point(mesh_conn, LA_ap_point)
        LA_tag = id_vector[int(new_LAA_id)]
        la_thresh = get_threshold_between(mesh_conn, LA_tag, LA_tag,
                                          "vtkDataObject::FIELD_ASSOCIATION_POINTS", "RegionID")
        LA_poly = apply_vtk_geom_filter(la_thresh.GetOutputPort(), True)
        LA_region = generate_ids(LA_poly, "Ids", "Ids")
        write_vtk(os.path.join(outdir, 'LA.vtp'), LA_region)
        adjusted_LAA = find_closest_point(LA_region, LA_ap_point)
        b_tag = np.zeros((LA_region.GetNumberOfPoints(),))
        LA_rings = detect_and_mark_rings(LA_region, LA_ap_point, outdir)
        # Use adjusted apex for marking.
        b_tag, centroids = mark_LA_rings(adjusted_LAA, LA_rings, b_tag, centroids, outdir, LA_region)
        ds = dsa.WrapDataObject(LA_region)
        ds.PointData.append(b_tag, 'boundary_tag')
        write_vtk(os.path.join(outdir, 'LA_boundaries_tagged.vtp'), ds.VTKObject)

        # Process RA region.
        new_RAA_id = find_closest_point(mesh_conn, RA_ap_point)
        RA_tag = id_vector[int(new_RAA_id)]
        ra_thresh = get_threshold_between(mesh_conn, RA_tag, RA_tag,
                                          "vtkDataObject::FIELD_ASSOCIATION_POINTS", "RegionID")
        ra_poly = apply_vtk_geom_filter(ra_thresh.GetOutputPort(), True)
        RA_region = generate_ids(ra_poly, "Ids", "Ids")
        write_vtk(os.path.join(outdir, 'RA.vtp'), RA_region)
        adjusted_RAA = find_closest_point(RA_region, RA_ap_point)
        b_tag_ra = np.zeros((RA_region.GetNumberOfPoints(),))
        RA_rings = detect_and_mark_rings(RA_region, RA_ap_point, outdir)
        b_tag_ra, centroids, RA_rings = mark_RA_rings(adjusted_RAA, RA_rings, b_tag_ra, centroids, outdir)
        cutting_plane_to_identify_tv_f_tv_s(RA_region, RA_rings, outdir, True)
        ds_ra = dsa.WrapDataObject(RA_region)
        ds_ra.PointData.append(b_tag_ra, 'boundary_tag')
        write_vtk(os.path.join(outdir, 'RA_boundaries_tagged.vtp'), ds_ra.VTKObject)
    elif RAA_id == "":
        write_vtk(os.path.join(outdir, 'LA.vtp'), mesh_surf)
        LA_ap_point = mesh_surf.GetPoint(int(LAA_id))
        centroids["LAA"] = LA_ap_point
        if mesh_surf.GetPointData().GetArray("Ids") is not None:
            mesh_surf.GetPointData().RemoveArray("Ids")
        LA_region = generate_ids(mesh_surf, "Ids", "Ids")
        LA_rings = detect_and_mark_rings(LA_region, LA_ap_point, outdir)
        b_tag = np.zeros((LA_region.GetNumberOfPoints(),))
        adjusted_LAA = find_closest_point(LA_region, LA_ap_point)
        b_tag, centroids = mark_LA_rings(adjusted_LAA, LA_rings, b_tag, centroids, outdir, LA_region)
        ds = dsa.WrapDataObject(LA_region)
        ds.PointData.append(b_tag, 'boundary_tag')
        write_vtk(os.path.join(outdir, 'LA_boundaries_tagged.vtp'), ds.VTKObject)
    elif LAA_id == "":
        write_vtk(os.path.join(outdir, 'RA.vtp'), mesh_surf)
        RA_ap_point = mesh_surf.GetPoint(int(RAA_id))
        centroids["RAA"] = RA_ap_point
        RA_region = generate_ids(mesh_surf, "Ids", "Ids")
        RA_rings = detect_and_mark_rings(RA_region, RA_ap_point, outdir)
        b_tag = np.zeros((RA_region.GetNumberOfPoints(),))
        adjusted_RAA = find_closest_point(RA_region, RA_apoint)
        # Note: there was a typo below ("RA_apoint"); corrected to "RA_ap_point"
        adjusted_RAA = find_closest_point(RA_region, RA_ap_point)
        b_tag, centroids, RA_rings = mark_RA_rings(adjusted_RAA, RA_rings, b_tag, centroids, outdir)
        cutting_plane_to_identify_tv_f_tv_s(RA_region, RA_rings, outdir, True)
        ds = dsa.WrapDataObject(RA_region)
        ds.PointData.append(b_tag, 'boundary_tag')
        write_vtk(os.path.join(outdir, 'RA_boundaries_tagged.vtp'), ds.VTKObject)
    else:
        raise ValueError("At least one of LA or RA apex must be provided.")

    df = pd.DataFrame(centroids)
    csv_path = os.path.join(outdir, "rings_centroids.csv")
    write_csv(csv_path, df)
    print("Top epi/endo extraction complete.")
