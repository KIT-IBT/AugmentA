import os
import shutil
import numpy as np
from scipy.spatial import cKDTree
from vtk_opencarp_helper_methods.vtk_methods.reader import smart_reader
from vtk_opencarp_helper_methods.vtk_methods.converters import vtk_to_numpy
from file_manager import write_vtx_file

def _prepare_output_directory(mesh_path: str, atrium: str) -> str:
    base, _ = os.path.splitext(mesh_path)
    outdir = f"{base}_{atrium}_vol_surf"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir

def generate_surf_id(mesh_path: str, atrium: str, resampled: bool = False) -> None:
    """
    Generates surface ID files mapping mesh points to anatomical boundaries.
    Writes boundary files as ".vtx" files.
    """
    base = mesh_path
    vol = smart_reader(f"{base}_{atrium}_vol.vtk")
    coords = vtk_to_numpy(vol.GetPoints().GetData())
    tree = cKDTree(coords)
    epi_obj = smart_reader(f"{base}_{atrium}_epi.obj")
    epi_pts = vtk_to_numpy(epi_obj.GetPoints().GetData())
    _, epi_indices = tree.query(epi_pts)
    epi_ids = np.array(epi_indices)
    outdir = _prepare_output_directory(mesh_path, atrium)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    shutil.copyfile(f"{base}_{atrium}_vol.vtk", os.path.join(outdir, f"{atrium}.vtk"))
    res_str = "_res" if resampled else ""
    shutil.copyfile(f"{base}_{atrium}_epi{res_str}_surf/rings_centroids.csv",
                    os.path.join(outdir, "rings_centroids.csv"))
    write_vtx_file(os.path.join(outdir, "EPI.vtx"), epi_indices)
    endo_obj = smart_reader(f"{base}_{atrium}_endo.obj")
    endo_pts = vtk_to_numpy(endo_obj.GetPoints().GetData())
    _, endo_indices = tree.query(endo_pts)
    endo_indices = np.setdiff1d(endo_indices, epi_ids)
    write_vtx_file(os.path.join(outdir, "ENDO.vtx"), endo_indices)
