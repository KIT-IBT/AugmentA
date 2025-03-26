import os
import subprocess
import vtk
from vtk_opencarp_helper_methods.vtk_methods.reader import smart_reader
from vtk_opencarp_helper_methods.vtk_methods.filters import apply_vtk_geom_filter

def load_mesh(mesh_path: str) -> vtk.vtkPolyData:
    """
    Loads a mesh from the given file path using smart_reader and applies the geometry filter.
    """
    mesh = smart_reader(mesh_path)
    return apply_vtk_geom_filter(mesh)

def generate_mesh(mesh_path: str, la_mesh_scale: float = 1.0) -> None:
    """
    Generates a volumetric mesh (VTK format) using meshtool.
    The output mesh is written to "<mesh_path>_vol".
    """
    subprocess.run([
        "meshtool",
        "generate",
        "mesh",
        f"-surf={mesh_path}.obj",
        "-ofmt=vtk",
        f"-outmsh={mesh_path}_vol"
    ], check=True)
