import os
import pandas as pd
from vtk_opencarp_helper_methods.vtk_methods.exporting import vtk_polydata_writer, vtk_obj_writer, write_to_vtx

def write_vtk(file_path: str, polydata) -> None:
    """Writes a VTK file using vtk_polydata_writer."""
    vtk_polydata_writer(file_path, polydata)

def write_obj(file_path: str, polydata) -> None:
    """Writes an OBJ file using vtk_obj_writer."""
    vtk_obj_writer(file_path, polydata)

def write_csv(file_path: str, df: pd.DataFrame) -> None:
    """Writes a DataFrame to CSV."""
    df.to_csv(file_path, float_format="%.2f", index=False)

def write_vtx_file(file_path: str, array) -> None:
    """Writes a VTX file using write_to_vtx."""
    write_to_vtx(file_path, array)
