import os
import pandas as pd
from typing import Any
from vtk_opencarp_helper_methods.vtk_methods.exporting import vtk_polydata_writer, vtk_obj_writer, write_to_vtx


def write_vtk(file_path: str, polydata: Any) -> None:
    """
    Writes a VTK file using vtk_polydata_writer.

    @param file_path: Destination file path for the VTK file.
    @param polydata: The VTK polydata object to be written.
    @return: None.
    """
    vtk_polydata_writer(file_path, polydata)


def write_obj(file_path: str, polydata: Any) -> None:
    """
    Writes an OBJ file using vtk_obj_writer.

    @param file_path: Destination file path for the OBJ file.
    @param polydata: The VTK polydata object to be written.
    @return: None.
    """
    vtk_obj_writer(file_path, polydata)


def write_csv(file_path: str, df: pd.DataFrame) -> None:
    """
    Writes a pandas DataFrame to a CSV file.

    @param file_path: Destination file path for the CSV file.
    @param df: The DataFrame to be written.
    @return: None.
    """
    df.to_csv(file_path, float_format="%.2f", index=False)


def write_vtx_file(file_path: str, array: Any) -> None:
    """
    Writes a VTX file using write_to_vtx.

    @param file_path: Destination file path for the VTX file.
    @param array: The array data to be written.
    @return: None.
    """
    write_to_vtx(file_path, array)
