import os
import pandas as pd
from typing import Any
from vtk_opencarp_helper_methods.vtk_methods.exporting import vtk_polydata_writer, vtk_obj_writer, write_to_vtx


def write_vtk(file_path: str, polydata: Any, xml_format: bool = False) -> None:
    """
    Writes a VTK file using vtk_polydata_writer.

    @param file_path: Destination file path for the VTK file.
    @param polydata: The VTK polydata object to be written.
    @return: None.
    """
    try:
        vtk_polydata_writer(file_path, polydata, store_xml=xml_format)
    except TypeError as te:
        # Fallback if 'store_xml' is not a valid argument for the helper
        # This might happen if the helper library version doesn't support it
        # OR if it infers based on extension only (which we shouldn't rely on)
        print(
            f"Warning: Underlying vtk_polydata_writer might not support explicit format flag 'store_xml'. Writing default format for {file_path}. Error: {te}")
        vtk_polydata_writer(file_path, polydata)
    except Exception as e:
        print(f"Error during vtk_polydata_writer call for {file_path}: {e}")
        raise


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
