import os
import pandas as pd
from typing import Any
from vtk_opencarp_helper_methods.vtk_methods.exporting import vtk_polydata_writer, vtk_obj_writer, write_to_vtx


def write_vtk(file_path: str, polydata: Any, xml_format: bool = False) -> None:
    """
    Writes a VTK file using vtk_polydata_writer.

    If the target file has a “.vtp” extension and the caller did **not**
    explicitly request binary output (i.e., xml_format is False), the file
    will be forced to XML format to respect the XML nature of .vtp files.

    @param file_path: Destination file path for the VTK file.
    @param polydata:  The VTK polydata object to be written.
    @param xml_format: When True, force XML output regardless of extension.
    """
    # --- ensure .vtp files default to XML unless caller overrides ----------
    final_use_xml = xml_format
    if file_path.lower().endswith(".vtp") and xml_format is False:
        final_use_xml = True

    try:
        vtk_polydata_writer(file_path, polydata, store_xml=final_use_xml)
    except TypeError as te:
        # Older helper versions might not accept 'store_xml'
        print(
            f"Warning: Underlying vtk_polydata_writer might not support the "
            f"'store_xml' flag. Writing default format for {file_path}. "
            f"Error: {te}"
        )
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
    Writes a pandas DataFrame to CSV.

    If the DataFrame’s index has a name (e.g., “RingName”), the index is
    written as the first column to preserve that information; otherwise the
    index is suppressed.
    """
    write_index_as_column = bool(df.index.name is not None)
    df.to_csv(file_path, float_format="%.2f", index=write_index_as_column)



def write_vtx_file(file_path: str, array: Any) -> None:
    """
    Writes a VTX file using write_to_vtx.

    @param file_path: Destination file path for the VTX file.
    @param array: The array data to be written.
    @return: None.
    """
    write_to_vtx(file_path, array)
