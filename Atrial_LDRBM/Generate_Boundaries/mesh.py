import os
import vtk
from vtk_openCARP_methods_ibt.vtk_methods.reader import smart_reader
from vtk_openCARP_methods_ibt.vtk_methods.filters import apply_vtk_geom_filter
from vtk_openCARP_methods_ibt.vtk_methods.exporting import vtk_polydata_writer, vtk_obj_writer


class Mesh:
    """
    Encapsulates a VTK polydata object and provides methods for I/O and basic operations.
    """
    def __init__(self, polydata: vtk.vtkPolyData):
        """
        Initialize the Mesh wrapper.

        :param polydata: A vtk.vtkPolyData object to encapsulate.
        :return: None
        """
        if not isinstance(polydata, vtk.vtkPolyData):
            raise TypeError("Input must be a vtk.vtkPolyData object.")
        self.polydata = polydata

    @classmethod
    def from_file(cls, mesh_path: str):
        """
        Create a Mesh instance by reading and filtering a mesh file.

        :param mesh_path: Path to the mesh file.
        :return: Mesh instance containing the loaded polydata.
        """
        if not isinstance(mesh_path, str) or not mesh_path:
            raise ValueError("mesh_path must be a non-empty string.")
        if not os.path.isfile(mesh_path):
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

        raw_data = smart_reader(mesh_path)
        polydata = apply_vtk_geom_filter(raw_data)

        if polydata is None or polydata.GetNumberOfPoints() == 0:
            raise ValueError(f"Mesh loaded from {mesh_path} is empty or invalid.")

        return cls(polydata)

    def save(self, file_path: str, xml_format: bool = False):
        """
        Save the mesh to a file, inferring format from the extension (.vtk, .vtp, .obj).

        :param file_path: Path to save the mesh file.
        :param xml_format: Force XML format for .vtp output if True.
        :return: None
        """
        if self.polydata is None or self.polydata.GetNumberOfPoints() == 0:
            raise ValueError("Cannot save an empty mesh.")
        if not isinstance(file_path, str) or not file_path:
            raise ValueError("file_path must be a non-empty string.")

        ext = os.path.splitext(file_path)[1].lower()

        if ext in ['.vtk', '.vtp']:
            is_vtp = ext == '.vtp'
            vtk_polydata_writer(file_path, self.polydata, store_xml=is_vtp or xml_format)
        elif ext == '.obj':
            vtk_obj_writer(file_path, self.polydata)
        else:
            raise ValueError(f"Unsupported file format '{ext}'. Use .vtk, .vtp, or .obj.")

    def get_polydata(self) -> vtk.vtkPolyData:
        """
        Return the underlying VTK polydata object.

        :return: vtk.vtkPolyData encapsulated by this Mesh.
        """
        return self.polydata
