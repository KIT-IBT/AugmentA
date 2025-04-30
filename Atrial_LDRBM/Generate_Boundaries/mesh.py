import os
import vtk
from vtk_opencarp_helper_methods.vtk_methods.reader import smart_reader
from vtk_opencarp_helper_methods.vtk_methods.filters import apply_vtk_geom_filter
from vtk_opencarp_helper_methods.vtk_methods.exporting import vtk_polydata_writer, vtk_obj_writer


class BaseMesh:
    """
    Base class for mesh operations. This class encapsulates a VTK polydata and provides methods for reading,
    writing, and converting the mesh.
    """
    def __init__(self, mesh_path: str = None, polydata: vtk.vtkPolyData = None):
        """
        Initializes the BaseMesh instance.

        @param mesh_path: Path to the mesh file.
        @param polydata: A VTK polydata object.
        @return: None.
        """
        self.mesh_path = mesh_path
        self.polydata = polydata

        if self.mesh_path and not self.polydata:
            self.read(self.mesh_path)

    def read(self, mesh_path: str = None) -> vtk.vtkPolyData:
        """
        Reads a mesh from file using smart_reader and applies a geometry filter.

        @param mesh_path: Optional file path to read the mesh from. If not provided, uses self.mesh_path.
        @return: The VTK polydata after applying the geometry filter.
        """
        if mesh_path:
            self.mesh_path = mesh_path
        if not self.mesh_path:
            raise ValueError("No mesh path specified for reading.")

        self.polydata = smart_reader(self.mesh_path)

        # Apply a geometry filter to ensure the mesh is in a consistent format.
        self.polydata = apply_vtk_geom_filter(self.polydata)

        return self.polydata

    def write(self, file_path: str, format: str = "vtk") -> None:
        """
        Writes the mesh to a file in the specified format.

        @param file_path: Destination file path for the mesh.
        @param format: Format to write the mesh in ("vtk" or "obj").
        @return: None.
        """
        if self.polydata is None:
            raise ValueError("No mesh loaded to write.")

        if format.lower() == "vtk":
            vtk_polydata_writer(file_path, self.polydata)
        elif format.lower() == "obj":
            vtk_obj_writer(file_path, self.polydata)
        else:
            raise ValueError("Unsupported format: choose 'vtk' or 'obj'")

    def convert(self, output_format: str) -> None:
        """
        Converts the mesh to the specified format by writing it out.

        @param output_format: The desired output format (e.g., "vtk" or "obj").
        @return: None.
        """
        # Derive the base filename (without extension).
        base, _ = os.path.splitext(self.mesh_path)
        file_path = f"{base}.{output_format.lower()}"
        self.write(file_path, format=output_format)

    def get_polydata(self) -> vtk.vtkPolyData:
        """
        Returns the underlying VTK polydata.

        @return: The VTK polydata object representing the mesh.
        """
        return self.polydata


class MeshReader(BaseMesh):
    """
    A class specifically for reading meshes from file.
    """

    def __init__(self, mesh_path: str):
        """
        Initializes a MeshReader instance and reads the mesh.

        @param mesh_path: Path to the mesh file.
        @return: None.
        """
        super().__init__(mesh_path=mesh_path)


class MeshWriter(BaseMesh):

    def write_mesh(self, file_path: str, format: str = "vtk") -> None:
        self.write(file_path, format=format)


class MeshConverter(BaseMesh):
    def convert_mesh(self, output_format: str) -> None:
        self.convert(output_format)

