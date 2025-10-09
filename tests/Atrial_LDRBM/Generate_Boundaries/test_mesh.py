import pytest
import os
import vtk
import tempfile
from unittest.mock import Mock, patch, MagicMock
from Atrial_LDRBM.Generate_Boundaries.mesh import Mesh


class TestMeshInit:
    """Test the Mesh constructor."""

    def test_init_with_valid_polydata(self):
        """Test successful initialization with valid vtkPolyData."""
        polydata = vtk.vtkPolyData()
        mesh = Mesh(polydata)
        assert mesh.polydata is polydata

    def test_init_with_valid_unstructured_grid(self):
        """Test successful initialization with valid vtkUnstructuredGrid."""
        ug = vtk.vtkUnstructuredGrid()
        mesh = Mesh(ug)
        assert mesh.polydata is ug

    def test_init_with_invalid_input(self):
        """Test that TypeError is raised for invalid input types."""
        with pytest.raises(TypeError, match="Input must be a vtk.vtkPolyData or vtk.vtkUnstructuredGrid object"):
            Mesh("not_polydata")

        with pytest.raises(TypeError, match="Input must be a vtk.vtkPolyData or vtk.vtkUnstructuredGrid object"):
            Mesh(None)

        with pytest.raises(TypeError, match="Input must be a vtk.vtkPolyData or vtk.vtkUnstructuredGrid object"):
            Mesh(123)

    def test_init_with_other_vtk_types(self):
        """Test that TypeError is raised for other VTK data types."""
        with pytest.raises(TypeError, match="Input must be a vtk.vtkPolyData or vtk.vtkUnstructuredGrid object"):
            Mesh(vtk.vtkImageData())

        with pytest.raises(TypeError, match="Input must be a vtk.vtkPolyData or vtk.vtkUnstructuredGrid object"):
            Mesh(vtk.vtkStructuredGrid())


class TestMeshFromFile:
    """Test the Mesh.from_file classmethod."""

    @patch('Atrial_LDRBM.Generate_Boundaries.mesh.smart_reader')
    @patch('Atrial_LDRBM.Generate_Boundaries.mesh.apply_vtk_geom_filter')
    def test_from_file_success(self, mock_filter, mock_reader):
        """Test successful loading of a mesh file."""
        # Create a mock polydata with points
        mock_polydata = Mock(spec=vtk.vtkPolyData)
        mock_polydata.GetNumberOfPoints.return_value = 10

        mock_reader.return_value = Mock()
        mock_filter.return_value = mock_polydata

        with tempfile.NamedTemporaryFile(suffix='.vtk', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mesh = Mesh.from_file(tmp_path)
            assert mesh.polydata is mock_polydata
            mock_reader.assert_called_once_with(tmp_path)
            mock_filter.assert_called_once()
        finally:
            os.unlink(tmp_path)

    def test_from_file_invalid_path_type(self):
        """Test that ValueError is raised for invalid path types."""
        with pytest.raises(ValueError, match="mesh_path must be a non-empty string"):
            Mesh.from_file(None)

        with pytest.raises(ValueError, match="mesh_path must be a non-empty string"):
            Mesh.from_file("")

        with pytest.raises(ValueError, match="mesh_path must be a non-empty string"):
            Mesh.from_file(123)

    def test_from_file_nonexistent_file(self):
        """Test that FileNotFoundError is raised for non-existent files."""
        with pytest.raises(FileNotFoundError, match="Mesh file not found"):
            Mesh.from_file("/nonexistent/path/to/file.vtk")

    @patch('Atrial_LDRBM.Generate_Boundaries.mesh.smart_reader')
    @patch('Atrial_LDRBM.Generate_Boundaries.mesh.apply_vtk_geom_filter')
    def test_from_file_empty_mesh(self, mock_filter, mock_reader):
        """Test that ValueError is raised when loaded mesh is empty."""
        mock_polydata = Mock(spec=vtk.vtkPolyData)
        mock_polydata.GetNumberOfPoints.return_value = 0

        mock_reader.return_value = Mock()
        mock_filter.return_value = mock_polydata

        with tempfile.NamedTemporaryFile(suffix='.vtk', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with pytest.raises(ValueError, match="is empty or invalid"):
                Mesh.from_file(tmp_path)
        finally:
            os.unlink(tmp_path)

    @patch('Atrial_LDRBM.Generate_Boundaries.mesh.smart_reader')
    @patch('Atrial_LDRBM.Generate_Boundaries.mesh.apply_vtk_geom_filter')
    def test_from_file_none_polydata(self, mock_filter, mock_reader):
        """Test that ValueError is raised when filter returns None."""
        mock_reader.return_value = Mock()
        mock_filter.return_value = None

        with tempfile.NamedTemporaryFile(suffix='.vtk', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with pytest.raises(ValueError, match="is empty or invalid"):
                Mesh.from_file(tmp_path)
        finally:
            os.unlink(tmp_path)


class TestMeshSave:
    """Test the Mesh.save method."""

    def create_mock_mesh(self, num_points=10, use_unstructured_grid=False):
        """Helper to create a mock mesh with polydata or unstructured grid."""
        if use_unstructured_grid:
            mock_data = Mock(spec=vtk.vtkUnstructuredGrid)
        else:
            mock_data = Mock(spec=vtk.vtkPolyData)
        mock_data.GetNumberOfPoints.return_value = num_points
        return Mesh(mock_data)

    @patch('Atrial_LDRBM.Generate_Boundaries.mesh.vtk_polydata_writer')
    def test_save_vtk_format(self, mock_writer):
        """Test saving mesh in .vtk format."""
        mesh = self.create_mock_mesh()

        with tempfile.NamedTemporaryFile(suffix='.vtk', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mesh.save(tmp_path)
            mock_writer.assert_called_once_with(tmp_path, mesh.polydata, store_xml=False)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @patch('Atrial_LDRBM.Generate_Boundaries.mesh.vtk_polydata_writer')
    def test_save_vtp_format(self, mock_writer):
        """Test saving mesh in .vtp format."""
        mesh = self.create_mock_mesh()

        with tempfile.NamedTemporaryFile(suffix='.vtp', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mesh.save(tmp_path)
            mock_writer.assert_called_once_with(tmp_path, mesh.polydata, store_xml=True)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @patch('Atrial_LDRBM.Generate_Boundaries.mesh.vtk_polydata_writer')
    def test_save_vtk_with_xml_format(self, mock_writer):
        """Test saving .vtk with xml_format=True."""
        mesh = self.create_mock_mesh()

        with tempfile.NamedTemporaryFile(suffix='.vtk', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mesh.save(tmp_path, xml_format=True)
            mock_writer.assert_called_once_with(tmp_path, mesh.polydata, store_xml=True)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @patch('Atrial_LDRBM.Generate_Boundaries.mesh.vtk_obj_writer')
    def test_save_obj_format(self, mock_writer):
        """Test saving mesh in .obj format."""
        mesh = self.create_mock_mesh()

        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mesh.save(tmp_path)
            mock_writer.assert_called_once_with(tmp_path, mesh.polydata)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @patch('Atrial_LDRBM.Generate_Boundaries.mesh.vtk_polydata_writer')
    def test_save_unstructured_grid_converts_to_polydata(self, mock_writer):
        """Test that vtkUnstructuredGrid is converted to vtkPolyData before saving."""
        # Create a real unstructured grid to test isinstance check
        ug = vtk.vtkUnstructuredGrid()
        # Add a point so it's not empty
        points = vtk.vtkPoints()
        points.InsertNextPoint(0, 0, 0)
        ug.SetPoints(points)

        mesh = Mesh(ug)

        with tempfile.NamedTemporaryFile(suffix='.vtk', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mesh.save(tmp_path)

            # Verify writer was called - the second argument should be vtkPolyData
            # (converted from vtkUnstructuredGrid)
            assert mock_writer.call_count == 1
            call_args = mock_writer.call_args[0]
            assert call_args[0] == tmp_path
            # The second argument should be a vtkPolyData (converted from vtkUnstructuredGrid)
            assert isinstance(call_args[1], vtk.vtkPolyData)
            assert mock_writer.call_args[1] == {'store_xml': False}
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_save_unsupported_format(self):
        """Test that ValueError is raised for unsupported file formats."""
        mesh = self.create_mock_mesh()

        with pytest.raises(ValueError, match="Unsupported file format '.stl'"):
            mesh.save("output.stl")

        with pytest.raises(ValueError, match="Unsupported file format '.ply'"):
            mesh.save("output.ply")

    def test_save_empty_mesh(self):
        """Test that ValueError is raised when trying to save an empty mesh."""
        mesh = self.create_mock_mesh(num_points=0)

        with pytest.raises(ValueError, match="Cannot save an empty mesh"):
            mesh.save("output.vtk")

    def test_save_invalid_file_path(self):
        """Test that ValueError is raised for invalid file paths."""
        mesh = self.create_mock_mesh()

        with pytest.raises(ValueError, match="file_path must be a non-empty string"):
            mesh.save("")

        with pytest.raises(ValueError, match="file_path must be a non-empty string"):
            mesh.save(None)

    @patch('Atrial_LDRBM.Generate_Boundaries.mesh.vtk_polydata_writer')
    def test_save_uppercase_extension(self, mock_writer):
        """Test that uppercase extensions are handled correctly."""
        mesh = self.create_mock_mesh()

        with tempfile.NamedTemporaryFile(suffix='.VTK', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mesh.save(tmp_path)
            mock_writer.assert_called_once_with(tmp_path, mesh.polydata, store_xml=False)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestMeshGetPolydata:
    """Test the Mesh.get_polydata method."""

    def test_get_polydata_returns_correct_polydata(self):
        """Test that get_polydata returns the encapsulated polydata."""
        polydata = vtk.vtkPolyData()
        mesh = Mesh(polydata)

        retrieved = mesh.get_polydata()
        assert retrieved is polydata

    def test_get_polydata_returns_correct_unstructured_grid(self):
        """Test that get_polydata returns the encapsulated unstructured grid."""
        ug = vtk.vtkUnstructuredGrid()
        mesh = Mesh(ug)

        retrieved = mesh.get_polydata()
        assert retrieved is ug

    def test_get_polydata_returns_vtk_polydata_type(self):
        """Test that get_polydata returns a vtkPolyData instance."""
        polydata = vtk.vtkPolyData()
        mesh = Mesh(polydata)

        retrieved = mesh.get_polydata()
        assert isinstance(retrieved, vtk.vtkPolyData)

    def test_get_polydata_returns_vtk_unstructured_grid_type(self):
        """Test that get_polydata returns a vtkUnstructuredGrid instance."""
        ug = vtk.vtkUnstructuredGrid()
        mesh = Mesh(ug)

        retrieved = mesh.get_polydata()
        assert isinstance(retrieved, vtk.vtkUnstructuredGrid)