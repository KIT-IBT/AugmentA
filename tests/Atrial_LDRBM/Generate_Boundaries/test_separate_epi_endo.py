import pytest
import vtk
from unittest.mock import Mock, patch, MagicMock
from Atrial_LDRBM.Generate_Boundaries.epi_endo_separator import EpiEndoSeparator
from Atrial_LDRBM.Generate_Boundaries.mesh import Mesh


class TestEpiEndoSeparatorInit:
    """Test the EpiEndoSeparator constructor."""

    def test_init_valid_LA(self):
        """Test successful initialization with valid LA tags."""
        element_tags = {
            'left_atrial_wall_epi': '10',
            'left_atrial_wall_endo': '20'
        }
        separator = EpiEndoSeparator(element_tags, 'LA')

        assert separator.atrium == 'LA'
        assert separator.epi_tag == 10
        assert separator.endo_tag == 20

    def test_init_valid_RA(self):
        """Test successful initialization with valid RA tags."""
        element_tags = {
            'right_atrial_wall_epi': '30',
            'right_atrial_wall_endo': '40'
        }
        separator = EpiEndoSeparator(element_tags, 'RA')

        assert separator.atrium == 'RA'
        assert separator.epi_tag == 30
        assert separator.endo_tag == 40

    def test_init_invalid_atrium(self):
        """Test that ValueError is raised for invalid atrium."""
        element_tags = {
            'left_atrial_wall_epi': '10',
            'left_atrial_wall_endo': '20'
        }

        with pytest.raises(ValueError, match="Atrium must be 'LA' or 'RA'"):
            EpiEndoSeparator(element_tags, 'XX')

        with pytest.raises(ValueError, match="Atrium must be 'LA' or 'RA'"):
            EpiEndoSeparator(element_tags, 'LV')

    def test_init_missing_epi_tag_LA(self):
        """Test that KeyError is raised when LA epi tag is missing."""
        element_tags = {
            'left_atrial_wall_endo': '20'
        }

        with pytest.raises(KeyError, match="Missing tag in element_tags"):
            EpiEndoSeparator(element_tags, 'LA')

    def test_init_missing_endo_tag_LA(self):
        """Test that KeyError is raised when LA endo tag is missing."""
        element_tags = {
            'left_atrial_wall_epi': '10'
        }

        with pytest.raises(KeyError, match="Missing tag in element_tags"):
            EpiEndoSeparator(element_tags, 'LA')

    def test_init_missing_epi_tag_RA(self):
        """Test that KeyError is raised when RA epi tag is missing."""
        element_tags = {
            'right_atrial_wall_endo': '40'
        }

        with pytest.raises(KeyError, match="Missing tag in element_tags"):
            EpiEndoSeparator(element_tags, 'RA')

    def test_init_missing_endo_tag_RA(self):
        """Test that KeyError is raised when RA endo tag is missing."""
        element_tags = {
            'right_atrial_wall_epi': '30'
        }

        with pytest.raises(KeyError, match="Missing tag in element_tags"):
            EpiEndoSeparator(element_tags, 'RA')

    def test_init_non_integer_epi_tag(self):
        """Test that ValueError is raised when epi tag is not an integer."""
        element_tags = {
            'left_atrial_wall_epi': 'not_an_int',
            'left_atrial_wall_endo': '20'
        }

        with pytest.raises(ValueError, match="Tag values must be integers"):
            EpiEndoSeparator(element_tags, 'LA')

    def test_init_non_integer_endo_tag(self):
        """Test that ValueError is raised when endo tag is not an integer."""
        element_tags = {
            'left_atrial_wall_epi': '10',
            'left_atrial_wall_endo': 'not_an_int'
        }

        with pytest.raises(ValueError, match="Tag values must be integers"):
            EpiEndoSeparator(element_tags, 'LA')

    def test_init_empty_tags_dict(self):
        """Test that KeyError is raised when tags dictionary is empty."""
        element_tags = {}

        with pytest.raises(KeyError, match="Missing tag in element_tags"):
            EpiEndoSeparator(element_tags, 'LA')


class TestEpiEndoSeparatorSeparate:
    """Test the EpiEndoSeparator.separate method."""

    def create_mock_mesh(self):
        """Helper to create a mock Mesh with vtkUnstructuredGrid."""
        mock_ug = Mock(spec=vtk.vtkUnstructuredGrid)
        mock_mesh = Mock(spec=Mesh)
        mock_mesh.get_polydata.return_value = mock_ug
        return mock_mesh, mock_ug

    def create_mock_threshold_output(self):
        """Helper to create a mock threshold filter with output."""
        mock_threshold = Mock()
        mock_output = Mock(spec=vtk.vtkUnstructuredGrid)
        mock_threshold.GetOutput.return_value = mock_output
        return mock_threshold, mock_output

    @patch('Atrial_LDRBM.Generate_Boundaries.epi_endo_separator.get_threshold_between')
    @patch('Atrial_LDRBM.Generate_Boundaries.epi_endo_separator.apply_vtk_geom_filter')
    @patch('Atrial_LDRBM.Generate_Boundaries.epi_endo_separator.Mesh')
    def test_separate_success_LA(self, mock_mesh_class, mock_geom_filter, mock_threshold):
        """Test successful separation for LA."""
        element_tags = {
            'left_atrial_wall_epi': '10',
            'left_atrial_wall_endo': '20'
        }
        separator = EpiEndoSeparator(element_tags, 'LA')

        # Create mock input mesh
        mock_mesh, mock_ug = self.create_mock_mesh()

        # Create mock threshold outputs
        mock_combined_thresh, mock_combined_output = self.create_mock_threshold_output()
        mock_epi_thresh, mock_epi_output = self.create_mock_threshold_output()
        mock_endo_thresh, mock_endo_output = self.create_mock_threshold_output()

        # Setup threshold mock to return different outputs for different calls
        mock_threshold.side_effect = [mock_combined_thresh, mock_epi_thresh, mock_endo_thresh]

        # Create mock surface outputs from geometry filter
        mock_combined_surf = Mock(spec=vtk.vtkPolyData)
        mock_epi_surf = Mock(spec=vtk.vtkPolyData)
        mock_endo_surf = Mock(spec=vtk.vtkPolyData)
        mock_geom_filter.side_effect = [mock_combined_surf, mock_epi_surf, mock_endo_surf]

        # Create mock Mesh objects
        mock_combined_mesh = Mock()
        mock_epi_mesh = Mock()
        mock_endo_mesh = Mock()
        mock_mesh_class.side_effect = [mock_combined_mesh, mock_epi_mesh, mock_endo_mesh]

        # Call separate
        result = separator.separate(mock_mesh)

        # Verify the result dictionary
        assert 'combined' in result
        assert 'epi' in result
        assert 'endo' in result
        assert result['combined'] == mock_combined_mesh
        assert result['epi'] == mock_epi_mesh
        assert result['endo'] == mock_endo_mesh

        # Verify get_polydata was called
        mock_mesh.get_polydata.assert_called_once()

        # Verify threshold was called three times with correct parameters
        assert mock_threshold.call_count == 3

        # Check combined threshold call (endo_tag to epi_tag)
        mock_threshold.assert_any_call(
            mock_ug,
            20,  # endo_tag
            10,  # epi_tag
            "vtkDataObject::FIELD_ASSOCIATION_CELLS",
            "tag"
        )

        # Check epi threshold call (epi_tag to epi_tag)
        mock_threshold.assert_any_call(
            mock_ug,
            10,  # epi_tag
            10,  # epi_tag
            "vtkDataObject::FIELD_ASSOCIATION_CELLS",
            "tag"
        )

        # Check endo threshold call (endo_tag to endo_tag)
        mock_threshold.assert_any_call(
            mock_ug,
            20,  # endo_tag
            20,  # endo_tag
            "vtkDataObject::FIELD_ASSOCIATION_CELLS",
            "tag"
        )

        # Verify geometry filter was called three times
        assert mock_geom_filter.call_count == 3
        mock_geom_filter.assert_any_call(mock_combined_output)
        mock_geom_filter.assert_any_call(mock_epi_output)
        mock_geom_filter.assert_any_call(mock_endo_output)

        # Verify Mesh constructor was called three times
        assert mock_mesh_class.call_count == 3
        mock_mesh_class.assert_any_call(mock_combined_surf)
        mock_mesh_class.assert_any_call(mock_epi_surf)
        mock_mesh_class.assert_any_call(mock_endo_surf)

    @patch('Atrial_LDRBM.Generate_Boundaries.epi_endo_separator.get_threshold_between')
    @patch('Atrial_LDRBM.Generate_Boundaries.epi_endo_separator.apply_vtk_geom_filter')
    @patch('Atrial_LDRBM.Generate_Boundaries.epi_endo_separator.Mesh')
    def test_separate_success_RA(self, mock_mesh_class, mock_geom_filter, mock_threshold):
        """Test successful separation for RA."""
        element_tags = {
            'right_atrial_wall_epi': '30',
            'right_atrial_wall_endo': '40'
        }
        separator = EpiEndoSeparator(element_tags, 'RA')

        # Create mock input mesh
        mock_mesh, mock_ug = self.create_mock_mesh()

        # Create mock threshold outputs
        mock_combined_thresh, mock_combined_output = self.create_mock_threshold_output()
        mock_epi_thresh, mock_epi_output = self.create_mock_threshold_output()
        mock_endo_thresh, mock_endo_output = self.create_mock_threshold_output()

        mock_threshold.side_effect = [mock_combined_thresh, mock_epi_thresh, mock_endo_thresh]

        # Create mock surface outputs
        mock_combined_surf = Mock(spec=vtk.vtkPolyData)
        mock_epi_surf = Mock(spec=vtk.vtkPolyData)
        mock_endo_surf = Mock(spec=vtk.vtkPolyData)
        mock_geom_filter.side_effect = [mock_combined_surf, mock_epi_surf, mock_endo_surf]

        # Create mock Mesh objects
        mock_combined_mesh = Mock()
        mock_epi_mesh = Mock()
        mock_endo_mesh = Mock()
        mock_mesh_class.side_effect = [mock_combined_mesh, mock_epi_mesh, mock_endo_mesh]

        # Call separate
        result = separator.separate(mock_mesh)

        # Verify the result
        assert 'combined' in result
        assert 'epi' in result
        assert 'endo' in result

        # Verify threshold was called with correct RA tags
        mock_threshold.assert_any_call(
            mock_ug,
            40,  # endo_tag
            30,  # epi_tag
            "vtkDataObject::FIELD_ASSOCIATION_CELLS",
            "tag"
        )

        mock_threshold.assert_any_call(
            mock_ug,
            30,  # epi_tag
            30,  # epi_tag
            "vtkDataObject::FIELD_ASSOCIATION_CELLS",
            "tag"
        )

        mock_threshold.assert_any_call(
            mock_ug,
            40,  # endo_tag
            40,  # endo_tag
            "vtkDataObject::FIELD_ASSOCIATION_CELLS",
            "tag"
        )

    @patch('Atrial_LDRBM.Generate_Boundaries.epi_endo_separator.get_threshold_between')
    def test_separate_threshold_returns_none(self, mock_threshold):
        """Test behavior when threshold operation returns None."""
        element_tags = {
            'left_atrial_wall_epi': '10',
            'left_atrial_wall_endo': '20'
        }
        separator = EpiEndoSeparator(element_tags, 'LA')

        # Create mock input mesh
        mock_mesh, mock_ug = self.create_mock_mesh()

        # Make threshold return None
        mock_threshold.return_value = None

        # This should raise an AttributeError when trying to call GetOutput() on None
        with pytest.raises(AttributeError):
            separator.separate(mock_mesh)

    def test_separate_with_invalid_mesh_type(self):
        """Test that error is raised when mesh is not a Mesh object."""
        element_tags = {
            'left_atrial_wall_epi': '10',
            'left_atrial_wall_endo': '20'
        }
        separator = EpiEndoSeparator(element_tags, 'LA')

        # Pass something that's not a Mesh
        with pytest.raises(AttributeError):
            separator.separate("not_a_mesh")


class TestEpiEndoSeparatorIntegration:
    """Integration tests using real VTK objects (not mocked)."""

    def create_simple_unstructured_grid(self):
        """Create a simple vtkUnstructuredGrid for testing."""
        points = vtk.vtkPoints()
        points.InsertNextPoint(0, 0, 0)
        points.InsertNextPoint(1, 0, 0)
        points.InsertNextPoint(0, 1, 0)
        points.InsertNextPoint(0, 0, 1)

        ug = vtk.vtkUnstructuredGrid()
        ug.SetPoints(points)

        # Create a tetrahedron
        tetra = vtk.vtkTetra()
        tetra.GetPointIds().SetId(0, 0)
        tetra.GetPointIds().SetId(1, 1)
        tetra.GetPointIds().SetId(2, 2)
        tetra.GetPointIds().SetId(3, 3)

        ug.InsertNextCell(tetra.GetCellType(), tetra.GetPointIds())

        # Add tag cell data
        tags = vtk.vtkIntArray()
        tags.SetName("tag")
        tags.InsertNextValue(10)  # Epi tag
        ug.GetCellData().AddArray(tags)
        ug.GetCellData().SetActiveScalars("tag")

        return ug

    @patch('Atrial_LDRBM.Generate_Boundaries.epi_endo_separator.get_threshold_between')
    @patch('Atrial_LDRBM.Generate_Boundaries.epi_endo_separator.apply_vtk_geom_filter')
    def test_separate_integration(self, mock_geom_filter, mock_threshold):
        """Test separate method with realistic VTK objects."""
        element_tags = {
            'left_atrial_wall_epi': '10',
            'left_atrial_wall_endo': '20'
        }
        separator = EpiEndoSeparator(element_tags, 'LA')

        # Create a real unstructured grid
        ug = self.create_simple_unstructured_grid()
        mesh = Mesh(ug)

        # Mock the threshold and filter operations
        mock_threshold_result = Mock()
        mock_threshold_result.GetOutput.return_value = vtk.vtkPolyData()
        mock_threshold.return_value = mock_threshold_result

        mock_geom_filter.return_value = vtk.vtkPolyData()

        # Call separate
        result = separator.separate(mesh)

        # Verify result structure
        assert isinstance(result, dict)
        assert 'combined' in result
        assert 'epi' in result
        assert 'endo' in result
        assert all(isinstance(m, Mesh) for m in result.values())