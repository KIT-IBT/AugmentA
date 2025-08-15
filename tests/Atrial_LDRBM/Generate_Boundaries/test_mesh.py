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

    def test_init_with_invalid_input(self):
        """Test that TypeError is raised for non-vtkPolyData input."""
        with pytest.raises(TypeError, match="Input must be a vtk.vtkPolyData object"):
            Mesh("not_polydata")

        with pytest.raises(TypeError, match="Input must be a vtk.vtkPolyData object"):
            Mesh(None)

        with pytest.raises(TypeError, match="Input must be a vtk.vtkPolyData object"):
            Mesh(123)


