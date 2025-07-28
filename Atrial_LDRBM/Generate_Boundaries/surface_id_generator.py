import os
import shutil
import numpy as np
from glob import glob
import vtk
from scipy.spatial import cKDTree

from vtk_openCARP_methods_ibt.vtk_methods.exporting import write_to_vtx
from vtk_openCARP_methods_ibt.vtk_methods.converters import vtk_to_numpy
from vtk_openCARP_methods_ibt.vtk_methods.reader import vtx_reader
from vtk_openCARP_methods_ibt.vtk_methods.filters import generate_ids

from Atrial_LDRBM.Generate_Boundaries.mesh import Mesh


class SurfaceIdMapper:
    """
    Map points from a surface mesh to a reference volume mesh using KDTree.
    """
    def __init__(self, volume_mesh: Mesh) -> None:
        """
        Initialize the mapper with a volume mesh.

        :param volume_mesh: Mesh object wrapping the reference volume mesh
        """
        vol_polydata = volume_mesh.get_polydata()
        if vol_polydata is None or vol_polydata.GetNumberOfPoints() == 0:
            raise ValueError("Input volume_mesh cannot be empty.")

        # Convert VTK points to numpy array for KDTree
        vol_coords = vtk_to_numpy(vol_polydata.GetPoints().GetData())
        if vol_coords.size == 0:
            raise ValueError("Volume mesh has no coordinates.")

        self.kdtree = cKDTree(vol_coords)

    def map_surface(self, surface_mesh: Mesh) -> np.ndarray:
        """
        Map each point in the surface mesh to the closest volume mesh point.

        :param surface_mesh: Mesh object wrapping the surface mesh
        :return:             Array of volume-mesh point indices for each surface point
        """
        surf_polydata = surface_mesh.get_polydata()
        if surf_polydata is None or surf_polydata.GetNumberOfPoints() == 0:
            # No points to map
            return np.array([], dtype=int)

        # Convert surface points to numpy
        surf_coords = vtk_to_numpy(surf_polydata.GetPoints().GetData())
        if surf_coords.size == 0:
            return np.array([], dtype=int)

        # Query KDTree for nearest neighbor indices
        _, indices = self.kdtree.query(surf_coords)
        return indices
