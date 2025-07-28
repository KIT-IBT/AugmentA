import os
import vtk
from typing import Dict, Any

from vtk_openCARP_methods_ibt.vtk_methods.reader import smart_reader
from vtk_openCARP_methods_ibt.vtk_methods.thresholding import get_threshold_between
from vtk_openCARP_methods_ibt.vtk_methods.filters import apply_vtk_geom_filter

from Atrial_LDRBM.Generate_Boundaries.mesh import Mesh


class EpiEndoSeparator:
    """
    Separate epicardial and endocardial surfaces from a tagged volume mesh.
    """

    def __init__(self, element_tags: Dict[str, str], atrium: str) -> None:
        """
        Initialize with element tags and atrium side.

        :param element_tags: Dictionary mapping tag names to tag values (strings)
        :param atrium: 'LA' or 'RA'
        """
        if atrium not in ("LA", "RA"):
            raise ValueError("Atrium must be 'LA' or 'RA'")
        self.atrium = atrium

        # Determine tag keys based on atrium
        side = "left" if atrium == "LA" else "right"
        key_epi = f"{side}_atrial_wall_epi"
        key_endo = f"{side}_atrial_wall_endo"

        try:
            self.epi_tag = int(element_tags[key_epi])
            self.endo_tag = int(element_tags[key_endo])
        except KeyError as e:
            raise KeyError(f"Missing tag in element_tags: {e}")
        except ValueError:
            raise ValueError("Tag values must be integers")

    def separate(self, volume_mesh: Mesh) -> Dict[str, Mesh]:
        """
        Perform epi/endo separation on the provided volume mesh.

        :param volume_mesh: Mesh wrapping a vtkUnstructuredGrid with 'tag' cell data
        :return: Dict with keys 'combined', 'epi', 'endo' mapping to surface Meshes
        """
        # Extract the underlying grid
        ug = volume_mesh.get_polydata()

        # Threshold to get combined wall (endocardium through epicardium)
        combined_pd = get_threshold_between(
            ug,
            self.endo_tag,
            self.epi_tag,
            "vtkDataObject::FIELD_ASSOCIATION_CELLS",
            "tag"
        ).GetOutput()

        # Threshold epicardium only
        epi_pd = get_threshold_between(
            ug,
            self.epi_tag,
            self.epi_tag,
            "vtkDataObject::FIELD_ASSOCIATION_CELLS",
            "tag"
        ).GetOutput()

        # Threshold endocardium only
        endo_pd = get_threshold_between(
            ug,
            self.endo_tag,
            self.endo_tag,
            "vtkDataObject::FIELD_ASSOCIATION_CELLS",
            "tag"
        ).GetOutput()

        # Apply geometry filter to generate surfaces
        combined_surf = apply_vtk_geom_filter(combined_pd)
        epi_surf = apply_vtk_geom_filter(epi_pd)
        endo_surf = apply_vtk_geom_filter(endo_pd)

        # Wrap in Mesh objects and return
        return {
            "combined": Mesh(combined_surf),
            "epi": Mesh(epi_surf),
            "endo": Mesh(endo_surf),
        }
