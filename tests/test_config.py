from pathlib import Path
from typing import Optional


class TestPaths:
    """
    Manages paths for test fixtures (inputs) and golden results (expected outputs).

    This is intentionally simple and stateless - it just builds paths based on
    naming conventions, unlike the stateful WorkflowPaths used in production.
    """

    def __init__(self, tests_root: Optional[Path] = None):
        """
        Initialize test paths.

        Args:
            tests_root: Root directory of tests. If None, auto-detects from this file's location.
        """
        if tests_root is None:
            tests_root = Path(__file__).parent.resolve()

        self.root = tests_root
        self.test_data = tests_root / "test_data"
        self.golden_results = tests_root / "good_results"

    def input_mesh(self, basename: str, extension: str = ".vtp") -> Path:
        """
        Path to input mesh file.
        Args:
            basename: Mesh base name without extension (e.g., "LA_MRI")
            extension: File extension including dot (e.g., ".vtp", ".vtk")
        Returns: Path to mesh file in test_data directory
        """
        return self.test_data / f"{basename}{extension}"

    def orifice_coordinates_file(self, basename: str) -> Path:
        """
        Path to orifice coordinates CSV file.
        Args: basename: Mesh base name (e.g., "LA_MRI")
        Returns: Path to orifice CSV in test_data directory
        """
        return self.test_data / f"{basename}_orifices.csv"

    def apex_coordinates_file(self, basename: str) -> Path:
        """
        Path to apex coordinates CSV file.
        Args: basename: Mesh base name (e.g., "LA_MRI")
        Returns: Path to apex CSV in test_data directory
        """
        return self.test_data / f"{basename}_apexes.csv"

    def golden_surf_dir(self, output_name: str) -> Path:
        """
        Path to expected output surf directory.
        Args: output_name: Name of the surf directory (e.g., "LA_cutted_res_surf")
        Returns: Path to surf directory in golden_results
        """
        return self.golden_results / output_name

    def validate_test_data_exists(self,
                                  basename: str,
                                  require_orifice: bool = False) -> None:
        """
        Validate that all required test input files exist.
        Args:
            basename: Mesh base name
            require_orifice: Whether orifice file is required
        Raises:
            FileNotFoundError: If any required file is missing
        """
        mesh = self.input_mesh(basename)
        if not mesh.exists():
            raise FileNotFoundError(f"Test mesh not found: {mesh}")

        apex = self.apex_coordinates_file(basename)
        if not apex.exists():
            raise FileNotFoundError(f"Apex coordinates not found: {apex}")

        if require_orifice:
            orifice = self.orifice_coordinates_file(basename)
            if not orifice.exists():
                raise FileNotFoundError(f"Orifice coordinates not found: {orifice}")

    def validate_golden_results_exist(self, surf_dir_name: str) -> None:
        """
        Validate that golden results exist for comparison.
        Args: surf_dir_name: Name of expected surf directory
        Raises: FileNotFoundError: If golden results are missing
        """
        golden_dir = self.golden_surf_dir(surf_dir_name)

        if not golden_dir.exists():
            raise FileNotFoundError(
                f"Golden results not found: {golden_dir}\n"
                f"Run pipeline with --save-golden-results to generate them."
            )
