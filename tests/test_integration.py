import os
import shutil
import filecmp
from pathlib import Path
import pytest
import pandas as pd
import pyvista as pv
import numpy as np
from scipy.spatial import cKDTree

from pipeline import AugmentA
from main import parser
from test_config import TestPaths

# Test configuration
COMPARE_WITH_TOLERANCE = False  # Set False to require exact matches for all files
COMPARE_BY_COORDINATES = True  # For VTX files, compare by coordinates instead of IDs

# Tolerance configuration - adjust these values based on acceptable variation
VTK_DISTANCE_THRESHOLD = 2e3  # micrometers (2mm) - for geometry comparison
CSV_RTOL = 1e-4  # 0.01% relative tolerance for CSV numerical values
CSV_ATOL = 2e3  # micrometers absolute tolerance for CSV coordinates
MESH_COUNT_TOLERANCE_PERCENT = 10.0  # Allow 10% variation in point/element counts
VTX_COORDINATE_TOLERANCE = 1000.0  # micrometers (1mm) for VTX coordinate matching

# Test mesh configuration
TEST_MESH_BASENAME = "LA_MRI"
EXPECTED_OUTPUT_SURF_DIR = "LA_cutted_res_surf"


def print_file_sizes_on_failure(current_path: Path, golden_path: Path):
    """Print file sizes for debugging when comparison fails."""
    try:
        current_size = current_path.stat().st_size
        golden_size = golden_path.stat().st_size
        print(f"File sizes: current={current_size} bytes, golden={golden_size} bytes")
    except Exception:
        pass


def read_vtx_to_set(filepath: Path) -> set:
    """Read VTX file and return set of IDs for order-independent comparison."""
    if not filepath.exists():
        return set()

    result_set = set()
    with open(filepath, 'r') as file_handle:
        for line in file_handle:
            stripped_line = line.strip()
            if stripped_line.isdigit():
                result_set.add(int(stripped_line))
    return result_set


def compare_vtx_with_tolerance(current_path: Path, golden_path: Path, max_diff_percent: float = 5.0) -> bool:
    """
    Compare VTX files allowing for a percentage of different IDs.
    """
    current_set = read_vtx_to_set(current_path)
    golden_set = read_vtx_to_set(golden_path)

    total_unique_ids = len(current_set.union(golden_set))
    different_ids = len(current_set.symmetric_difference(golden_set))

    if total_unique_ids == 0:
        return True  # Both empty

    diff_percent = (different_ids / total_unique_ids) * 100

    print(f"VTX comparison for {current_path.name}:")
    print(f" - Golden: {len(golden_set)} IDs")
    print(f" - Current: {len(current_set)} IDs")
    print(f" - Different: {different_ids} IDs ({diff_percent:.1f}%)")

    return diff_percent <= max_diff_percent


def find_mesh_file_for_vtx_comparison(result_dir: Path, temp_input_dir: Path) -> Path:
    """
    Find the appropriate mesh file for VTX coordinate comparison.

    Args:
        result_dir: The surf directory containing VTX files
        temp_input_dir: The input directory containing the pipeline meshes

    Returns:
        Path to the mesh file that the VTX files reference
    """
    # Look for common mesh file names in the surf directory first
    possible_mesh_files = ["LA.vtk", "LA.vtp", "LA.obj", "LA.ply"]

    for mesh_name in possible_mesh_files:
        mesh_path = result_dir / mesh_name
        if mesh_path.exists():
            return mesh_path

    # If not found in surf directory, look in the parent input directory
    possible_parent_meshes = [
        "LA_cutted_res.ply",
        "LA_cutted_res.vtk",
        "LA_cutted_res.obj",
        "LA_cutted.vtk",
        "LA_cutted.obj"
    ]

    for mesh_name in possible_parent_meshes:
        mesh_path = temp_input_dir / mesh_name
        if mesh_path.exists():
            return mesh_path

    # Last resort: look for any mesh file in the result directory
    for file_path in result_dir.iterdir():
        if file_path.is_file() and file_path.suffix in ['.vtk', '.vtp', '.obj', '.ply']:
            return file_path

    raise FileNotFoundError(f"Could not find mesh file for VTX comparison in {result_dir} or {temp_input_dir}")


def compare_vtx_by_coordinates(current_vtx_path: Path,
                               golden_vtx_path: Path,
                               current_mesh_path: Path,
                               golden_mesh_path: Path,
                               tolerance: float = VTX_COORDINATE_TOLERANCE) -> bool:
    """
    Compare VTX files by looking up point coordinates in their respective meshes.

    Args:
        current_vtx_path: Path to current VTX file
        golden_vtx_path: Path to golden VTX file
        current_mesh_path: Path to current mesh file
        golden_mesh_path: Path to golden mesh file
        tolerance: Distance tolerance in micrometers

    Returns:
        True if at least 95% of points match within tolerance
    """
    try:
        current_mesh = pv.read(current_mesh_path)
        golden_mesh = pv.read(golden_mesh_path)
    except Exception as e:
        print(f"Error reading mesh files: {e}")
        print(f"Current mesh: {current_mesh_path}")
        print(f"Golden mesh: {golden_mesh_path}")
        return False

    current_ids = read_vtx_to_set(current_vtx_path)
    golden_ids = read_vtx_to_set(golden_vtx_path)

    print(f"VTX comparison for {current_vtx_path.name}:")
    print(f"Current IDs: {len(current_ids)}, Golden IDs: {len(golden_ids)}")

    # Handle empty VTX files
    if not current_ids and not golden_ids:
        print("Both files empty - match")
        return True
    if not current_ids or not golden_ids:
        print("One file empty, other not - no match")
        return False

    try:
        current_coords = [current_mesh.points[i] for i in current_ids if i < len(current_mesh.points)]
        golden_coords = [golden_mesh.points[i] for i in golden_ids if i < len(golden_mesh.points)]

        if not golden_coords or not current_coords:
            result = len(current_coords) == len(golden_coords)
            print(f"No valid coordinates - match: {result}")
            return result

        tree = cKDTree(golden_coords)
        distances, _ = tree.query(current_coords, k=1)

        max_distance = np.max(distances)
        mean_distance = np.mean(distances)
        num_within_tolerance = np.sum(distances <= tolerance)

        print(f"Coordinate comparison:")
        print(f"Max distance: {max_distance:.6f} mm")
        print(f"Mean distance: {mean_distance:.6f} mm")
        print(f"Points within {tolerance} µm: {num_within_tolerance}/{len(distances)}")

        # Require 95% of points to be within tolerance
        match_percentage = num_within_tolerance / len(distances)
        required_match = 0.95

        result = match_percentage >= required_match

        if not result:
            print(f"FAIL: Only {match_percentage:.1%} of points matched (required: {required_match:.1%})")
            print(f"This suggests the apex point may have been at the wrong location during processing")
        else:
            print(f"PASS: {match_percentage:.1%} of points matched")

        return result

    except Exception as e:
        print(f"Error during coordinate comparison: {e}")
        return False


def compare_vtk_meshes_by_geometry(current_path: Path,
                                   golden_path: Path,
                                   distance_threshold: float = VTK_DISTANCE_THRESHOLD) -> bool:
    """
    Compare VTK meshes allowing for different point counts, by checking if geometries are similar.
    """
    try:
        current_mesh = pv.read(current_path)
        golden_mesh = pv.read(golden_path)

        print(f"\nVTK geometry comparison for {current_path.name}:")
        print(f"Points: current={current_mesh.n_points}, golden={golden_mesh.n_points}")

        # Build KD-tree for nearest neighbor search
        tree = cKDTree(golden_mesh.points)

        # For each current point, find distance to nearest golden point
        distances, _ = tree.query(current_mesh.points, k=1)

        max_dist = np.max(distances)
        mean_dist = np.mean(distances)

        print(f"Max distance to golden mesh: {max_dist:.2f} µm ({max_dist / 1000:.3f} mm)")
        print(f"Mean distance to golden mesh: {mean_dist:.2f} µm ({mean_dist / 1000:.3f} mm)")
        print(f"Threshold: {distance_threshold} µm ({distance_threshold / 1000} mm)")

        within_threshold = np.all(distances <= distance_threshold)

        if within_threshold:
            print(f"PASS: All points within threshold")
            return True
        else:
            num_exceeding = np.sum(distances > distance_threshold)
            print(f"FAIL: {num_exceeding}/{len(distances)} points exceed threshold")
            return False

    except Exception as e:
        print(f"Error comparing VTK meshes by geometry: {e}")
        return False


def compare_elem_files(current_path: Path,
                       golden_path: Path,
                       tolerance_percent: float = MESH_COUNT_TOLERANCE_PERCENT) -> bool:
    """Compare CARP .elem files allowing for some variation in element count."""
    try:
        with open(current_path, 'r') as f:
            current_lines = f.readlines()
        with open(golden_path, 'r') as f:
            golden_lines = f.readlines()

        current_count = int(current_lines[0].strip())
        golden_count = int(golden_lines[0].strip())

        print(f"\nElem file comparison for {current_path.name}:")
        print(f"Elements: current={current_count}, golden={golden_count}")

        diff_percent = abs(current_count - golden_count) / golden_count * 100
        print(f"Difference: {diff_percent:.1f}%")
        print(f"Tolerance: {tolerance_percent}%")

        if diff_percent <= tolerance_percent:
            print(f"PASS: Within tolerance")
            return True
        else:
            print(f"FAIL: Exceeds tolerance")
            return False

    except Exception as e:
        print(f"Error comparing elem files: {e}")
        return False


def compare_lon_files(current_path: Path,
                      golden_path: Path,
                      tolerance_percent: float = MESH_COUNT_TOLERANCE_PERCENT) -> bool:
    """Compare CARP .lon (fiber) files allowing for different vector counts."""
    try:
        with open(current_path, 'r') as f:
            current_first_line = f.readline().strip()
            current_count = int(current_first_line)

        with open(golden_path, 'r') as f:
            golden_first_line = f.readline().strip()
            golden_count = int(golden_first_line)

        print(f"\nLon file comparison for {current_path.name}:")
        print(f"Fiber vectors: current={current_count}, golden={golden_count}")

        diff_percent = abs(current_count - golden_count) / golden_count * 100
        print(f"Difference: {diff_percent:.1f}%")
        print(f"Tolerance: {tolerance_percent}%")

        if diff_percent <= tolerance_percent:
            print(f"PASS: Vector count within tolerance")
            return True
        else:
            print(f"FAIL: Vector count exceeds tolerance")
            return False

    except Exception as e:
        print(f"Error comparing lon files: {e}")
        return False


def compare_pts_files(current_path: Path,
                      golden_path: Path,
                      tolerance_percent: float = MESH_COUNT_TOLERANCE_PERCENT) -> bool:
    """Compare CARP .pts (points/coordinates) files allowing for different point counts."""
    try:
        with open(current_path, 'r') as f:
            current_count = int(f.readline().strip())

        with open(golden_path, 'r') as f:
            golden_count = int(f.readline().strip())

        print(f"\nPts file comparison for {current_path.name}:")
        print(f"  Points: current={current_count}, golden={golden_count}")

        diff_percent = abs(current_count - golden_count) / golden_count * 100
        print(f"  Difference: {diff_percent:.1f}%")
        print(f"  Tolerance: {tolerance_percent}%")

        if diff_percent <= tolerance_percent:
            print(f"  PASS: Point count within tolerance")
            return True
        else:
            print(f"  FAIL: Point count exceeds tolerance")
            return False

    except Exception as e:
        print(f"Error comparing pts files: {e}")
        return False


def compare_directories(current_dir: Path, golden_dir: Path, temp_input_dir: Path):
    """
    Compare two directories for structure and content with tolerance-based comparisons.
    """
    # Compare file structures
    golden_files = {path.relative_to(golden_dir) for path in golden_dir.rglob('*') if path.is_file()}
    current_files = {path.relative_to(current_dir) for path in current_dir.rglob('*') if path.is_file()}

    if golden_files != current_files:
        missing = golden_files - current_files
        extra = current_files - golden_files
        assert False, f"File mismatch. Missing: {missing}, Extra: {extra}"

    # Compare each file's content
    for golden_path in golden_dir.rglob('*'):
        if not golden_path.is_file():
            continue

        relative_path = golden_path.relative_to(golden_dir)
        current_path = current_dir / relative_path

        print(f"Comparing: {current_path} against {golden_path}")

        file_extension = golden_path.suffix
        file_name = golden_path.name

        try:
            if file_name == "rings_centroids.csv":
                golden_df = pd.read_csv(golden_path)
                current_df = pd.read_csv(current_path)

                try:
                    # Sort columns alphabetically
                    golden_df = golden_df.reindex(sorted(golden_df.columns), axis=1)
                    current_df = current_df.reindex(sorted(current_df.columns), axis=1)

                    pd.testing.assert_frame_equal(golden_df, current_df,
                                                  check_exact=False,
                                                  rtol=CSV_RTOL,
                                                  atol=CSV_ATOL)
                    print("  PASS: CSV content matches within tolerance")

                except AssertionError as e:
                    assert False, f"CSV content mismatch in {file_name}: {e}"

            elif file_extension == ".vtx":
                if COMPARE_WITH_TOLERANCE:
                    if not compare_vtx_with_tolerance(current_path, golden_path, max_diff_percent=5.0):
                        print_file_sizes_on_failure(current_path, golden_path)
                        assert False, f"VTX content differs too much in {file_name}"

                elif COMPARE_BY_COORDINATES:
                    try:
                        current_mesh_path = find_mesh_file_for_vtx_comparison(current_dir, temp_input_dir)
                        golden_mesh_path = find_mesh_file_for_vtx_comparison(golden_dir, golden_dir.parent)

                        print(f"Using meshes for VTX comparison:")
                        print(f"Current: {current_mesh_path}")
                        print(f"Golden: {golden_mesh_path}")

                        if not compare_vtx_by_coordinates(current_path, golden_path, current_mesh_path,
                                                          golden_mesh_path):
                            print_file_sizes_on_failure(current_path, golden_path)
                            assert False, f"VTX coordinate mismatch in {file_name}"

                    except FileNotFoundError as e:
                        print(f"Warning: Could not find mesh files for coordinate comparison: {e}")
                        print("Falling back to direct ID comparison")

                        golden_set = read_vtx_to_set(golden_path)
                        current_set = read_vtx_to_set(current_path)

                        if golden_set != current_set:
                            print_file_sizes_on_failure(current_path, golden_path)
                            assert False, f"VTX content mismatch in {file_name}"

                else:
                    # Direct ID comparison
                    golden_set = read_vtx_to_set(golden_path)
                    current_set = read_vtx_to_set(current_path)
                    if golden_set != current_set:
                        print_file_sizes_on_failure(current_path, golden_path)
                        assert False, f"VTX content mismatch in {file_name}"

            elif file_extension == ".pts":
                if not compare_pts_files(current_path, golden_path, tolerance_percent=10):
                    assert False, f"Point count mismatch in {file_name}"

            elif file_extension == ".lon":
                if not compare_lon_files(current_path, golden_path, tolerance_percent=10):
                    assert False, f"Fiber orientation mismatch in {file_name}"

            elif file_extension == ".elem":
                if not compare_elem_files(current_path, golden_path, tolerance_percent=10):
                    assert False, f"Element connectivity mismatch in {file_name}"

            elif file_extension == ".vtk":
                if not compare_vtk_meshes_by_geometry(current_path, golden_path,
                                                      distance_threshold=VTK_DISTANCE_THRESHOLD):
                    assert False, f"VTK mesh content mismatch in {file_name}"
            else:
                # Binary comparison for other files
                if not filecmp.cmp(golden_path, current_path, shallow=False):
                    print_file_sizes_on_failure(current_path, golden_path)
                    assert False, f"File content mismatch in {file_name}"
                print("PASS: Binary content matches")

        except AssertionError:
            raise
        except Exception as e:
            print(f"ERROR: Unexpected error comparing {file_name}: {e}")
            raise


def test_la_cut_resample_pipeline_regression(tmp_path: Path):
    """
    Test the full pipeline against golden results.
    """
    # Initialize test paths
    test_paths = TestPaths()

    # Validate test data exists
    test_paths.validate_test_data_exists(TEST_MESH_BASENAME, require_orifice=True)
    test_paths.validate_golden_results_exist(EXPECTED_OUTPUT_SURF_DIR)

    # Setup temporary test directory with copies of input files
    temp_input_dir = tmp_path / "input"
    temp_input_dir.mkdir()

    original_cwd = Path.cwd()

    try:
        # Copy input files to temp directory
        input_mesh_source = test_paths.input_mesh(TEST_MESH_BASENAME, ".vtp")
        apex_file_source = test_paths.apex_coordinates_file(TEST_MESH_BASENAME)
        orifice_file_source = test_paths.orifice_coordinates_file(TEST_MESH_BASENAME)

        shutil.copy(input_mesh_source, temp_input_dir)
        shutil.copy(apex_file_source, temp_input_dir)
        shutil.copy(orifice_file_source, temp_input_dir)

        # Build absolute paths for pipeline arguments
        abs_mesh_path = (temp_input_dir / input_mesh_source.name).resolve()
        abs_apex_file_path = (temp_input_dir / apex_file_source.name).resolve()
        abs_orifice_file_path = (temp_input_dir / orifice_file_source.name).resolve()

        # Setup pipeline arguments
        test_args_list = [
            '--mesh', str(abs_mesh_path),
            '--apex-file', str(abs_apex_file_path),
            '--orifice-file', str(abs_orifice_file_path),
            '--closed_surface', '0',
            '--use_curvature_to_open', '0',
            '--atrium', 'LA',
            '--open_orifices', '1',
            '--MRI', '1',
            '--resample_input', '1',
            '--target_mesh_resolution', '0.4',
            '--find_appendage', '0',
            '--debug', '1',
            '--no-plot'
        ]

        # Parse arguments
        argument_parser = parser()
        args_instance = argument_parser.parse_args(test_args_list)

        # Run pipeline
        try:
            AugmentA(args_instance)
        except SystemExit as e:
            pytest.fail(f"The pipeline exited prematurely with code {e.code}. See logs for details.")
        except Exception as e:
            pytest.fail(f"The pipeline raised an unexpected exception: {e.__class__.__name__}: {e}")

        # Determine output directory name
        if args_instance.resample_input:
            output_dir_name = f"{args_instance.atrium}_cutted_res_surf"
        else:
            output_dir_name = f"{args_instance.atrium}_cutted_surf"

        current_result_dir = temp_input_dir / output_dir_name
        golden_comparison_dir = test_paths.golden_surf_dir(EXPECTED_OUTPUT_SURF_DIR)

        # Verify both directories exist
        assert current_result_dir.exists(), f"Output directory '{current_result_dir}' was not created."
        assert golden_comparison_dir.exists(), f"Golden results folder '{golden_comparison_dir}' not found."

        # Save results for inspection
        inspection_dir = Path("/tmp/test_results_for_inspection")
        inspection_dir.mkdir(exist_ok=True)
        if current_result_dir.exists():
            shutil.copytree(current_result_dir, inspection_dir / "current_results", dirs_exist_ok=True)
            shutil.copytree(golden_comparison_dir, inspection_dir / "golden_results", dirs_exist_ok=True)
            print(f"Results saved for inspection in: {inspection_dir}")
            print(f"Current results: {inspection_dir / 'current_results'}")
            print(f"Good results: {inspection_dir / 'golden_results'}")

        # Compare results
        compare_directories(current_result_dir, golden_comparison_dir, temp_input_dir)
        print("ALL COMPARISONS PASSED!")

    finally:
        os.chdir(original_cwd)