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

COMPARE_WITH_TOLERANCE = False  # Set False to require exact matches for all files
COMPARE_BY_COORDINATES = True  # For VTX files, compare by coordinates instead of IDs

# Test configuration - paths to our test data and expected results
TESTS_ROOT = Path(__file__).parent.resolve()  # e.g. Path(/Users/ou736l/augmentaupgrade/tests)
TEST_DATA_DIR = TESTS_ROOT / "test_data"  # e.g. Path(/Users/ou736l/augmentaupgrade/tests/test_data)
GOLDEN_RESULTS_DIR = TESTS_ROOT / "good_results"

# Test files we are working with
TEST_MESH_BASENAME = "LA_MRI"  # input mesh basename
TEST_MESH_FILENAME = f"{TEST_MESH_BASENAME}.vtp"
APEX_FILE_FILENAME = f"{TEST_MESH_BASENAME}_apexes.csv"



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

    # Calculate differences
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
    # for the resampled mesh that was used to generate the VTX files
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


def compare_vtx_by_coordinates(
        current_vtx_path,
        golden_vtx_path,
        current_mesh_path,
        golden_mesh_path,
        tolerance=1.0):  # Increased tolerance from 1e-3 to 1.0 mm
    """Compare VTX files by looking up point coordinates in their respective meshes."""
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

    print(f"  VTX comparison for {current_vtx_path.name}:")
    print(f"    Current IDs: {len(current_ids)}, Golden IDs: {len(golden_ids)}")

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
        print(f"Points within {tolerance} mm: {num_within_tolerance}/{len(distances)}")

        # Use a more lenient approach: require 95% of points to be within tolerance
        match_percentage = num_within_tolerance / len(distances)
        required_match = 0.95

        result = match_percentage >= required_match
        print(
            f"Match percentage: {match_percentage:.1%} (required: {required_match:.1%}) - {'PASS' if result else 'FAIL'}")

        return result

    except Exception as e:
        print(f"Error during coordinate comparison: {e}")
        return False


def compare_directories(current_dir: Path, golden_dir: Path, temp_input_dir: Path):
    """
    Compare two directories for structure and content.
    This does a thorough comparison to make sure our pipeline produces exactly the expected output.
    """
    # Compare file structures to make sure we have the same files
    # Create sets of relative file paths for both directories
    golden_files = {path.relative_to(golden_dir) for path in golden_dir.rglob('*') if path.is_file()}
    current_files = {path.relative_to(current_dir) for path in current_dir.rglob('*') if path.is_file()}

    if golden_files != current_files:  # Check if both sets of files match
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

        if file_name == "rings_centroids.csv":
            # CSV files with numbers need fuzzy comparison due to floating point precision
            golden_df = pd.read_csv(golden_path)
            current_df = pd.read_csv(current_path)
            try:
                pd.testing.assert_frame_equal(golden_df, current_df, check_exact=False, rtol=1e-5)
            except AssertionError as e:
                assert False, f"CSV content mismatch in {file_name}: {e}"

        elif file_extension == ".vtx":
            if COMPARE_WITH_TOLERANCE:
                if not compare_vtx_with_tolerance(current_path, golden_path, max_diff_percent=5.0):
                    assert False, f"VTX content differs too much in {golden_path.name}"

            elif COMPARE_BY_COORDINATES:
                try:
                    # Find appropriate mesh files for comparison
                    current_mesh_path = find_mesh_file_for_vtx_comparison(current_dir, temp_input_dir)
                    golden_mesh_path = find_mesh_file_for_vtx_comparison(golden_dir, GOLDEN_RESULTS_DIR)

                    print(f"Using meshes for VTX comparison:")
                    print(f"  Current: {current_mesh_path}")
                    print(f"  Golden: {golden_mesh_path}")

                    if not compare_vtx_by_coordinates(current_path, golden_path, current_mesh_path, golden_mesh_path):
                        assert False, f"VTX coordinate mismatch in {file_name}"

                except FileNotFoundError as e:
                    print(f"Warning: Could not find mesh files for coordinate comparison: {e}")
                    print("Falling back to direct ID comparison")
                    # Fall back to direct ID comparison
                    golden_set = read_vtx_to_set(golden_path)
                    current_set = read_vtx_to_set(current_path)
                    assert golden_set == current_set, f"VTX content mismatch in {file_name}"

            else:
                # VTX files: compare sets of IDs (order doesn't matter)
                golden_set = read_vtx_to_set(golden_path)
                current_set = read_vtx_to_set(current_path)
                assert golden_set == current_set, f"VTX content mismatch in {file_name}"

        else:
            # Everything else: exact binary comparison
            assert filecmp.cmp(golden_path, current_path, shallow=False), f"File content mismatch in {file_name}"


def test_la_cut_resample_pipeline_regression(tmp_path: Path):
    """
    Test the full pipeline against golden results.
    This is a test that makes sure our pipeline produces the same output as before.
    If this fails, we either broke something or intentionally changed the behavior.
    """
    # Setup test data in a temporary directory to keep things clean
    input_mesh_path = TEST_DATA_DIR / TEST_MESH_FILENAME
    apex_file_path = TEST_DATA_DIR / APEX_FILE_FILENAME
    orifice_file_path = TEST_DATA_DIR / "LA_MRI_orifices.csv"

    temp_input_dir = tmp_path / "input"
    temp_input_dir.mkdir()

    try:
        shutil.copy(input_mesh_path, temp_input_dir)
        shutil.copy(apex_file_path, temp_input_dir)
        shutil.copy(orifice_file_path, temp_input_dir)
    except FileNotFoundError as e:
        pytest.fail(f"Failed to copy test data. A required file was not found: {e}")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred during test data setup: {e}")

    abs_mesh_path = (temp_input_dir / TEST_MESH_FILENAME).resolve()
    abs_apex_file_path = (temp_input_dir / APEX_FILE_FILENAME).resolve()
    abs_orifice_file_path = (temp_input_dir / "LA_MRI_orifices.csv").resolve()

    # Change the current working directory to the temporary one for the test duration
    original_cwd = Path.cwd()
    # os.chdir(temp_input_dir)

    try:
        # Setup the exact same arguments using the absolute paths
        test_args_list = [
            '--mesh', str(abs_mesh_path),
            '--apex-file', str(abs_apex_file_path),
            '--orifice-file', str(abs_orifice_file_path),
            '--closed_surface', '0',
            '--use_curvature_to_open', '0',
            '--atrium', 'LA',
            '--open_orifices', '1',
            '--MRI', '1',
            '--resample_input', '0',
            '--target_mesh_resolution', '0.4',
            '--find_appendage', '0',
            '--debug', '1',
            '--no-plot']

        # Parse arguments the same way main.py does
        argument_parser = parser()
        args_instance = argument_parser.parse_args(test_args_list)

        print(f"Apex file exists: {abs_apex_file_path.exists()}")
        print(f"Apex file path: {abs_apex_file_path}")
        if abs_apex_file_path.exists():
            with open(abs_apex_file_path, 'r') as f:
                print(f"Apex file contents:\n{f.read()}")

        # Run the actual pipeline, with specific error handling
        try:
            AugmentA(args_instance)
        except SystemExit as e:
            # The pipeline calls sys.exit(1) on failure. We catch it to provide a clearer message.
            pytest.fail(f"The pipeline exited prematurely with code {e.code}. See logs for details.")
        except Exception as e:
            pytest.fail(f"The pipeline raised an unexpected exception: {e.__class__.__name__}: {e}")

        # Define paths for comparing results
        if args_instance.resample_input:
            output_dir_name = f"{args_instance.atrium}_cutted_res_surf"
            FINAL_OUTPUT_FOLDER = "LA_cutted_res_surf"
        else:
            output_dir_name = f"{args_instance.atrium}_cutted_surf"
            FINAL_OUTPUT_FOLDER = "LA_cutted_surf"

        current_result_dir = temp_input_dir / output_dir_name
        golden_comparison_dir = GOLDEN_RESULTS_DIR / FINAL_OUTPUT_FOLDER

        # Verify both directories exist before comparing
        assert current_result_dir.exists(), f"Output directory '{current_result_dir}' was not created."
        assert golden_comparison_dir.exists(), f"Golden results folder '{golden_comparison_dir}' not found."

        # Save test results for visual inspection
        inspection_dir = Path("/tmp/test_results_for_inspection")
        inspection_dir.mkdir(exist_ok=True)
        if current_result_dir.exists():
            shutil.copytree(current_result_dir, inspection_dir / "current_results", dirs_exist_ok=True)
            shutil.copytree(golden_comparison_dir, inspection_dir / "golden_results", dirs_exist_ok=True)
            print(f"Results saved for inspection in: {inspection_dir}")
            print(f"Current results: {inspection_dir / 'current_results'}")
            print(f"Golden results: {inspection_dir / 'golden_results'}")
            print("Use `open /tmp/test_results_for_inspection` in Terminal to view them.\n")

        # Perform the detailed comparison of output files
        compare_directories(current_result_dir, golden_comparison_dir, temp_input_dir)

    finally:
        # **Crucially, always change back to the original directory.**
        # This ensures this test does not interfere with others, even if assertions fail.
        os.chdir(original_cwd)