import shutil
import filecmp
from pathlib import Path
import pytest
import pandas as pd

from pipeline import AugmentA
from main import parser

# Test configuration - paths to our test data and expected results
TESTS_ROOT = Path(__file__).parent.resolve()  # e.g. Path(/Users/ou736l/augmentaupgrade/tests)
TEST_DATA_DIR = TESTS_ROOT / "test_data"  # e.g. Path(/Users/ou736l/augmentaupgrade/tests/test_data)
GOLDEN_RESULTS_DIR = TESTS_ROOT / "good_results"

# Test files we are working with
TEST_MESH_BASENAME = "LA_MRI"  # input mesh basename
TEST_MESH_FILENAME = f"{TEST_MESH_BASENAME}.vtp"
APEX_FILE_FILENAME = f"{TEST_MESH_BASENAME}_apexes.csv"
FINAL_OUTPUT_FOLDER = f"{TEST_MESH_BASENAME}_cutted_res_surf"


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


def compare_directories(current_dir: Path, golden_dir: Path):
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

        file_extension = golden_path.suffix
        file_name = golden_path.name

        if file_name == "rings_centroids.csv":
            # CSV files with numbers need fuzzy comparison due to floating point precision
            golden_df = pd.read_csv(golden_path)
            current_df = pd.read_csv(current_path)
            pd.testing.assert_frame_equal(golden_df, current_df, check_exact=False, rtol=1e-5)

        elif file_extension == ".vtx":
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

    # Copy test files to temp directory (pytest provides tmp_path)
    temp_input_dir = tmp_path / "input"
    temp_input_dir.mkdir()
    shutil.copy(input_mesh_path, temp_input_dir)
    shutil.copy(apex_file_path, temp_input_dir)

    temp_mesh_path = temp_input_dir / TEST_MESH_FILENAME

    # Setup the exact same arguments used to create the golden results
    test_args_list = [
        '--mesh', str(temp_mesh_path),
        '--apex-file', str(temp_input_dir / APEX_FILE_FILENAME),
        '--closed_surface', '0',
        '--use_curvature_to_open', '1',
        '--atrium', 'LA',
        '--open_orifices', '1',
        '--MRI', '1',
        '--resample', '1',
        '--resample_input', '1',
        '--target_mesh_resolution', '0.4',
        '--find_appendage', '0',
        '--debug', '1']

    # Parse arguments the same way main.py does
    argument_parser = parser()
    args_instance = argument_parser.parse_args(test_args_list)

    # Run the actual pipeline
    AugmentA(args_instance)

    # Check that output was created correctly
    output_dir_name = f"{temp_mesh_path.stem}_cutted_res_surf"
    current_result_dir = temp_input_dir / output_dir_name
    golden_comparison_dir = GOLDEN_RESULTS_DIR / FINAL_OUTPUT_FOLDER

    # Verify both directories exist before comparing
    assert current_result_dir.exists(), f"Output directory '{current_result_dir}' was not created."
    assert golden_comparison_dir.exists(), f"Golden results folder '{golden_comparison_dir}' not found."

    # Do the full comparison
    compare_directories(current_result_dir, golden_comparison_dir)
