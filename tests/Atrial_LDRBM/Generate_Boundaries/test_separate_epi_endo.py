import os
import sys
import csv
import tempfile
import unittest
from unittest.mock import patch, MagicMock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Atrial_LDRBM.Generate_Boundaries import separate_epi_endo

class TestSeparateEpiEndo(unittest.TestCase):
    # -------------------- Tests for load_element_tags --------------------
    def test_load_element_tags_valid(self):
        # Create a temporary CSV file with valid data.
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, newline='') as tmp:
            fieldnames = ['name', 'tag']
            writer = csv.DictWriter(tmp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({'name': 'a', 'tag': '1'})
            writer.writerow({'name': 'b', 'tag': '2'})
            tmp_filepath = tmp.name

        try:
            result = separate_epi_endo.load_element_tags(tmp_filepath)
            self.assertEqual(result, {'a': '1', 'b': '2'})
        finally:
            os.remove(tmp_filepath)

    def test_load_element_tags_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            separate_epi_endo.load_element_tags("non_existent.csv")

    def test_load_element_tags_missing_columns(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, newline='') as tmp:
            fieldnames = ['wrong', 'column']
            writer = csv.DictWriter(tmp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({'wrong': 'a', 'column': '1'})
            tmp_filepath = tmp.name

        try:
            with self.assertRaises(RuntimeError):
                separate_epi_endo.load_element_tags(tmp_filepath)
        finally:
            os.remove(tmp_filepath)

    # -------------------- Tests for get_wall_tags --------------------
    def test_get_wall_tags_LA(self):
        tag_dict = {'left_atrial_wall_epi': '10', 'left_atrial_wall_endo': '20'}
        result = separate_epi_endo.get_wall_tags(tag_dict, "LA")
        self.assertEqual(result, (10, 20))

    def test_get_wall_tags_RA(self):
        tag_dict = {'right_atrial_wall_epi': '30', 'right_atrial_wall_endo': '40'}
        result = separate_epi_endo.get_wall_tags(tag_dict, "RA")
        self.assertEqual(result, (30, 40))

    def test_get_wall_tags_invalid_atrium(self):
        tag_dict = {}
        with self.assertRaises(ValueError):
            separate_epi_endo.get_wall_tags(tag_dict, "XX")

    def test_get_wall_tags_missing_key(self):
        tag_dict = {'left_atrial_wall_epi': '10'}  # Missing left_atrial_wall_endo
        with self.assertRaises(KeyError) as cm:
            separate_epi_endo.get_wall_tags(tag_dict, "LA")
        self.assertIn("Missing expected tag for LA", str(cm.exception))

    # -------------------- Tests for threshold_model --------------------
    @patch('Atrial_LDRBM.Generate_Boundaries.separate_epi_endo.get_threshold_between')
    def test_threshold_model_success(self, mock_get_threshold_between):
        dummy_thresh = MagicMock()
        dummy_thresh.GetOutput.return_value = "dummy_output"
        mock_get_threshold_between.return_value = dummy_thresh

        result = separate_epi_endo.threshold_model("model", 1, 2)
        self.assertEqual(result, dummy_thresh)

    @patch('Atrial_LDRBM.Generate_Boundaries.separate_epi_endo.get_threshold_between')
    def test_threshold_model_failure(self, mock_get_threshold_between):
        mock_get_threshold_between.return_value = None
        with self.assertRaises(RuntimeError):
            separate_epi_endo.threshold_model("model", 1, 2)

    # -------------------- Tests for write_filtered_meshes --------------------
    @patch('Atrial_LDRBM.Generate_Boundaries.separate_epi_endo.vtk_polydata_writer')
    @patch('Atrial_LDRBM.Generate_Boundaries.separate_epi_endo.vtk_obj_writer')
    def test_write_filtered_meshes(self, mock_obj_writer, mock_polydata_writer):
        dummy_mesh = "dummy_mesh"
        meshname = "testmesh"
        atrium = "LA"
        suffix = "_test"
        separate_epi_endo.write_filtered_meshes(meshname, atrium, suffix, dummy_mesh)
        expected_filename_obj = f"{meshname}_{atrium}{suffix}.obj"
        expected_filename_vtk = f"{meshname}_{atrium}{suffix}.vtk"
        mock_obj_writer.assert_called_with(expected_filename_obj, dummy_mesh)
        mock_polydata_writer.assert_called_with(expected_filename_vtk, dummy_mesh)

    # -------------------- Tests for separate_epi_endo --------------------
    def test_separate_epi_endo_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            separate_epi_endo.separate_epi_endo("non_existent.obj", "LA")

    @patch('Atrial_LDRBM.Generate_Boundaries.separate_epi_endo.write_filtered_meshes')
    @patch('Atrial_LDRBM.Generate_Boundaries.separate_epi_endo.apply_vtk_geom_filter')
    @patch('Atrial_LDRBM.Generate_Boundaries.separate_epi_endo.threshold_model')
    @patch('Atrial_LDRBM.Generate_Boundaries.separate_epi_endo.smart_reader')
    @patch('Atrial_LDRBM.Generate_Boundaries.separate_epi_endo.get_wall_tags')
    @patch('Atrial_LDRBM.Generate_Boundaries.separate_epi_endo.load_element_tags')
    @patch('os.path.exists')
    def test_separate_epi_endo_success(self, mock_exists, mock_load_tags, mock_get_wall_tags,
                                         mock_smart_reader, mock_threshold_model,
                                         mock_apply_filter, mock_write_filtered_meshes):
        # Simulate file exists for both mesh and CSV.
        mock_exists.return_value = True

        # Setup dummy tags and wall tags.
        dummy_tags = {'left_atrial_wall_epi': '10', 'left_atrial_wall_endo': '20'}
        mock_load_tags.return_value = dummy_tags
        mock_get_wall_tags.return_value = (10, 20)

        # Setup dummy model from smart_reader.
        dummy_model = MagicMock(name="dummy_model")
        mock_smart_reader.return_value = dummy_model

        # Create dummy threshold object with GetOutput method.
        dummy_thresh = MagicMock()
        dummy_thresh.GetOutput.return_value = "dummy_output"
        # threshold_model will be called three times; return dummy_thresh each time.
        mock_threshold_model.return_value = dummy_thresh

        # apply_vtk_geom_filter returns filtered output.
        mock_apply_filter.return_value = "filtered_output"

        # Call the function under test.
        dummy_path = "dummy.obj"
        separate_epi_endo.separate_epi_endo(dummy_path, "LA")

        # The meshname is the input file name without extension.
        expected_meshname = "dummy"

        # Check that write_filtered_meshes is called three times with correct suffixes.
        calls = [
            ((expected_meshname, "LA", "", "filtered_output"),),
            ((expected_meshname, "LA", "_epi", "filtered_output"),),
            ((expected_meshname, "LA", "_endo", "filtered_output"),)
        ]
        actual_calls = mock_write_filtered_meshes.call_args_list
        self.assertEqual(len(actual_calls), 3)
        for call, expected in zip(actual_calls, calls):
            self.assertEqual(call[0], expected[0])

if __name__ == '__main__':
    unittest.main()
