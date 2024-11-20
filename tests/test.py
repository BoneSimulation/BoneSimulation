"""
Unit tests for image processing and visualization functions.

This module contains tests for the functions defined in the src.main module,
including load_image, process_image, process_and_visualize, and visualize_3d.
"""

import unittest
from io import BytesIO
from unittest.mock import patch
import numpy as np
from PIL import Image

from src.main import load_image, process_image, process_and_visualize, visualize_3d


class TestLoadImage(unittest.TestCase):
    """Test case for the load_image function."""

    def test_load_image(self):
        """Test loading an image and checking its properties."""
        test_image = Image.new('L', (10, 10), color=255)  # color white
        img_byte_arr = BytesIO()
        test_image.save(img_byte_arr, format='TIFF')
        img_byte_arr.seek(0)
        result = load_image(img_byte_arr)
        self.assertEqual(result.shape, (10, 10))
        self.assertTrue(np.all(result == 255))  # check if picture is white


class TestProcessImage(unittest.TestCase):
    """Test case for the process_image function."""

    def test_process_image(self):
        """Test processing an image and checking the output shapes."""
        test_image = np.zeros((10, 10))
        test_image[5:10, 5:10] = 1
        blurred, binary = process_image(test_image)
        self.assertEqual(blurred.shape, (10, 10))
        self.assertEqual(binary.shape, (10, 10))
        self.assertTrue(np.all(binary[5:10, 5:10]))  # should be binary true
        # self.assertTrue(np.all(binary[0:5, 0:10] == False))  # should be binary false


class TestVisualize3D(unittest.TestCase):
    """Test case for the visualize_3d function."""

    def test_visualize_3d(self):
        """Test the visualize_3d function without raising exceptions."""
        test_image = np.random.rand(10, 10, 10)  # random 3D image
        try:
            visualize_3d(test_image)
        except Exception as e:
            self.fail(f"visualize_3d raised an exception: {e}")


class TestProcessAndVisualize(unittest.TestCase):
    """Test case for the process_and_visualize function."""

    @patch('os.listdir')
    @patch('os.path.join')
    @patch('src.main.process_image')
    @patch('src.main.load_image')
    def test_process_and_visualize(self, mock_load_image, mock_process_image, mock_join, mock_listdir):
        """Test the process_and_visualize function with mocked dependencies."""
        mock_listdir.return_value = ['image1.tif', 'image2.tif']
        mock_load_image.side_effect = [np.zeros((10, 10)), np.ones((10, 10))]
        mock_process_image.side_effect = [
            (np.zeros((10, 10)), np.zeros((10, 10))),
            (np.ones((10, 10)), np.ones((10, 10)))
        ]
        directory = "dummy_directory"
        process_and_visualize(directory)
        mock_load_image.assert_called()
        mock_process_image.assert_called()


if __name__ == "__main__":
    unittest.main()
