from src.main import load_image, process_image, process_and_visualize, plot_images, plot_histogram, visualize_3d
import unittest
import numpy as np
from PIL import Image
from io import BytesIO
from unittest.mock import patch
from src.utils.utils import generate_timestamp, check_os

class TestVisualization(unittest.TestCase):
    def test_load_image(self):
        test_image = Image.new('L', (10, 10), color=255) # color white
        img_byte_arr = BytesIO()
        test_image.save(img_byte_arr, format='TIFF')
        img_byte_arr.seek(0)
        result = load_image(img_byte_arr)
        self.assertEqual(result.shape, (10, 10))
        self.assertTrue(np.all(result == 255))  # check if picture is white

class test_process_image(unittest.TestCase):
    def test_process_image(self):
        test_image = np.zeros((10, 10))
        test_image[5:10, 5:10] = 1
        blurred, binary = process_image(test_image)
        self.assertEqual(blurred.shape, (10, 10))
        self.assertEqual(binary.shape, (10, 10))
        self.assertTrue(np.all(binary[5:10, 5:10])) # should be binary true
        # self.assertTrue(np.all(binary[0:5, 0:10] == False)) # should be binary false

class TestVisualize3D(unittest.TestCase):
    def test_visualize_3d(self):
        test_image = np.random.rand(10, 10, 10)  # random 3d image will be created
        try:
            visualize_3d(test_image)
        except Exception as e:
            self.fail(f"visualize_3d raised an exception: {e}")

class TestProcessAndVisualize(unittest.TestCase):
    @patch('os.listdir')
    @patch('os.path.join')
    @patch('src.main.process_image')
    @patch('src.main.load_image')
    def test_process_and_visualize(self, mock_load_image, mock_process_image, mock_join, mock_listdir):
        mock_listdir.return_value = ['image1.tif', 'image2.tif']
        mock_load_image.side_effect = [np.zeros((10, 10)), np.ones((10, 10))]
        mock_process_image.side_effect = [(np.zeros((10, 10)), np.zeros((10, 10))), (np.ones((10, 10)), np.ones((10, 10)))]
        directory = "dummy_directory"
        process_and_visualize(directory)
        mock_load_image.assert_called()
        mock_process_image.assert_called()
