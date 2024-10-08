import unittest
from src.data_processing.data_preprocessing import preprocess_data
import pandas as pd

class TestDataProcessing(unittest.TestCase):
    def test_preprocess_data(self):
        data = pd.DataFrame({'A': [1, 2, None], 'B': [4, None, 6]})
        processed_data = preprocess_data(data)
        self.assertFalse(processed_data.isnull().values.any())

if __name__ == '__main__':
    unittest.main()
