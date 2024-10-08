import unittest
from src.models.model_training import train_model
import pandas as pd
import numpy as np

class TestModelTraining(unittest.TestCase):
    def test_train_model(self):
        X_train = pd.DataFrame(np.random.rand(100, 5))
        y_train = np.random.randint(2, size=100)
        model = train_model(X_train, y_train)
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()
