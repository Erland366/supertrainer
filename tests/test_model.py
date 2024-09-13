import unittest

from supertrainer.models.model import Model


class TestModel(unittest.TestCase):
    def test_model_prediction(self):
        # Create an instance of the model
        model = Model()

        # Perform prediction using the model
        result = model.predict()

        # Assert the expected result
        self.assertEqual(result, "Prediction")


if __name__ == "__main__":
    unittest.main()
