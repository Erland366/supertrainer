# Import necessary modules for testing
import pytest
from preskripsi_training.data.dataset import Dataset

# Test cases for the Dataset class
class TestDataset:
    def test_load_data(self):
        # Create an instance of the Dataset class
        dataset = Dataset()

        # Test loading data
        data = dataset.load_data()

        # Assert that the loaded data is not empty
        assert len(data) > 0

    def test_preprocess_data(self):
        # Create an instance of the Dataset class
        dataset = Dataset()

        # Test data preprocessing
        preprocessed_data = dataset.preprocess_data()

        # Assert that the preprocessed data is not empty
        assert len(preprocessed_data) > 0

    def test_split_data(self):
        # Create an instance of the Dataset class
        dataset = Dataset()

        # Test data splitting
        train_data, test_data = dataset.split_data()

        # Assert that the train and test data are not empty
        assert len(train_data) > 0
        assert len(test_data) > 0

    def test_get_data_stats(self):
        # Create an instance of the Dataset class
        dataset = Dataset()

        # Test getting data statistics
        stats = dataset.get_data_stats()

        # Assert that the statistics are calculated correctly
        assert isinstance(stats, dict)
        assert "num_samples" in stats
        assert "num_classes" in stats
        assert "class_distribution" in stats