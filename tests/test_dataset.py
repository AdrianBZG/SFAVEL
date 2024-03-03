import pytest
from unittest.mock import MagicMock, patch
from src.dataset.data_handling import get_dataset

@patch('src.dataset.data_handling.get_fact_dataset')
@patch('src.dataset.data_handling.get_kg_dataset')
def test_get_dataset(mock_get_kg_dataset, mock_get_fact_dataset):
    # Mock the return values of get_fact_dataset and get_kg_dataset
    mock_get_fact_dataset.return_value = MagicMock()
    mock_get_kg_dataset.return_value = MagicMock()

    fact_dataset, kg_dataset = get_dataset(root_path="./data", split="train", fact_dataset="FEVER", kg_dataset="wikidata5m", node_feature_dim=128)

    # Check that get_fact_dataset and get_kg_dataset were called with the correct arguments
    mock_get_fact_dataset.assert_called_once_with("./data", "train", "FEVER")
    mock_get_kg_dataset.assert_called_once_with("./data", "wikidata5m", node_feature_dim=128)

    # Check that the returned values are as expected
    assert fact_dataset == mock_get_fact_dataset.return_value
    assert kg_dataset == mock_get_kg_dataset.return_value

    # Test that a ValueError is raised when an invalid split is provided
    with pytest.raises(AssertionError):
        get_dataset(root_path="./data", split="invalid", fact_dataset="FEVER", kg_dataset="wikidata5m", node_feature_dim=128)