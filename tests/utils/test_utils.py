from src.utils.utils import backpropagate_relevance
from src.utils.bitset import BitSet
import torch
import pytest

class TestBackpropagateRelevance:

    def test_backpropagate_relevance_empty(self):
        fm = BitSet.from_subset(3, {})
        matrix = torch.tensor(
            [
                [1, -1, 2],
                [0, 0, 0],
                [-5, -2, -1],
            ]
        )
        result = backpropagate_relevance(fm,matrix)
        assert isinstance(result, BitSet)
        assert result.as_set() == set()

    def test_backpropagate_relevance_with_activations(self):
        fm = BitSet.from_subset(3, {0,1,2})
        matrix = torch.tensor(
            [
                [1, 0, 0],
                [0, 0, 0],
                [1, 0, 1],
            ]
        )
        activations = torch.tensor([0, 0, 2])
        result = backpropagate_relevance(fm, matrix, activations)
        assert result.as_set() == {2}

    def test_zero_activations_cancel_everything(self):
        fm = BitSet.from_subset(3, {0,1,2})
        matrix = torch.tensor(
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ]
        )
        activations = torch.tensor([0, 1, 1])
        result = backpropagate_relevance(fm, matrix, activations)
        assert result.as_set() == {1,2}


    def test_backpropagate_relevance_activations_filter_everything(self):
        fm = BitSet.from_subset(2, {0,1})
        matrix = torch.tensor(
            [
                [1, 1, 1],
                [0, 0, 0],
            ]
        )
        activations = torch.tensor([0, 0, 0])
        result = backpropagate_relevance(fm, matrix, activations)
        assert result.is_empty()

    def test_backpropagate_relevance_empty_mask_returns_empty(self):
        fm = BitSet.from_subset(3, set())
        matrix = torch.tensor(
            [
                [1, 2],
                [3, 4],
                [5, 6],
            ]
        )
        result = backpropagate_relevance(fm,matrix)
        assert result.dimension == 2
        assert result.is_empty()

    def test_backpropagate_relevance_dimension_mismatch(self):
        fm = BitSet.from_subset(3, {0})
        matrix = torch.zeros((4, 2))
        with pytest.raises(
            ValueError,
            match="does not match matrix row dimension",
        ):
            backpropagate_relevance(fm,matrix)

    def test_backpropagate_relevance_multiple_relevant_rows(self):
        fm = BitSet.from_subset(3, {0, 2})
        matrix = torch.tensor(
            [
                [0, 1, 0, 0],
                [1, 1, 1, 1],
                [1, 0, 0, 2],
            ]
        )
        result = backpropagate_relevance(fm,matrix)
        assert result.as_set() == {0, 1, 3}