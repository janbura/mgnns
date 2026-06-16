from src.utils.bitset import BitSet
import numpy as np

# The features should be between 1 and dimension
class FeatureMask:
    def __init__(self, dimension: int, features: set[int]):
        self.mask = BitSet.from_subset(dimension, features)

    # Given a matrix M of dimension m x n and a vector v of dim n, this computes the mask for that vector.
    def backpropagate_relevance(self, matrix, activations=None):
        if self.mask != 0:
            matrix_relevant_rows = matrix[self.mask.elements(), :]
            any_positive = np.any(matrix_relevant_rows > 0, axis=0)  # dim_{l-1} Boolean vector
            if activations:
                mask = (activations > 0) & any_positive
            else:
                mask = any_positive
            return FeatureMask(len(activations), set(np.where(mask)[0]))
        else:
            # For some reason the above formula does not work in the degenerate case where it's all zeroes.
            return FeatureMask(len(activations), set())

class TreeShapedConjunction:
    def __init__(self, n_features, n_colours):
        self.n_features = n_features
        self.n_colours = n_colours
        self.root_node = None

    def walk(self):
        return walk(self.root_node)

    def is_empty(self):
        return self.root_node is None

    def __len__(self):
        return  sum(1 for _ in self.walk())

class Variable:
    def __init__(self, feature_mask: FeatureMask, level=int):
        self.features = feature_mask  # Features are represented by bitmasks
        self.level = level
        self.children = {}  # Maps triple (l,col,j) to the relevant node.

    def get_feature_list(self):
        return self.features.mask.elements()

def walk(node: Variable):
    if node is None:
        return
    yield node
    for child in list(node.children.values()): # Snapshot to avoid iterating over new nodes
        yield from walk(child)





