from src.utils.bitset import BitSet
import numpy as np

class TreeShapedConjunction:
    def __init__(self, n_colours):
        self.n_colours = n_colours
        self.root_node = None

    def walk(self):
        return walk(self.root_node)

    def is_empty(self):
        return self.root_node is None

    def __len__(self):
        return  sum(1 for _ in self.walk())


# Represents a variable, with a given level and set of features (becoming unary predicates when unfolding the rule)
class Variable:
    def __init__(self, features: BitSet=None, level=int):
        self.features = features
        self.level = level
        self.children: dict[tuple[int,int,int], Variable] = {}  # Maps triple (l,col,j) to the relevant node.

    def get_feature_list(self):
        return self.features.elements()

# Note that this is a generator function
def walk(node: Variable):
    if node is None:
        return
    children_snapshot = list(node.children.values())  # snapshot BEFORE yielding
    yield node
    for child in children_snapshot:
        yield from walk(child)





