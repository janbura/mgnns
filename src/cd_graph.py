import torch

# Implementation of a (col,d)-graph.
# Colours are ALWAYS represented by integers 1...n (this is what the model needs)
# Features is a |nodes| x delta matrix. Each row represents a node and its feature.
# Edges is a |edges| x 2 matrix where each row is of the form (i,j) representing an edge from the node corresp. to row i
# in features to the node corresp. to row j in features.
# Edge_colours is a |edges|-sized list where the ith element is the colour of edge in row i of edges.
# Node_names is a |nodes|-sized list where ith element is the name of the node in row i of features.
class CDGraph:
    def __init__(self, col_size: int, delta: int, features: torch.FloatTensor, edges: torch.LongTensor,
                 edge_colours: torch.LongTensor, node_names: list):

        # Sanity checks on the input
        assert col_size > 0
        assert delta > 0
        assert features.shape[0] == len(node_names)
        assert features.shape[1] == delta
        assert edges.shape[0] == edge_colours.shape[0]
        assert edges.shape[1] == 2
        assert all(colour in col_size for colour in edge_colours)
        assert len(node_names) == len(set(node_names))

        self.col_size = col_size
        self.delta = delta
        self.features = features
        self.edges = edges
        self.edge_colours = edge_colours
        self.node_names = node_names
