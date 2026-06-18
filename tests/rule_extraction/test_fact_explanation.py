import numpy as np
import torch
import pytest

from src.encodings.canonical import CanonicalEncoderDecoder
from src.encodings.noncanonical.iclr22 import ICLREncoderDecoder
from src.encodings.noncanonical.identity import IdentityEncoderDecoder
from src.model.cd_graph import TraceCollector, CDGraph
from src.model.gnn_architectures import GNN
from src.model.gnn_transformation import apply_model, apply_nc_decoder
from src.rule_extraction.tree_shaped_conjunction import TreeShapedConjunction, Variable, walk
from src.rule_extraction.fact_explanation import BasicExplanation, FactExplainer
from src.utils.utils import TYPE_PRED
from src.utils.bitset import BitSet

# Assume a signature with A, B (unary), R, S (binary)
# Assume also a canonical encoding so (\col,\delta) = (2,2)
# Layer_1:
# Feature 1: current node has feature A
# Feature 2: R-neighbour has feature A
# Feature 3: R-neighbour has feature B and S-neighbour has feature B
# Feature 4: fixed value of 1
# Layer 2:
# Feature 1: current node has features 1 & 2 and S-neighbour with feature 1
# Feature 2: has an S-neighbour with feature 3 & S-neighbour with feature 4
def example_model_1(device):
    model = GNN(feature_dimension=2, num_edge_colours=2,aggregation_1="max",aggregation_2="max").to(device)
    # Checks
    assert model.num_layers == 2
    assert model.num_colours == 2
    assert model.dimensions == [2,4,2]
    # Matrix B1-c1
    model.conv1.weights.data[0] = torch.tensor([
    [0., 0.],
    [1., 0.],
    [0., 1.],
    [0., 0.]
    ])
    # Matrix B1-c2
    model.conv1.weights.data[1] = torch.tensor([
        [0., 0.],
        [0., 0.],
        [0., 1.],
        [0., 0.]
    ])
    # Matrix B2-c1
    model.conv2.weights.data[0] = torch.tensor([
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]
    ])
    # Matrix B2-c2
    model.conv2.weights.data[1] = torch.tensor([
        [1., 0., 0., 0.],
        [0., 0., 1., 1.]
    ])
    # Matrix A1
    model.lin_self_1.weight.data = torch.tensor([
        [1., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.]
    ])
    # Matrix A2
    model.lin_self_2.weight.data = torch.tensor([
        [1., 1., 0., 0.],
        [0., 0., 0., 0.]
    ])
    # Bias b1
    model.lin_self_1.bias.data = torch.tensor([0.,0.,-1.,1.])
    # Bias b2
    model.lin_self_2.bias.data = torch.tensor([-2.,-1.])
    return model

def example_cdgraph_1():
    # R(b,a), R(c,a), S(b,a), A(a), A(b), B(b), B(c)
    features = torch.tensor([
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0]
    ], dtype=torch.float)
    edges = torch.tensor([
     [1,2,1],
     [0,0,0]
    ], dtype=torch.long)
    edge_colours = torch.tensor([0,0,1], dtype=torch.long)
    node_names = ["a","b","c"]
    cdgraph = CDGraph(col_size=2,delta=2,features=features,edges=edges,edge_colours=edge_colours,node_names=node_names)
    assert cdgraph.node_names_to_indices["a"] == 0
    assert cdgraph.node_names_to_indices["b"] == 1
    assert cdgraph.node_names_to_indices["c"] == 2
    return cdgraph


def ex_1_1_activation_0():
    return torch.tensor([
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=torch.float)

def ex_1_1_activation_1():
    return torch.tensor([
        [1., 1., 1., 1.],
        [1., 0., 0., 1.],
        [0., 0., 0., 1.]
    ], dtype=torch.float)

def ex_1_1_activation_2():
    return torch.sigmoid(torch.tensor([
        [-9,-10],
        [-11,-11],
        [-12,-11]
    ], dtype=torch.float)) # Sorry for magic numbers. This just matches the GNN architecture AND the selected bias

# Assuming here a canonical encoding with signature A, B (unary), R, S (binary)
def make_mock_fe():
    fact = ("a",TYPE_PRED,"A")
    trace = TraceCollector()
    external_encoder = IdentityEncoderDecoder(load_from_document=None, unary_predicates=["A","B"],
                                              binary_predicates=["R","S"])
    internal_encoder = CanonicalEncoderDecoder(load_from_document=None,unary_predicates=["A","B"],
                                               binary_predicates=["R","S"])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = example_model_1(device)
    cd_graph = example_cdgraph_1()
    threshold = 0.000123 # sigmoid of 1-10 (GNN architecture does this -10)
    apply_model(cd_graph, device, model, trace)
    # Check the activations match what we expect
    assert torch.equal(trace.fl0,ex_1_1_activation_0())
    assert torch.equal(trace.fl1,ex_1_1_activation_1())
    assert torch.equal(trace.fl2,ex_1_1_activation_2())
    fe = FactExplainer(device,fact,model,threshold,trace,external_encoder,internal_encoder)
    return fe


class TestBasicExplanation:

    # We'd expect a conjunction S(x,y) and R(x,z) with both y and z mapped to b; with y at level 1 and z at level 0.
    # Explanation: node for a gathers info about itself and R.b in first layer, then about S.b in second layer
    def test_basic_explanation(self):
        fe = make_mock_fe()
        ba = BasicExplanation(fe)

        # Verify the TreeShapedConjunction
        assert isinstance(ba.conjunction, TreeShapedConjunction)
        assert len(ba.conjunction) == 3
        vx = ba.conjunction.root_node
        assert isinstance(vx, Variable)
        assert vx.features == BitSet.from_subset(dimension=2, subset={0})
        assert vx.level == 2
        assert (2,1,0) in vx.children # In layer 2, we introduce a child variable via S (1) for position 0
        assert (1,0,0) in vx.children # In layer 1, we introduce a child variable via R (0) for position 0
        vy = vx.children[(2,1,0)]
        assert isinstance(vy, Variable)
        assert vy.features == BitSet.from_subset(dimension=2, subset={0})
        assert vy.level == 1
        assert vy.children == {}
        vz = vx.children[(1,0,0)]
        assert isinstance(vz, Variable)
        assert vz.features == BitSet.from_subset(dimension=2, subset={0})
        assert vz.level == 0
        assert vz.children == {}

        # Verify the mapping of variables to constants (the paper's \nu)
        assert vx in ba.var_const_idx
        assert ba.var_const_idx[vx] == 0
        assert vy in ba.var_const_idx
        assert ba.var_const_idx[vy] == 1
        assert vz in ba.var_const_idx
        assert ba.var_const_idx[vz] == 1

        # Verify the (var,layer) to mask mapping (the paper's \mu)
        # These expected results were done by hand.
        assert ba.var_layer_mask[(vx,2)] == BitSet.from_subset(2, {0})
        assert ba.var_layer_mask[(vx,1)] == BitSet.from_subset(4, {0,1})
        assert ba.var_layer_mask[(vx,0)] == BitSet.from_subset(2, {0})
        assert (vy,2) not in ba.var_layer_mask
        assert ba.var_layer_mask[(vy, 1)] == BitSet.from_subset(4, {0})
        assert ba.var_layer_mask[(vy, 0)] == BitSet.from_subset(2, {0})
        assert (vz,2) not in ba.var_layer_mask
        assert (vz,1) not in ba.var_layer_mask
        assert ba.var_layer_mask[(vz, 0)] == BitSet.from_subset(2, {0})


class TestFactExplainer:

    def test_fat_explainer(self):
        fe = make_mock_fe()

        assert fe.node_to_index == {"a": 0, "b":1, "c":2}
        assert fe.cd_ent1 == "a"
        assert fe.cd_ent3 == "A"
        assert fe.cd_fact_const_index == 0
        assert fe.node_to_index["a"] == 0
        assert fe.cd_fact_pred_pos == 0
        assert isinstance(fe.basic_explanation,BasicExplanation)
        rule =  fe.rule
        head, body = rule.rstrip(' .\n').split(' :- ')
        body_atoms = set(body.split(', '))
        assert head ==  "<A>[?X0]"
        assert body_atoms == {"<A>[?X0]", "<S>[?X1,?X0]", "<A>[?X1]", "<R>[?X2,?X0]", "<A>[?X2]"}

