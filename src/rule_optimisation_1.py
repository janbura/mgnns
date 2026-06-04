from src.gnn_transformation import apply_c_encoder, apply_model
from utils import type_pred
from fact_explanation import FactExplainer, BasicExplanation
import torch
from gnn_transformation import apply_gnn_transformation
from torch_geometric.data import Data
import numpy as np

# BEST APPROXIMATION 1
# Make a dictionary of all unary atoms of gamma_i and compute an ``influence'' value of each
# by creating the minimal sub-rule of Gamma_i that connects this atom with the head variable.
# Ground such rule and apply the GNN to it, then check the value of the node and position corresponding to the head
# fact. If zero, start travelling backwards through the relevant list of colours until you find one that
# has relevant nodes not equal to zero.

# Auxiliary method to quickly test if a dataset suffices to derive the target fact
def test_gr_dataset(body, fe: FactExplainer):
    if not body:
        gr_features = torch.FloatTensor(np.zeros((1, fe.model.layer_dimension(0))))
        gr_edge_list = torch.LongTensor(2, 0)
        gr_dataset = Data(x=gr_features, edge_index=gr_edge_list, edge_type=torch.LongTensor([])).to(fe.device)
        gnn_output_gr = fe.model(gr_dataset)
        return gnn_output_gr[0][fe.cd_fact_pred_pos] >= fe.threshold
    else:
        cd_graph = apply_c_encoder(body,fe.internal_encoder)
        output_cd_graph = apply_model(cd_graph, fe.device, fe.model)
        return (output_cd_graph.features[output_cd_graph.get_node_name_to_index_dict()[
            fe.basic_explanation.root_variable]][fe.cd_fact_pred_pos] >= fe.threshold)

def run_optimisation_1(fe: FactExplainer):

    L = fe.model.num_layers

    # First we compute a dictionary that maps each unary atom U_j(y) in \Gamma_i, represented by the pair (y,j), to a
    # "contribution": a pair (layer, value) representing the first layer (from the end) where this has a non-zero
    # contribution, and "value" is the value of this contribution.

    # Maps each variable to a quadruple (var, l, col, j) indicating the "parent" of each variable.
    successors_reverse = {v: k for k, v in fe.basic_explanation.successors.items()}

    # Extract the minimal sub-conjunction of Basic Rule which connects U_j(y) to the head variable
    def get_minimal_subconjunction(y,j):
        variable_progression = [y]  # maps each layer to the variable that ``carries'' U_j(y)'s contribution
        mini_dataset = [(y, type_pred, fe.internal_encoder.get_unary_predicate_for_index(j))]
        for l in range(1, L + 1):
            prev_var = variable_progression[-1]
            next_var, ell, col, _ = successors_reverse[prev_var]
            if l == ell:
                mini_dataset.append((prev_var, fe.internal_encoder.get_binary_predicate_for_colour(col), next_var))
                variable_progression.append(next_var)
            else:
                variable_progression.append(prev_var)
        return mini_dataset, variable_progression

    contributors_to_influence_dict = {}
    # Iterate over each fact U_j(y) in the Basic Rule
    for y in fe.basic_explanation.variable_to_level:
        for j in fe.basic_explanation.relevant_positions[(y, 0)]:
            mini_dataset, variable_progression = get_minimal_subconjunction(y,j)
            # Apply the GNN to this conjunction
            graph = fe.internal_encoder.encode_dataset(mini_dataset)
            node_index_dict = graph.get_node_name_to_index_dict()
            gr_dataset = Data(x=graph.features, edge_index=graph.edges, edge_type=graph.edge_colours).to(fe.device)
            gnn_output_gr, _ = fe.model(gr_dataset)
            # Compute its influence
            for l in range(L,-1,-1):
                contribution_score = 0
                for k in fe.basic_explanation.relevant_positions[(variable_progression[l],l)]:
                    contribution_score = (contribution_score +
                                          gnn_output_gr[node_index_dict[variable_progression[l]]][k])
                if contribution_score > 0:
                    contributors_to_influence_dict[(y,j)] = (l, contribution_score)
                    break
            if (y,j) not in contributors_to_influence_dict:
                contributors_to_influence_dict[(y, j)] = (0, 0)

    # Start trying rules, adding atoms in increasing order of contribution
    contributors_list = []
    for (y, j) in contributors_to_influence_dict:
        contributors_list.append((contributors_to_influence_dict[(y, j)], y, j))
    contributors_list = sorted(contributors_list)
    current_body_set = set()
    output_body = None
    while not output_body:
        if test_gr_dataset(current_body_set):
            output_body = current_body_set
        else:
            (inf, vy, pj) = contributors_list.pop()
            current_body_set = current_body_set.union(get_minimal_subconjunction(vy, pj))

    return list(output_body)



    # TODO ITP: maybe a clean-up step like in approximation 2 can help make improvements