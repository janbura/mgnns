import torch
from torch_geometric.data import Data
import numpy as np
import argparse
import os.path
import data_parser
import nodes
import datetime
import sys
from config import ExperimentConfig, EncoderType

from encoding_schemes import CanonicalEncoderDecoder, ICLREncoderDecoder
from utils import remove_redundant_atoms, type_pred, check

class FactExplainer:

    def explain_fact(self, ent1, ent2, ent3):

        # Encode fact
        fact = (ent1, ent2, ent3)
        if self.cfg.encoding_scheme == EncoderType.CANONICAL:
            cd_fact = fact
        else:
            cd_fact = self.iclr_encoder_decoder.get_canonical_equivalent(fact)
        (cd_fact_constant, _, cd_fact_predicate) = cd_fact
        cd_fact_node = nodes.const_node_dict[cd_fact_constant]
        # inside the class.
        if cd_fact_node not in self.node_to_gd_row_dict:
            sys.exit("Error: the encoded fact mentions a constant not in the encoded dataset.")
        cd_fact_gd_row = self.node_to_gd_row_dict[cd_fact_node]
        cd_fact_pred_pos = self.can_encoder_decoder.unary_pred_position_dict[cd_fact_predicate]
        # Sanity check: ensure the fact is a consequence of the model and the dataset
        assert  list(self.activations)[self.model.num_layers][cd_fact_gd_row][cd_fact_pred_pos] >= self.cfg.derivation_threshold, \
            "Error: the fact to be explained is not derived by the model on this dataset."

        # Compute Gamma_i
        gamma_i = self.compute_gamma_i(cd_fact_node, cd_fact_pred_pos)

        # Attempt abbreviation 1
        print("Attempting approximation 1...")
        # TODO: write abbrev 1

        # Attempt abbreviation 2
        print("Attempting approximation 2...")
        # TODO: write abbrev 2

        # SELECT FINAL RULE

        rule_body = []
        print("Length rule 1: {}".format(len(short_body_1)))
        print("Length rule 2: {}".format(len(short_body_2)))
        if not minimal:
            if short_body_2 and len(short_body_2) < len(short_body_1):
                print("Approximation 2 wins")
                rule_body = short_body_2
            else:
                print("Approximation 1 wins")
                rule_body = short_body_1
        else:
            if short_body_2 and len(short_body_2) < len(short_body_1):
                print("Approximation 2 wins")
                max_number_of_body_atoms = len(short_body_2)
            else:
                print("Approximation 1 wins")
                max_number_of_body_atoms = len(short_body_1)

            # Dictionary mapping each variables in gamma_i to its relevant positions (given by label of layer 0).
            total_variable_to_positions_dict = {}
            for (y, j) in contributors_to_influence_dict:
                if y not in total_variable_to_positions_dict:
                    total_variable_to_positions_dict[y] = {j}
                else:
                    total_variable_to_positions_dict[y].add(j)

            def get_successors(partial_variable_to_positions_dict, conjunction_form):
                successors = []
                for y in total_variable_to_positions_dict:
                    if y not in partial_variable_to_positions_dict:
                        for j in total_variable_to_positions_dict[y]:
                            new_successor = partial_variable_to_positions_dict.copy()
                            new_successor[y] = {j}
                            new_conjunction_form = conjunction_form.union(get_conjunction_for_contributor(y, j))
                            successors.append((len(new_conjunction_form), new_successor, new_conjunction_form))
                    else:
                        for j in total_variable_to_positions_dict[y]:
                            if j not in partial_variable_to_positions_dict[y]:
                                new_successor = partial_variable_to_positions_dict.copy()
                                new_successor[y].add(j)
                                new_conjunction_form = conjunction_form.union(get_conjunction_for_contributor(y, j))
                                successors.append((len(new_conjunction_form), new_successor, new_conjunction_form))
                return successors

            frontier = get_successors({}, set())
            while frontier:
                # Sort in decreasing order of cost
                frontier = sorted(frontier, reverse=True)
                (score, dictionary, conjunction) = frontier.pop()
                (gr_features, node_to_gr_row_dict, gr_edge_list, gr_colour_list) = can_encoder_decoder.encode_dataset(
                    conjunction)
                gr_dataset = Data(x=gr_features, edge_index=gr_edge_list, edge_type=gr_colour_list).to(device)
                gnn_output_gr = model(gr_dataset)
                if (gnn_output_gr[node_to_gr_row_dict[nodes.const_node_dict[x1]]][cd_fact_pred_pos] >=
                        cfg.derivation_threshold):
                    frontier = None
                    rule_body = conjunction
                else:
                    frontier = list(set(frontier).union(set(get_successors(dictionary, conjunction))))

        rule_body = remove_redundant_atoms(rule_body)

        # Correctness check: check that each atom is grounded in the canonical dataset via \nu
        for (s, p, o) in rule_body:
            if p == type_pred:
                constant = nodes.node_const_dict[nu_variable_to_node_dict[s]]
                assert (constant, p, o) in cd_dataset, "ERROR: This rule does not unify with the dataset. Bug."
            else:
                origin_constant = nodes.node_const_dict[nu_variable_to_node_dict[s]]
                dest_constant = nodes.node_const_dict[nu_variable_to_node_dict[o]]
                assert (origin_constant, p, dest_constant) in cd_dataset, \
                    "ERROR: the extracted rule does not unify with the dataset. Bug."
        # Correctness check: ensure that the rule is captured.
        if not rule_body:
            gr_features = torch.FloatTensor(np.zeros((1, model.layer_dimension(0))))
            gr_edge_list = torch.LongTensor(2, 0)
            gr_dataset = Data(x=gr_features, edge_index=gr_edge_list, edge_type=torch.LongTensor([])).to(device)
            gnn_output_gr = model(gr_dataset)
            assert gnn_output_gr[0][cd_fact_pred_pos] >= cfg.derivation_threshold, \
                "ERROR: the extracted rule seems not to be captured by the model. This means there is a bug."
        else:
            (gr_features, node_to_gr_row_dict, gr_edge_list, gr_colour_list) = can_encoder_decoder.encode_dataset(rule_body)
            gr_dataset = Data(x=gr_features, edge_index=gr_edge_list, edge_type=gr_colour_list).to(device)
            gnn_output_gr = model(gr_dataset)
            assert (gnn_output_gr[node_to_gr_row_dict[nodes.const_node_dict[x1]]][cd_fact_pred_pos] >
                    cfg.derivation_threshold), \
                "ERROR: the extracted rule seems not to be captured by the model. This means there is a bug."

        # Unfold extracted rules with the encoder's rules
        if cfg.encoding_scheme == EncoderType.ICLR22:
            (rule_body, can_variable_to_data_variable,
             top_facts) = iclr_encoder_decoder.unfold(rule_body, cd_fact_predicate)
            # Process top_facts
            for pair in top_facts:
                [y1, y2] = list(pair)
                cvar_list = iclr_encoder_decoder.find_canonical_variable(can_variable_to_data_variable, y1, y2)
                if len(cvar_list) == 1:
                    # y1 and y2 come from a binary canonical variable
                    ab = nodes.node_const_dict[nu_variable_to_node_dict[cvar_list[0]]]
                    a, b = iclr_encoder_decoder.term_tuple_dict[ab]
                    ba = iclr_encoder_decoder.pair_term_dict[(b, a)]
                else:
                    # y1 and y2 come from unary canonical variables
                    a = nu_variable_to_node_dict[cvar_list[0]]
                    b = nu_variable_to_node_dict[cvar_list[1]]
                    ab = iclr_encoder_decoder.pair_term_dict[(a, b)]
                    ba = iclr_encoder_decoder.pair_term_dict[(b, a)]
                fact_found = False
                for (a1, a2, a3) in cd_dataset:
                    if not fact_found and a2 == type_pred:
                        if a1 == ab:
                            fact_found = True
                            rule_body.append((y1, iclr_encoder_decoder.unary_canonical_to_input_predicate_dict[a3], y2))
                        elif a1 == ba:
                            fact_found = True
                            rule_body.append((y2, iclr_encoder_decoder.unary_canonical_to_input_predicate_dict[a3], y1))
                assert fact_found

        # Write the rule
        body_atoms = []
        rule_body = set(rule_body)  # Remove duplicates
        for (s, p, o) in rule_body:
            if p == type_pred:
                body_atoms.append("<{}>[?{}]".format(o, s))
            else:
                body_atoms.append("<{}>[?{},?{}]".format(p, s, o))
        if cfg.encoding_scheme == EncoderType.CANONICAL:
            head = "<{}>[?X1]".format(cd_fact_predicate)
        else:
            if iclr_encoder_decoder.input_predicate_to_arity[
                iclr_encoder_decoder.unary_canonical_to_input_predicate_dict[cd_fact_predicate]] == 1:
                head = "<{}>[?X1]".format(iclr_encoder_decoder.unary_canonical_to_input_predicate_dict[cd_fact_predicate])
            else:
                head = "<{}>[?X1,?X2]".format(
                    iclr_encoder_decoder.unary_canonical_to_input_predicate_dict[cd_fact_predicate])
        rule = head + " :- " + ", ".join(body_atoms) + " .\n"

        output_file.write("{}\n".format(fact))
        output_file.write(rule + '\n')

    def compute_gamma_i(self, cd_fact_node, cd_fact_pred_pos):

        gamma_i = []

        L = self.model.num_layers
        variable_counter = 1
        x1 = "X1"
        # The following two dictionaries capture a substitution \nu of variables in the rule to terms that unify with them (expressed as nodes)
        nu_variable_to_node_dict = {x1: cd_fact_node}
        nu_node_to_variable_dict = {cd_fact_node: x1} # TODO: confused. Wouldn't there be more than one?
        # We construct a tree where nodes are variables of Gamma_i. Each node at depth ell has a label for each layer from \ell to 0.
        # The label is the paper's \mu: maps a variable and layer to the list of relevant positions.
        labels = {(x1, L): {cd_fact_pred_pos}}
        # Each node has a successor indexed by (layer l, colour c_i, position j) for the variable mapped to the constant that contributes
        # with the maximum value aggregating over c_i in position j in layer l.
        successors = {}  # (var, layer, colour, pos) -> var
        # The predecessor dictionary helps find the parent of a given node, and the labels of the edge.
        predecessors = {}  # (var, layer) -> (var, colour, pos)

        # Iterate backwards over all layers
        variables_for_next_layer = [x1]
        for l in range(L, 0, -1):
            variables_for_this_layer = variables_for_next_layer.copy() # This ensures we never lose a variable
            for y in variables_for_this_layer:
                # labels for this variable, next layer
                labels[(y, l - 1)] = set()
                for j in range(self.model.layer_dimension(l - 1)):
                    for i in labels[(y, l)]:
                        if self.model.matrix_A(l)[i][j].item() > 0  and self.activations[l - 1][self.node_to_gd_row_dict[nu_variable_to_node_dict[y]]][j].item() > 0:
                            labels[(y, l - 1)].add(j)
                            if l == 1:
                                gamma_i.append( (y, type_pred, self.can_encoder_decoder.position_unary_pred_dict[j]))
                            break
                # children variables
                for colour in self.can_encoder_decoder.colours:
                    edge_mask = self.gd_edge_colour_list == colour
                    colour_edges = self.gd_edge_list[:, edge_mask]
                    neighbours = colour_edges[:, colour_edges[1] == self.node_to_gd_row_dict[nu_variable_to_node_dict[y]]][0].tolist()
                    for j in range(self.model.layer_dimension(l - 1)):
                        # Check if the value in position j of the aggregation is used in the matrix multiplications.
                        relevant = False
                        for i in labels[(y, l)]:
                            if self.model.matrix_B(l, colour)[i][j].item() > 0:
                                # Find the neighbour that contributes maximum to aggregation
                                max_neighbour = None
                                max_value = 0
                                for neighbour in neighbours:
                                    element = self.activations[l - 1][neighbour][j].item()
                                    # TODO: break ties using number of neighbours
                                    if element > max_value:
                                        max_neighbour = neighbour
                                        max_value = element
                                if max_neighbour is not None:
                                    variable_counter += 1
                                    z = "X" + str(variable_counter)
                                    nu_variable_to_node_dict[z] = self.gd_row_to_node_dict[max_neighbour]
                                    nu_node_to_variable_dict[self.gd_row_to_node_dict[max_neighbour]] = z
                                    successors[(y, l, colour, j)] = z
                                    predecessors[(z, l)] = (y, colour, j)
                                    variables_for_next_layer.append(z)
                                    labels[(z, l - 1)] = {j}
                                    gamma_i.append((y, self.can_encoder_decoder.colour_binary_pred_dict[colour], z))
                                break

        # Sanity check: ensure Gamma_i is sound
        (gr_features, node_to_gr_row_dict, gr_edge_list, gr_colour_list) = self.can_encoder_decoder.encode_dataset(gamma_i)
        gnn_output_gr, _  = self.model(Data(x=gr_features, edge_index=gr_edge_list, edge_type=gr_colour_list).to(self.device))
        assert (gnn_output_gr[node_to_gr_row_dict[nodes.const_node_dict[x1]]][cd_fact_pred_pos] >=
                self.cfg.derivation_threshold), "ERROR: Gamma_i is not sound. This should not happen; there's a bug."

    def __init__(self, cfg: ExperimentConfig, graph, scored_facts, activations, minimal=False, n_explanations=10):

        self.cfg = cfg
        self.graph =  graph
        self.scored_facts = scored_facts
        self.activations = activations
        self.n_explanations = n_explanations

        ed = self.cfg.get_exp_dir()

        self.can_encoder_decoder = CanonicalEncoderDecoder(check(ed / 'canonical_encoder.tsv', "Canonical encoding"))
        self.iclr_encoder_decoder = None

        if self.cfg.encoding_scheme == EncoderType.CANONICAL:
            self.cd_dataset = self.graph
        elif cfg.encoding_scheme == EncoderType.ICLR22:
            self.iclr_encoder_decoder = ICLREncoderDecoder(check(ed / "iclr22_encoder.tsv", "ICLR encoding"))
            self.cd_dataset = self.iclr_encoder_decoder.encode_dataset(self.graph)
        else:
            raise ValueError(f"Script does not support encoding scheme: {self.cfg.encoding_scheme}")

         # gd stands for "graph dataset"
        gd_features, node_to_gd_row_dict, gd_edge_list, gd_edge_colour_list = self.can_encoder_decoder.encode_dataset(self.cd_dataset)
        # TODO assert gd_features equals fl0
        self.node_to_gd_row_dict = node_to_gd_row_dict
        self.gd_edge_list = gd_edge_list
        self.gd_edge_colour_list = gd_edge_colour_list
        self.gd_row_to_node_dict = {self.node_to_gd_row_dict[node]: node for node in self.node_to_gd_row_dict}
        self.output_file = open(ed / "explanations.txt", 'w')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(check(ed / "model.pt","Model"), weights_only=False).to(self.device)

        for ent1, ent2, ent3 in list(self.scored_facts[:self.n_explanations]):
            self.explain_fact(ent1, ent2, ent3)