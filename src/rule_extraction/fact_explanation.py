import numpy as np
from src.model.cd_graph import TraceCollector
from src.encodings.canonical import CanonicalEncoderDecoder
from src.encodings.noncanonical.noncanonical import NonCanonicalEncoder
from src.rule_extraction.tree_shaped_conjunction import TreeShapedConjunction, Variable
from src.utils.utils import TYPE_PRED, backpropagate_relevance
from src.utils.bitset import BitSet

# This class represents the Gamma_i in the papers. It is based on the basic rule extraction algorithm, optimised
# with information about sparsity in the model's matrices.

class BasicExplanation:

    def __init__(self, fe):

        # We initialise the conjunction as an empty tree-shaped conjunction
        self.conjunction = TreeShapedConjunction(fe.internal_encoder.get_n_binary_predicates)
        L = fe.model.num_layers
        initial_mask = BitSet.from_subset(fe.internal_encoder.get_n_unary_predicates(), {fe.cd_fact_pred_pos})
        root_variable = Variable(level=L)
        self.conjunction.root_node = root_variable
        # Maps a Variable to the (index of the) constant that grounds it. This is the \nu mapping in the paper
        self.var_const_idx = {root_variable: fe.cd_fact_const_index}
        # Maps a (Variable, Layer) to the relevant Feature Mask. This is the paper's \mu.
        self.var_layer_mask = {(root_variable, L): initial_mask}

        # Paper's algorithm for constructing the conjunction
        for l in range(L, 0, -1): # Iterate backwards over all layers from L to 1 (both inclusive).
            for var in self.conjunction.walk():
                self.var_layer_mask[(var, l - 1)] = backpropagate_relevance(self.var_layer_mask[(var,l)],
                                                                            fe.model.matrix_A(l),
                                                                            fe.activations[l - 1][self.var_const_idx[var]])
                # Introduce children new variables for var and define their relevant positions
                for colour in fe.internal_encoder.get_colours():
                    edge_mask = fe.trace.cd_graph.edge_colours == colour
                    colour_edges = fe.trace.cd_graph.edges[:, edge_mask]
                    neighbours = colour_edges[:, colour_edges[1] == self.var_const_idx[var]][0].tolist()
                    neighbour_vectors = np.array([fe.activations[l - 1][neighbour] for neighbour in neighbours])
                    for j in backpropagate_relevance(self.var_layer_mask[(var,l)],
                                                     fe.model.matrix_B(l, colour)
                                                     ).elements():
                        # Find the neighbour that contributes maximum to aggregation
                        best_idx = np.argmax(neighbour_vectors[:, j])
                        if neighbour_vectors[best_idx, j] > 0:
                            new_variable = Variable(BitSet.from_subset(fe.model.layer_dimension(l-1),{}),level=l-1)
                            var.children[(l, colour, j)] = new_variable
                            self.var_const_idx[new_variable] = neighbours[best_idx]
                            self.var_layer_mask[(new_variable, l - 1)] = (
                                BitSet.from_subset(fe.model.layer_dimension(l - 1), {j}))

        for var in self.conjunction.walk(): # Add the atoms for the feature vectors in layer 0
            var.features = self.var_layer_mask[(var,0)]
            # Needs to be done separately, otherwise this is not done to the new variables added!

        # TODO: Redo this
        # (gr_features, node_to_gr_row_dict, gr_edge_list, gr_colour_list) = self.can_encoder_decoder.encode_dataset(gamma_i)
        # gnn_output_gr, _  = self.model(Data(x=gr_features, edge_index=gr_edge_list, edge_type=gr_colour_list).to(self.device))
        # assert (gnn_output_gr[node_to_gr_row_dict[nodes.const_node_dict[x1]]][cd_fact_pred_pos] >=
        #         self.cfg.derivation_threshold), "ERROR: Gamma_i is not sound. This should not happen; there's a bug."
        # TODO: also check that the variable levels match the \mu and the trees.

# This class manages the explanation of a fact derived by the GNN

class FactExplainer:

    def __init__(self, device, fact: tuple[str,str,str], model, threshold, trace: TraceCollector,
                 external_encoder: NonCanonicalEncoder, internal_encoder: CanonicalEncoderDecoder, minimal=False):

        self.device = device
        self.model = model
        self.threshold = threshold
        self.external_encoder = external_encoder
        self.internal_encoder = internal_encoder
        self.activations = [trace.fl2,trace.fl1,trace.fl0]
        self.trace = trace
        self.fact = fact
        self.ent1, self.ent2, self.ent3 = fact
        self.node_to_index = {node: i for i, node in enumerate(trace.cd_graph.node_names)} # Helpful dictionary
        self.cd_ent1, _, self.cd_ent3 = self.external_encoder.get_canonical_equivalent(self.fact)
        self.cd_fact_const_index = self.node_to_index[self.cd_ent1]
        self.cd_fact_pred_pos = self.internal_encoder.unary_pred_position_dict[self.cd_ent3]
        # Sanity check: ensure the fact is a consequence of the model and the dataset
        assert self.activations[0][self.cd_fact_const_index][self.cd_fact_pred_pos] >= threshold, \
            "Error: the fact to be explained is not derived by the model on this dataset."

        print("Computing Gamma_i")
        self.basic_explanation = BasicExplanation(self)
        rule_body = self.basic_explanation.conjunction
        print("Length Gamma_i: {}".format(len(rule_body)))

        # TODO: Refactor all 3 optimisations
        # print("Attempting approximation 1...")
        # optimised_body_1 = run_optimisation_1(self)
        # print("Length rule 1: {}".format(len(optimised_body_1)))

        # print("Attempting approximation 2...")
        # optimised_body_2 = run_optimisation_2(self)
        # print("Length rule 2: {}".format(len(optimised_body_2)))

        #if len(optimised_body_2) < len(optimised_body_1):
        #    print("Approximation 2 wins")
        #    rule_body = optimised_body_2
        # else:
        #    print("Approximation 1 wins")

        # Optimisation 3 used to go here and was applied to the best of 1 or 2

        # Unfold into body via external encoder/decoder
        head_is_binary = True
        if self.ent2 is TYPE_PRED:
            head_is_binary = False
        # This converts a TreeShapedConjunction into a simple list of triples, plus a list of head variables
        rule_body, head_variables = external_encoder.unfold(can_conj=rule_body,
                                                            head_is_binary=head_is_binary,
                                                            internal_encoder=internal_encoder)

        # TODO: rule-writing is a separate responsibility so probably can go somewhere else.
        # Write the rule
        body_atoms = []
        rule_body = set(rule_body)  # Remove duplicates
        for (s, p, o) in rule_body:
            if p == TYPE_PRED:
                body_atoms.append("<{}>[?{}]".format(o, s))
            else:
                body_atoms.append("<{}>[?{},?{}]".format(p, s, o))
        if head_is_binary:
            head =  "<{}>[?{},?{}]".format(self.ent2,head_variables[0],head_variables[1])
        else:
            head = "<{}>[?{}]".format(self.ent3,head_variables[0])
        self.rule = head + " :- " + ", ".join(body_atoms) + " .\n"

