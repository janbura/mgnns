import torch
from sympy.physics.units import second
from torch.fx.experimental.unification.unification_tools import first

from src.tree_shaped_conjunction import TreeShapedConjunction
from utils import type_pred
from bidict import bidict
from cd_graph import CDGraph
from collections import defaultdict
import sys
from abc import ABC, abstractmethod
from tree_shaped_conjunction import Variable

ineq_pred = "owl:differentFrom"

class CanonicalEncoderDecoder:

    # If the signature lacks unary or binary predicates, we add dummies to the signature (but not use them in any facts)
    DUMMY_PRED = "DUMMY_PRED"
    DUMMY_COL = "DUMMY_COL"

    def __init__(self, load_from_document=None, unary_predicates=None, binary_predicates=None):

        self.unary_pred_position_dict: bidict[str,int] = bidict()
        self.binary_pred_colour_dict: bidict[str,int] = bidict()

        if load_from_document is not None:
            # We trust the document!
            for line in open(load_from_document, 'r').readlines():
                arity, position, predicate = line.split()
                if arity == "UNARY":
                    self.unary_pred_position_dict[predicate] = int(position)
                elif arity == "BINARY":
                    self.binary_pred_colour_dict[predicate] = int(position)
                else:
                    sys.exit("ERROR: line not recognised: {}".format(line))
        else:
            for i, predicate in enumerate(unary_predicates):
                self.unary_pred_position_dict[predicate] = i
            if not self.unary_pred_position_dict:
                self.unary_pred_position_dict[self.DUMMY_PRED] = 0 # Add a single dummy unary predicate
            for i, predicate in enumerate(binary_predicates):
                self.binary_pred_colour_dict[predicate] = i
            if not self.binary_pred_colour_dict:
                self.binary_pred_colour_dict[self.DUMMY_COL] = 0 # Add a single dummy colour

    def get_colours(self):
        return self.binary_pred_colour_dict.values()

    def get_unary_predicate_for_index(self,i):
        return self.unary_pred_position_dict.inverse[i]

    def get_binary_predicate_for_colour(self,i):
        return self.binary_pred_colour_dict.inverse[i]

    def get_n_unary_predicates(self):
        return len(self.unary_pred_position_dict)

    def get_n_binary_predicates(self):
        return len(self.binary_pred_colour_dict)

    def save_to_file(self, target_file):
        output = open(target_file, 'w')
        for i in self.unary_pred_position_dict.inverse:
            output.write("{}\t{}\t{}\n".format("UNARY", i, self.unary_pred_position_dict.inverse[i]))
        for i in self.binary_pred_colour_dict.inverse:
            output.write("{}\t{}\t{}\n".format("BINARY", i, self.binary_pred_colour_dict.inverse[i]))
        output.close()

    # Given a (col,d)-dataset, returns its resulting (col,d)-graph
    def encode_dataset(self, dataset):

        nodename_feature_dict = defaultdict(lambda: torch.zeros(delta, dtype=torch.float))
        edges = set()
        delta = len(self.unary_pred_position_dict)
        col_size = len(self.binary_pred_colour_dict)

        for RDF_triple in dataset:
            if RDF_triple[1] == type_pred:  # Fact of form C(a), written (a type C)
                if RDF_triple[2] not in self.unary_pred_position_dict:
                    sys.exit(f"Predicate {RDF_triple[2]} not in the list of unary predicates recognised by this encoder.")
                nodename_feature_dict[RDF_triple[0]][self.unary_pred_position_dict[RDF_triple[2]]] = 1
            else:  # Fact of form R(a,b), written (a R b)
                if RDF_triple[1] not in self.binary_pred_colour_dict:
                    sys.exit(f"Predicate {RDF_triple[1]} not in the list of binary predicates recognised by this encoder.")
                nodename_feature_dict[RDF_triple[0]] = torch.zeros(delta, dtype=torch.float)
                nodename_feature_dict[RDF_triple[2]] = torch.zeros(delta, dtype=torch.float)
                edges.add((RDF_triple[0], RDF_triple[2], RDF_triple[1]))

        features = torch.FloatTensor(torch.stack(list(nodename_feature_dict.values())))
        assert features.shape[1] == delta
        node_names = list(nodename_feature_dict.keys()) # Correctness of this relies on dictionaries being ordered.
        edge_list = []
        edge_colour_list = []
        for (oc, dc, pred) in edges:
            edge_list.append([node_names.index(oc),node_names.index(dc)])
            edge_colour_list.append(self.binary_pred_colour_dict[pred])

        return CDGraph(col_size=col_size, delta=len(self.unary_pred_position_dict),
                       features=features, edges= torch.LongTensor(torch.LongTensor(edge_list).t().contiguous()),
                       edge_colours=torch.LongTensor(edge_colour_list), node_names=node_names)

    # Returns a dictionary where the keys are cd_facts and the values are their scores
    def decode_graph(self, cd_graph: CDGraph, threshold):

        facts_scores_dict = {}
        for i, j in torch.nonzero(cd_graph.features > threshold).tolist():
            facts_scores_dict[(cd_graph.node_names[i], type_pred, self.unary_pred_position_dict.inverse[j])] = (
                cd_graph.features[i,j].item())

        return facts_scores_dict

class NonCanonicalEncoder(ABC):

    canonical_unary_predicates = list
    canonical_binary_predicates = list

    @abstractmethod
    def encode_dataset(self, dataset: set[tuple], **kwargs) -> set[tuple]:
        pass

    @abstractmethod
    def decode_dataset(self, dataset: set[tuple]) -> set[tuple]:
        pass

    @abstractmethod
    def decode_fact(self, s: str, p:str, o:str) -> tuple[str, str, str]:
        pass

    @abstractmethod
    def get_canonical_equivalent(self, fact: tuple[str, str, str]) -> tuple[str, str, str]:
        pass

    @abstractmethod
    def unfold(self, can_conj: TreeShapedConjunction, head_is_binary: bool, internal_encoder: CanonicalEncoderDecoder):
        pass


class IdentityEncoderDecoder(NonCanonicalEncoder):

    def __init__(self, load_from_document=None, unary_predicates=None, binary_predicates=None):
        self.canonical_unary_predicates = []
        self.canonical_binary_predicates = []
        if load_from_document is not None:
            for line in open(load_from_document, 'r').readlines():
                predicate, _, arity = line.split() # predicates are duplicate so we ignore the second
                arity = int(arity)
                if arity == 1:
                    self.canonical_unary_predicates.append(predicate)
                else:
                    self.canonical_binary_predicates.append(predicate)
        else:
            self.canonical_binary_predicates = binary_predicates
            self.canonical_unary_predicates = unary_predicates

    def encode_dataset(self, dataset, **kwargs):
        return dataset

    def decode_dataset(self, dataset):
        return dataset

    def decode_fact(self, s, p, o):
        return s, p, o

    # Save format (predicate \t new_predicate \t arity)
    def save_to_file(self, target_file):
        output = open(target_file, 'w')
        for predicate in self.canonical_unary_predicates:
            output.write("{}\t{}\t{}\n".format(predicate, predicate, 1))
        for predicate in self.canonical_binary_predicates:
            output.write("{}\t{}\t{}\n".format(predicate, predicate, 2))
        output.close()

    def get_canonical_equivalent(self, fact):
        s, p, o = fact
        return s, p, o

    def unfold(self, can_conj: TreeShapedConjunction, head_is_binary: bool, internal_encoder: CanonicalEncoderDecoder):

        data_conj = []  # Not necessarily tree-shaped

        # Data variable list
        data_var_prefix = "X"
        data_var_counter = 0
        def new_variable():
            nonlocal data_var_counter
            data_var_counter += 1
            return data_var_prefix + str(data_var_counter)

        if can_conj.is_empty():
            return data_conj  # Return the empty conjunction if the tree-shaped conjunction is empty.

        def unfold_variable(can_var: Variable, data_var: str):
            for feat in can_var.get_feature_list():
                can_predicate = internal_encoder.unary_pred_position_dict.inverse[feat]
                data_conj.append((data_var, type_pred, can_predicate)) # canonical predicate is data predicate
            for (col, _), child_var in can_var.children:
                new_data_var = new_variable()
                bin_predicate = internal_encoder.binary_pred_colour_dict.inverse[col]
                data_conj.append((data_var, bin_predicate, new_data_var))
                unfold_variable(child_var, new_data_var)

        root_variables = [data_var_prefix + str(data_var_counter)]
        unfold_variable(can_conj.root_node, root_variables[0])

        return data_conj, root_variables

class ICLREncoderDecoder(NonCanonicalEncoder):

    # Fresh predicates that correspond to colours c1, c2, c3, c4 in the paper. Abbreviations match paper names
    col1 = "binary-pred-1"
    col2 = "binary-pred-2"
    col3 = "binary-pred-3"
    col4 = "binary-pred-4"


    def __init__(self, load_from_document=None, unary_predicates=None, binary_predicates=None):
        self.canonical_binary_predicates = [self.col1, self.col2, self.col3, self.col4]
        self.input_predicate_to_unary_canonical_dict = bidict()
        self.input_predicate_to_arity = {}
        if load_from_document is not None:
            for line in open(load_from_document, 'r').readlines():
                input_predicate, canonical_predicate, arity = line.split()
                self.input_predicate_to_unary_canonical_dict[input_predicate] = canonical_predicate
                self.input_predicate_to_arity[input_predicate] = int(arity)
        else:
            assert set(unary_predicates).isdisjoint(set(binary_predicates)) # Sanity check
            for pred in unary_predicates:
                self.input_predicate_to_unary_canonical_dict[pred] = pred
                self.input_predicate_to_arity[pred] = 1
            for pred in binary_predicates:
                self.input_predicate_to_unary_canonical_dict[pred] = "unary-for-{}".format(pred)
                self.input_predicate_to_arity[pred] = 2
        # Maps pairs of constants to a new single term
        self.pair_term_dict = bidict()
        self.canonical_unary_predicates = list(self.input_predicate_to_unary_canonical_dict.inverse.keys())

    # Save format (predicate \t new_predicate \t arity)
    def save_to_file(self, target_file):
        output = open(target_file, 'w')
        for input_predicate in self.input_predicate_to_unary_canonical_dict:
            output.write("{}\t{}\t{}\n".format(input_predicate,
                                               self.input_predicate_to_unary_canonical_dict[input_predicate],
                                               self.input_predicate_to_arity[input_predicate]))
        output.close()

    def term_for_pair(self, pair):
        if pair not in self.pair_term_dict:
            self.pair_term_dict[pair] = "term-for-{}-{}".format(pair[0], pair[1])
        return self.pair_term_dict[pair]

    def encode_fact(self, fact):
        encoded_dataset = []
        s, p, o = fact
        if p == type_pred:
            encoded_dataset.append((s, p, self.input_predicate_to_unary_canonical_dict[o]))
        else:
            a = s    # We rename to the paper's notation to make the code easier to read and write
            b = o
            ab = self.term_for_pair((a, b))
            ba = self.term_for_pair((b, a))
            encoded_dataset.append((ab, type_pred, self.input_predicate_to_unary_canonical_dict[p]))
            encoded_dataset.extend([(a, self.col1, ab), (ab, self.col1, a), (b, self.col1, ba), (ba, self.col1, b)])
            encoded_dataset.extend([(b, self.col2, ab), (ab, self.col2, b), (a, self.col2, ba), (ba, self.col2, a)])
            encoded_dataset.extend([(ab, self.col3, ba), (ba, self.col3, ab)])
            encoded_dataset.extend([(a, self.col4, b), (b, self.col4, a)])
        return encoded_dataset

    # Returns a new dataset over the new signature
    def encode_dataset(self, dataset, use_dummy_constants=False):

        encoded_dataset = []
        for fact in dataset:
            encoded_dataset.extend(self.encode_fact(fact))

        if use_dummy_constants:
            # Extract all constants from encoded_dataset. Less efficient than doing it on-the-fly, but it's cleaner code
            constants = set()
            for s, p, o in encoded_dataset:
                constants.add(s)
                if p != type_pred:
                    constants.add(o)
            for a in ['#', '##']:
                for b in constants:
                    ab = self.term_for_pair((a, b))
                    ba = self.term_for_pair((b, a))
                    encoded_dataset.extend([(a, self.col1, ab), (ab, self.col1, a), (b, self.col1, ba), (ba, self.col1, b)])
                    encoded_dataset.extend([(b, self.col2, ab), (ab, self.col2, b), (a, self.col2, ba), (ba, self.col2, a)])
                    encoded_dataset.extend([(ab, self.col3, ba), (ba, self.col3, ab)])
                    encoded_dataset.extend([(a, self.col4, b), (b, self.col4, a)])

        return encoded_dataset

    def get_canonical_equivalent(self, fact):
        (s, p, o) = fact
        if p == type_pred:
            return s, p, self.input_predicate_to_unary_canonical_dict[o]
        else:
            ab = self.term_for_pair((s, o))
            return ab, type_pred, self.input_predicate_to_unary_canonical_dict[p]

    def decode_dataset(self, canonical_dataset):
        return {self.decode_fact(s, p, o) for s, p, o in canonical_dataset}

    def decode_fact(self, s, p, o):
        # All facts in an encoded dataset are unary; binary facts should not be decoded.
        assert(p == type_pred)
        if s in self.pair_term_dict.values():
            a, b = self.pair_term_dict.inverse[s]
            return a, self.input_predicate_to_unary_canonical_dict.inverse[o], b
        else:
            # The node is a for a single constant.
            return s, type_pred, self.input_predicate_to_unary_canonical_dict.inverse[o]

    def is_term_for_pair(self, const):
        return const in self.pair_term_dict.inverse

    # Maps a canonical predicate to the arity of the original predicate, either 1 or 2
    def original_arity(self, canonical_predicate):
        return self.input_predicate_to_arity[self.input_predicate_to_unary_canonical_dict.inverse[canonical_predicate]]

    # This function takes a tree-shaped conjunction expressed in the Canonical Signature and returns a conjunction in
    # the Data Signature.
    # We traverse the canonical conjunction unfolding as we go.
    # Note that we only unfold canonic unary atoms, (which turn into either unary or binary data atoms)
    # Canonical binary atoms are used only to choose the correct unfolding function.
    # Returns the unfolded conjunction and a list of the variables in the head (might be one or two)
    def unfold(self, can_conj: TreeShapedConjunction, head_is_binary: bool, internal_encoder: CanonicalEncoderDecoder):

        data_conj = [] # Not necessarily tree-shaped

        # For binary atoms involving colour c4, we don't know which predicate generated this connection, so instead we
        # return a "top" predicate, to be interpreted as a two constants being connected.
        top_predicate = "top-pred"

        # Data variable list
        data_var_prefix = "X"
        data_var_counter = 0
        def new_variable():
            nonlocal data_var_counter
            data_var_counter += 1
            return data_var_prefix + str(data_var_counter)

        if can_conj.is_empty():
            return data_conj # Return the empty conjunction if the tree-shaped conjunction is empty.

        # Unfold unary canonical atom that unifies with a canonical constant for a pair of data_constants.
        def unfold_variable_for_pair(can_var: Variable, first_data_var: str=None, second_data_var: str=None):
            # First, add the relevant atoms.
            assert first_data_var is not None or second_data_var is not None # We should know at least one of them.
            if first_data_var is None:
                first_data_var = new_variable()
            if second_data_var is None:
                second_data_var = new_variable()
            for feat in can_var.get_feature_list():
                can_predicate = internal_encoder.unary_pred_position_dict.inverse[feat]
                data_predicate = self.input_predicate_to_unary_canonical_dict.inverse[can_predicate]
                data_conj.append((first_data_var, data_predicate, second_data_var))
            # Next, unfold children
            for (col, _), child_var in can_var.children:
                if col==self.col1:
                    # This is a binary node, and the edge is c1, so target must be unary node matching first var
                    unfold_variable_for_single(child_var, first_data_var)
                elif col==self.col2:
                    # Analogous to above
                    unfold_variable_for_single(child_var, second_data_var)
                    # Still a pair, but order must be reversed
                elif col==self.col3:
                    unfold_variable_for_pair(child_var, second_data_var, first_data_var)
                else:
                    raise ValueError(f"Binary fact in canonical atom uses predicate {col} which is not valid.")

        # Unfold unary canonical atom that unifies with a constant in the original signature (single).
        def unfold_variable_for_single(can_var, data_var: str):
            # First, add the relevant atoms.
            for feat in can_var.get_feature_list():
                can_predicate = internal_encoder.unary_pred_position_dict.inverse[feat]
                data_predicate = self.input_predicate_to_unary_canonical_dict.inverse[can_predicate]
                data_conj.append((data_var, type_pred, data_predicate))
            # Next, unfold children
            for (col, _), child_var in can_var.children:
                if col == self.col1:
                    # This is a unary node, and the edge is c1, so target must be a binary node matching first var
                    unfold_variable_for_pair(child_var,first_data_var=data_var)
                elif col == self.col2:
                    # Analogous to above
                    unfold_variable_for_pair(child_var, second_data_var=data_var)
                elif col == self.col4:
                    # This must be another unary node.
                    # A top fact must be added to reflect these two variables are connected.
                    nonlocal data_var_counter
                    new_data_var = new_variable()
                    unfold_variable_for_single(child_var,new_data_var)
                else:
                    raise ValueError(f"Binary fact in canonical atom uses predicate {col} which is not valid.")

        root_variables = [data_var_prefix + str(data_var_counter)]
        if head_is_binary:
            second_root_data_var = new_variable()
            root_variables.append(second_root_data_var)
            unfold_variable_for_pair(can_conj.root_node, root_variables[0], root_variables[1])
        else:
            unfold_variable_for_single(can_conj.root_node, root_variables[0])

        return data_conj, root_variables


    # This is a rather specific function. Given two data variables y1 and y2, this returns a single variable if y1 y2
    # correspond to a single canonical variable y, and two variables if they correspond to a canonical variable each.
    def find_canonical_variable(self, can_variables_to_data_variables, y1, y2):
        binary = None
        unary_y1 = None
        unary_y2 = None
        for cvar in can_variables_to_data_variables:
            if len(can_variables_to_data_variables[cvar]) == 2 and (
                    (can_variables_to_data_variables[cvar][0] == y1 and can_variables_to_data_variables[cvar][1] == y2) or
                    (can_variables_to_data_variables[cvar][0] == y2 and can_variables_to_data_variables[cvar][1] == y1)):
                binary = cvar
            elif len(can_variables_to_data_variables[cvar]) == 1:
                if can_variables_to_data_variables[cvar][0] == y1:
                    unary_y1 = cvar
                elif can_variables_to_data_variables[cvar][0] == y2:
                    unary_y2 = cvar
        if binary:
            return [binary]
        elif unary_y1 and unary_y2:
            return [unary_y1, unary_y2]
        else:
            raise Exception("Error: data variables {} and {} do not seem to match any canonical variable. Bug.".format(y1,y2))
