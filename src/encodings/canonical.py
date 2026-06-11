import torch
from src.utils.utils import TYPE_PRED
from bidict import bidict
from src.model.cd_graph import CDGraph
from collections import defaultdict
import sys

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
            if RDF_triple[1] == TYPE_PRED:  # Fact of form C(a), written (a type C)
                if RDF_triple[2] not in self.unary_pred_position_dict:
                    sys.exit(f"Predicate {RDF_triple[2]} not in the list of unary predicates recognised by this encoder.")
                nodename_feature_dict[RDF_triple[0]][self.unary_pred_position_dict[RDF_triple[2]]] = 1
            else:  # Fact of form R(a,b), written (a R b)
                if RDF_triple[1] not in self.binary_pred_colour_dict:
                    sys.exit(f"Predicate {RDF_triple[1]} not in the list of binary predicates recognised by this encoder.")
                if RDF_triple[0] not in nodename_feature_dict:
                    nodename_feature_dict[RDF_triple[0]] = torch.zeros(delta, dtype=torch.float)
                if RDF_triple[2] not in nodename_feature_dict:
                    nodename_feature_dict[RDF_triple[2]] = torch.zeros(delta, dtype=torch.float)
                edges.add((RDF_triple[0], RDF_triple[2], RDF_triple[1]))

        features = torch.FloatTensor(torch.stack(list(nodename_feature_dict.values())))
        assert features.shape[1] == delta
        node_names = list(nodename_feature_dict.keys()) # Correctness of this relies on dictionaries being ordered.
        edge_list = []
        edge_colour_list = []
        node_index = {name: i for i, name in enumerate(node_names)}
        for oc, dc, pred in edges:
            edge_list.append([node_index[oc], node_index[dc]])
            edge_colour_list.append(self.binary_pred_colour_dict[pred])
        return CDGraph(col_size=col_size, delta=len(self.unary_pred_position_dict),
                       features=features, edges= torch.LongTensor(torch.LongTensor(edge_list).t().contiguous()),
                       edge_colours=torch.LongTensor(edge_colour_list), node_names=node_names)

    # Returns a dictionary where the keys are cd_facts and the values are their scores
    def decode_graph(self, cd_graph: CDGraph, threshold):

        facts_scores_dict = {}
        for i, j in torch.nonzero(cd_graph.features > threshold).tolist(): # List of matrix positions with nonzero
            facts_scores_dict[(cd_graph.node_names[i], TYPE_PRED, self.unary_pred_position_dict.inverse[j])] = (
                cd_graph.features[i,j].item())

        return facts_scores_dict