import torch
from torch_geometric.data import Data, DataLoader
import argparse
import data_parser
from encoding_schemes import CanonicalEncoderDecoder, ICLREncoderDecoder, IdentityEncoderDecoder
from config import ExperimentConfig, EncoderType
from utils import check, load_predicates
from train import train
from compute_metrics import compute_metrics
from fact_explanation import FactExplainer
from gnn_architectures import GNN
from cd_graph import CDGraph

parser = argparse.ArgumentParser()
parser.add_argument('--config-file', help='Path of the configuration file that controls the experiment.')
parser.add_argument("--skip-train", action='store_true', help='Skip training phase')
parser.add_argument( "--minimal", action="store_true", help="Minimise explanatory rules" )


if __name__ == "__main__":

    args = parser.parse_args()
    cfg = check(args.config_file, "Configuration")
    dd = cfg.get_data_dir()
    ed = cfg.get_exp_dir()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Reusable function to apply the model to a given cd-graph, returning a prediction dictionary (in the input data
    # signature) and the activations of layers 2, 1, and 0, respectively, as tensors.
    def apply_model(cd_graph: CDGraph, save_run=False):

        data = Data(x=cd_graph.features, edge_index=cd_graph.edges, edge_type=cd_graph.edge_colours).to(device)
        model.eval()

        features_layer_2, features_layer_1 = model(data)  # Actual GNN application

        features_layer_0 = data.x.detach().clone()
        # features_layer_1 is already detached
        features_layer_2 = features_layer_2.detach().clone()

        output_cd_graph = CDGraph(cd_graph.col_size, cd_graph.delta, features_layer_2, cd_graph.edges,
                                  cd_graph.edge_colours, cd_graph.node_names)
        cd_facts_scores_dict = internal_encoder.decode_graph(output_cd_graph, cfg.derivation_threshold)
        facts_scores_dict = {}  # a dictionary mapping triples (s,p,o) to a value (in str) score
        for (s, p, o), score in cd_facts_scores_dict:
            ss, pp, oo = external_encoder.decode_fact(s, p, o)
            facts_scores_dict[(ss, pp, oo)] = cd_facts_scores_dict[(s, p, o)]

        if save_run:
            derivations_file = ed / "predicted_triples.tsv"
            derivations_file_scored = ed / "predicted_triples_scored.tsv"
            # Print from the fact with the highest score to that with the least
            to_print = []
            for (s, p, o) in facts_scores_dict:
                to_print.append((facts_scores_dict[s, p, o], (s, p, o)))
            to_print = sorted(to_print, reverse=True)
            with open(derivations_file, 'w') as output:
                for (score, (s, p, o)) in to_print:
                    output.write("{}\t{}\t{}\n".format(s, p, o))
            with open(derivations_file_scored, 'w') as output2:
                for (score, (s, p, o)) in to_print:
                    output2.write("{}\t{}\t{}\t{}\n".format(s, p, o, score))
            output.close()

        return facts_scores_dict, features_layer_2, features_layer_1, features_layer_0


    print("Training...")
    if args.skip_train: # Load encoder and model from file
        internal_encoder = CanonicalEncoderDecoder(check(ed / 'internal_encoder.tsv', "Internal encoding"))
        if cfg.encoding_scheme == ICLREncoderDecoder:
            external_encoder = ICLREncoderDecoder(check(ed / 'external_encoder.tsv', "External encoding"))
        else:
            external_encoder = IdentityEncoderDecoder(check(ed / 'external_encoder.tsv', "External encoding"))
        model = torch.load(check(ed / "model.pt", "Model"), weights_only=False).to(device)
    else:
        # Encoder-decoder set-up
        # We use the paper's non-canonical encoding framework. Given input dataset,
        # --we first apply an external encoding to produce a cd_dataset (cd stands for (col,delta)),
        # --we then apply an internal encoding (always the canonical encoding) to produce a cd_graph
        # The decoding works in the reverse: internal decoding first, then external.
        # The default external decoding is an "identity" external decoding that behaves as the identity
        data_binary_predicates, data_unary_predicates = load_predicates(check(dd / "predicates.csv", "Predicates"))
        if cfg.encoding_scheme == EncoderType.ICLR22:
            external_encoder = ICLREncoderDecoder(load_from_document=None, unary_predicates=data_unary_predicates,
                                                  binary_predicates=data_binary_predicates)
        else:
            external_encoder = IdentityEncoderDecoder(load_from_document=None, unary_predicates=data_unary_predicates,
                                                      binary_predicates=data_binary_predicates)
        external_encoder.save_to_file(ed / 'external_encoder.tsv')
        internal_encoder = CanonicalEncoderDecoder(load_from_document=None,
                                                   unary_predicates=external_encoder.canonical_unary_predicates(),
                                                   binary_predicates=external_encoder.canonical_binary_predicates())
        internal_encoder.save_to_file(ed / 'internal_encoder.tsv')

        graph_dataset = data_parser.parse(check(dd / "train_graph.tsv", "Training Graph"))
        # TODO: sanity check - warn if the dataset contains any predicates out of the signature.
        cd_dataset = external_encoder.encode_dataset(graph_dataset)
        cd_graph = internal_encoder.encode_dataset(cd_dataset)

        train_examples = check(dd / "train_examples.tsv)", "Training positive examples")
        cd_train_examples = external_encoder.encode_dataset(train_examples)

        model = GNN(feature_dimension=len(external_encoder.canonical_unary_predicates()),
                    num_edge_colours=len(external_encoder.canonical_binary_predicates),
                    aggregation_1=cfg.agg_function_1, aggregation_2=cfg.agg_function_2).to(device)

        # TODO: try this: use a non-uniform encoding where much like in the code of the original ICLR paper, we encode
        #  the query facts into the cd_graph that we use. This is expressive enough to support transitivity
        train(cfg=cfg, device=device, internal_encoder=internal_encoder, model=model, cd_graph=cd_graph,
              train_examples=cd_train_examples)

    # Validation
    print("Validating...")
    predictions, _, _, _ = apply_model(dd / "valid_graph.tsv") # Ignore activations, don't save.
    compute_metrics(predictions, dd/"valid_pos.tsv", dd/"valid_neg.tsv", ed/"valid_metrics.txt")
    # TODO: print_best_threshold

    # Test (saves predictions for manual exam)
    print("Testing...")
    predictions, fl2, fl1, fl0 = apply_model(dd /"test_graph.tsv", save_run=True)
    compute_metrics(predictions, dd/"test_pos.tsv", dd/"test_neg.tsv", ed/"test_metrics.txt")

    # Explain
    print("Computing prediction explanations...")
    # explainer = FactExplainer(exp_config, dd /"test_graph.tsv", predictions, (fl2, fl1, fl0), minimal=args.minimal_rule)


