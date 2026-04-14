import torch
import argparse
import data_parser
from encoding_schemes import CanonicalEncoderDecoder, ICLREncoderDecoder
from config import ExperimentConfig, EncoderType
import sys
from utils import check, threshold_matrix_values
from train import train
from compute_metrics import compute_metrics
from fact_explanation import explain_facts

parser = argparse.ArgumentParser(description="Run an experiment instance")

parser.add_argument('--config-file',
                    help='Path of the configuration file that controls the experiment.')
parser.add_argument("--train", action="store_true")
parser.add_argument("--valid", action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("--explain", action="store_true")
parser.add_argument("--minimal_rule", action="store_true")

# Applies the saved model to a given graph
def apply_model(cfg: ExperimentConfig, graph_path, save_predictions=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading graph data from {}".format(graph_path))
    check(graph_path,f"Graph file not found: {graph_path}")
    graph_dataset = data_parser.parse(graph_path)

    if cfg.encoding_scheme == EncoderType.CANONICAL:
        cd_dataset = graph_dataset
    else:
        ep = cfg.get_exp_dir() / "encoders/iclr22_encoder.tsv"
        check(ep,f"Error: ICLR encoding file not found: {ep}")
        iclr_encoder_decoder = ICLREncoderDecoder(load_from_document=cfg.get_exp_dir() / "encoders/iclr22_encoder.tsv")
        cd_dataset = iclr_encoder_decoder.encode_dataset(graph_dataset)
    ep = cfg.exp_dir / 'encoders/canonical_encoder.tsv'
    check(ep,f"Error: Canonical encoding file not found: {ep}")
    can_encoder_decoder = CanonicalEncoderDecoder(ep)

    (data_x, nodes, edge_list, edge_colour_list) = can_encoder_decoder.encode_dataset(cd_dataset)
    data = Data(x=data_x, edge_index=edge_list, edge_type=edge_colour_list).to(device)

    mp = cfg.exp_dir / "models/model.pt"
    check(mp,f"Error: model file not found: {mp}")
    model = torch.load(mp, weights_only=False).to(device)
    model.eval()

    # Weight clamping. Note that this modifies the model.
    if cfg.clamping > 0:
        n_clamped_weights = 0
        n_total_weights = 0
        for layer in range(1, model.num_layers + 1):
            a, b = threshold_matrix_values(model.matrix_A(layer), cfg.clamping)
            n_clamped_weights += a
            n_total_weights += b
            for colour in range(model.num_colours):
                a, b = threshold_matrix_values(model.matrix_B(layer, colour), cfg.clamping)
                n_clamped_weights += a
                n_total_weights += b
        print("Percentage of clamped weights: {}".format(n_clamped_weights / n_total_weights))

    # features_layer_2 : torch.FloatTensor of size i x j, with i = num graph nodes, j = length of feature vectors
    # importantly, the ith row of gnn_output and test_x represent the same node
    features_layer_2, features_layer_1 = model(data)
    features_layer_0 = data.x.detach().clone()

    cd_output_dataset_scores_dict = can_encoder_decoder.decode_graph(nodes, features_layer_2, cfg.derivation_threshold)
    # facts_scores_dict:  a dictionary mapping triples (s,p,o) to a value (in str) score
    facts_scores_dict = {}
    if cfg.encoding_scheme == EncoderType.CANONICAL:
        facts_scores_dict = cd_output_dataset_scores_dict
    elif cfg.encoding_scheme == EncoderType.ICLR22:
        facts_scores_dict = {}
        for (s, p, o) in cd_output_dataset_scores_dict:
            ss, pp, oo = iclr_encoder_decoder.decode_fact(s, p, o)
            facts_scores_dict[(ss, pp, oo)] = cd_output_dataset_scores_dict[(s, p, o)]

    if save_predictions:
        derivations_file = cfg.exp_dir / "predicted_triples.tsv"
        derivations_file_scored = cfg.exp_dir / "predicted_triples_scored.tsv"
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


if __name__ == "__main__":

    args = parser.parse_args()
    try:
        exp_config = ExperimentConfig(args.config_file)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"ERROR: Could not load config — {e}")

    dd = exp_config.get_data_dir()
    ed = exp_config.get_exp_dir()

    if args.train:
        train(exp_config)
    if args.valid:
        predictions, features_layer_2, features_layer_1, features_layer_0 = apply_model(exp_config, dd/"valid_graph.tsv")
        compute_metrics(exp_config, predictions, dd/"valid_pos.tsv", dd/"valid_neg.tsv", ed/"valid_metrics.txt")
        # TODO: print_best_threshold
    if args.test:
        predictions, features_layer_2, features_layer_1, features_layer_0 = (
            apply_model(exp_config, dd /"test_graph.tsv", save_predictions=True))
        compute_metrics(exp_config, predictions, dd/"test_pos.tsv", dd/"test_neg.tsv", ed/"test_metrics.txt")

    if args.explain:
        print("Computing fact explanations...")
        explain_facts(exp_config, dd /"test_graph.tsv", predictions, features_layer_2, features_layer_1,
                      features_layer_0, minimal=args.minimal_rule)





