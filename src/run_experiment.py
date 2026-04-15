import torch
import argparse
import data_parser
from encoding_schemes import CanonicalEncoderDecoder, ICLREncoderDecoder
from config import ExperimentConfig, EncoderType
import sys
from utils import check, threshold_matrix_values
from train import train
import os.path
from pathlib import Path
from utils import load_predicates
from compute_metrics import compute_metrics
from fact_explanation import explain_facts

parser = argparse.ArgumentParser(description="Run an experiment instance")

parser.add_argument('--config-file', help='Path of the configuration file that controls the experiment.')
parser.add_argument("--start", choices=["train", "valid", "explain"], default="train",
                    help="Phase to start from, allowing to skip steps (default: train)" )
parser.add_argument( "--minimal", action="store_true", help="Minimise explanatory rules" )

def apply_model(cfg: ExperimentConfig, graph_path, save_run=True):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    graph_dataset = data_parser.parse(check(graph_path, "Graph"))
    # TODO: add check that this is compatible with our encoders

    expdir = cfg.get_exp_dir() # same as ed

    can_encoder_decoder = CanonicalEncoderDecoder(check(expdir / 'canonical_encoder.tsv', "Canonical encoding"))
    if cfg.encoding_scheme == EncoderType.CANONICAL:
        cd_dataset = graph_dataset
    elif cfg.encoding_scheme == EncoderType.ICLR22:
        iclr_encoder_decoder = ICLREncoderDecoder(check(expdir / "iclr22_encoder.tsv", "ICLR encoding"))
        cd_dataset = iclr_encoder_decoder.encode_dataset(graph_dataset)
    else:
        raise ValueError(f"Script does not support encoding scheme: {cfg.encoding_scheme}")

    (data_x, nodes, edge_list, edge_colour_list) = can_encoder_decoder.encode_dataset(cd_dataset)
    data = Data(x=data_x, edge_index=edge_list, edge_type=edge_colour_list).to(device)

    model = torch.load(check(expdir / "model.pt", "Model"), weights_only=False).to(device)
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
    features_layer_2 = features_layer_2.detach().clone()

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

    if save_run:
        derivations_file = expdir / "predicted_triples.tsv"
        derivations_file_scored = expdir / "predicted_triples_scored.tsv"
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

        torch.save((features_layer_2, features_layer_1, features_layer_0), expdir / "activations.pt")


    return facts_scores_dict, features_layer_2, features_layer_1, features_layer_0


if __name__ == "__main__":

    args = parser.parse_args()
    # The 'start' flag helps us re-run the experiments quicker, by skipping some phases
    PHASE_ORDER = {"train": 1, "valid": 2, "explain": 3}
    start = PHASE_ORDER[args.start]

    exp_config = check(args.config_file, "Configuration")

    dd = exp_config.get_data_dir()
    ed = exp_config.get_exp_dir()

    # If running from scratch, this creates and saves the relevant encoder/decoder schemes
    # Note that there's ALWAYS a canonical encoder; other encoding schemes, if used, are applied BEFORE the canonical
    if start == 1:
        data_binary_predicates, data_unary_predicates = load_predicates(check(dd / "predicates.csv", "Predicates"))
        # 'cd' is short for (col,\delta), referring to the (col, \delta)-signature
        if exp_config.encoding_scheme == EncoderType.CANONICAL:
            cd_unary_predicates = data_unary_predicates
            cd_binary_predicates = data_binary_predicates
        elif exp_config.encoding_scheme == EncoderType.ICLR22:
            iclr_encoder_decoder = ICLREncoderDecoder(load_from_document=None,
                                                      unary_predicates=data_unary_predicates,
                                                      binary_predicates=data_binary_predicates)
            iclr_encoder_decoder.save_to_file(ed / 'iclr22_encoder.tsv')
            cd_unary_predicates = iclr_encoder_decoder.canonical_unary_predicates()
            cd_binary_predicates = iclr_encoder_decoder.canonical_binary_predicates()
        else:
            raise ValueError(f"Script does not support encoding scheme: {exp_config.encoding_scheme}")
        can_encoder_decoder = CanonicalEncoderDecoder(load_from_document=None,
                                                      unary_predicates=cd_unary_predicates,
                                                      binary_predicates=cd_binary_predicates)
        can_encoder_decoder.save_to_file(ed / 'canonical_encoder.tsv')

    if start < 2:
        train(exp_config)
    if start < 3:
        # Validation
        predictions, _, _, _ = apply_model(exp_config, dd/"valid_graph.tsv")
        compute_metrics(exp_config, predictions, dd/"valid_pos.tsv", dd/"valid_neg.tsv", ed/"valid_metrics.txt")
        # TODO: print_best_threshold
        # Test (saves predictions and activations, for manual exam and to be passed directly to explanation only)
        predictions, fl2, fl1, fl0 = apply_model(exp_config, dd /"test_graph.tsv", save_run=True)
        compute_metrics(exp_config, predictions, dd/"test_pos.tsv", dd/"test_neg.tsv", ed/"test_metrics.txt")
    else:
        predictions = {}
        with open(check(ed / "predicted_triples_scored.tsv", "Scored predictions"), "r") as f:
            for line in f:
                s, p, o, score = line.strip().split("\t")
                predictions[(s, p, o)] = float(score)
        fl2, fl1, fl0 = torch.load(ed / "activations.pt")

    print("Computing prediction explanations...")
    explain_facts(exp_config, dd /"test_graph.tsv", predictions, fl2, fl1, fl0, minimal=args.minimal_rule)





