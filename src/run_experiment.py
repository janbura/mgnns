import torch
from torch_geometric.data import Data, DataLoader
import argparse
import data_parser
from encoding_schemes import CanonicalEncoderDecoder, ICLREncoderDecoder, IdentityEncoderDecoder
from config import EncoderType
from utils import check, load_predicates
from train import train
from config import ExperimentConfig
from compute_metrics import compute_metrics
from fact_explanation import FactExplainer
from gnn_architectures import GNN
from cd_graph import CDGraph
from datetime import datetime
from pathlib import Path
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--config-file', help='Path of the configuration file that controls the experiment.')
parser.add_argument("--skip-train", action='store_true', help='Skip training phase')
parser.add_argument( "--minimal", action="store_true", help="Minimise explanatory rules" )


if __name__ == "__main__":

    args = parser.parse_args()

    cfg = ExperimentConfig(check(args.config_file, "Configuration"))
    dd = cfg.data_dir # useful abbreviation
    ef = cfg.exp_dir / f"{dd.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}" # Experiment folder
    ef.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config_file, ef) # Copy configuration into experiment folder

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Reusable function to apply the model to a given dataset (in the input data signature), returning a dictionary
    # mapping each prediction (also in the input data signature) to its score, and the activations of layers
    # 2, 1, and 0 as tensors.
    def apply_model(dataset: set[tuple[str, str, str]], save_run=False):

        facts_scores_dict = {}  # return argument: dictionary mapping triples (s,p,o) to a score

        # Encode
        cd_dataset = external_encoder.encode_dataset(dataset) #  dataset in input signature -> dataset in cd-signature
        cd_graph = internal_encoder.encode_dataset(cd_dataset) # dataset in cd-signature -> cd_graph
        data = Data(x=cd_graph.features, edge_index=cd_graph.edges, edge_type=cd_graph.edge_colours).to(device)
        model.eval() # cd_graph -> pytorch geometric graph

        # Apply model
        features_layer_2, features_layer_1 = model(data) # note: features_layer_1 comes already detached

        # Decode
        output_cd_graph = CDGraph(cd_graph.col_size, cd_graph.delta, features_layer_2.detach().clone(), cd_graph.edges,
                                  cd_graph.edge_colours, cd_graph.node_names) # pytorch geometric graph -> cd_graph
        cd_facts_scores_dict = internal_encoder.decode_graph(output_cd_graph, cfg.derivation_threshold) # cd_graph ->
                                                                                            # dataset in cd-signature
        for (s, p, o), score in cd_facts_scores_dict.items(): # cd_dataset -> dataset in input signature
            ss, pp, oo = external_encoder.decode_fact(s, p, o)
            facts_scores_dict[(ss, pp, oo)] = cd_facts_scores_dict[(s, p, o)]

        # Save to disk
        if save_run:
            derivations_file = ef / "predicted_triples.tsv"
            derivations_file_scored = ef / "predicted_triples_scored.tsv"
            to_print = []
            for (s, p, o) in facts_scores_dict:
                to_print.append((facts_scores_dict[s, p, o], (s, p, o)))
            to_print = sorted(to_print, reverse=True) # Print from the fact with the highest score to the least
            with open(derivations_file, 'w') as output:
                for (score, (s, p, o)) in to_print:
                    output.write("{}\t{}\t{}\n".format(s, p, o))
            with open(derivations_file_scored, 'w') as output2:
                for (score, (s, p, o)) in to_print:
                    output2.write("{}\t{}\t{}\t{}\n".format(s, p, o, score))
            output.close()

        return facts_scores_dict, features_layer_2.detach().clone(), features_layer_1, data.x.detach().clone()

    # Training (or Model Loading, if training is skipped)
    if args.skip_train:
        print("Loading encoder and model from file...")
        internal_encoder = CanonicalEncoderDecoder(check(ef / 'internal_encoder.tsv', "Internal encoding"))
        if cfg.encoding_scheme == EncoderType.ICLR22:
            external_encoder = ICLREncoderDecoder(check(ef / 'external_encoder.tsv', "External encoding"))
        else:
            external_encoder = IdentityEncoderDecoder(check(ef / 'external_encoder.tsv', "External encoding"))
        model = torch.load(check(ef / "model.pt", "Model"), weights_only=False).to(device)
    else:
        print("Training...")

        # Set-up Encoder-Decoder
        dbp, dup = load_predicates(check(dd / "predicates.csv", "Predicates"))
        if cfg.encoding_scheme == EncoderType.ICLR22:
            external_encoder = ICLREncoderDecoder(load_from_document=None, unary_predicates=dup, binary_predicates=dbp)
        else:
            # Default external encoding is the identity, encoding each fact as itself.
            external_encoder = IdentityEncoderDecoder(load_from_document=None, unary_predicates=dup, binary_predicates=dbp)
        internal_encoder = CanonicalEncoderDecoder(load_from_document=None,
                                                   unary_predicates=external_encoder.canonical_unary_predicates,
                                                   binary_predicates=external_encoder.canonical_binary_predicates)
        external_encoder.save_to_file(ef / 'external_encoder.tsv')
        internal_encoder.save_to_file(ef / 'internal_encoder.tsv')

        # Load & encode training data
        # TODO: sanity check - warn if the training data contains any predicates out of the signature.
        graph_dataset = data_parser.parse(check(dd / "train_graph.tsv", "Training graph"))
        cd_dataset = external_encoder.encode_dataset(graph_dataset, use_dummy_constants=cfg.use_dummies)
        cd_graph = internal_encoder.encode_dataset(cd_dataset)
        train_examples_dataset = data_parser.parse(check(dd / "train_pos.tsv", "Training positive examples"))
        cd_train_examples = external_encoder.encode_dataset(train_examples_dataset)

        model = GNN(feature_dimension=cd_graph.delta,num_edge_colours=cd_graph.col_size,
                    aggregation_1=cfg.agg_function_1, aggregation_2=cfg.agg_function_2).to(device)

        # TODO: try this: use a non-uniform encoding where much like in the code of the original ICLR paper, we encode
        #  the query facts into the cd_graph that we use. This is expressive enough to support transitivity
        train(cfg=cfg, device=device, internal_encoder=internal_encoder, model=model, cd_graph=cd_graph,
              train_examples=cd_train_examples, experiment_folder=ef)

    # Validation
    print("Validating...")
    valid_graph_dataset = data_parser.parse(check(dd / "valid_graph.tsv", "Validation graph"))
    predictions, _, _, _ = apply_model(valid_graph_dataset)   # Ignore activations, don't save.
    compute_metrics(predictions, dd/"valid_pos.tsv", dd/"valid_neg.tsv", ef/"valid_metrics.txt")
    # TODO: print_best_threshold

    # Test (saves predictions for manual exam)
    print("Testing...")
    test_graph_dataset = data_parser.parse(check(dd / "test_graph.tsv", "Test graph"))
    predictions, fl2, fl1, fl0 = apply_model(test_graph_dataset, save_run=True)
    compute_metrics(predictions, dd/"test_pos.tsv", dd/"test_neg.tsv", ef/"test_metrics.txt")

    # Explanation
    # print("Computing prediction explanations...")
    # explainer = FactExplainer(cfg, dd /"test_graph.tsv", predictions, (fl2, fl1, fl0), minimal=args.minimal_rule)


