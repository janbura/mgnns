This file explains how to use our implementation of monotonic GNNs. Please check the help command in each script for further details. This project is licensed under the Apache License 2.0 – see the LICENSE file for details.

# MGNNs 

This code requires [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric); it was tested with Python 3.9.6, PyTorch v2.8.0 and PyTorch Geometric v2.6.1. 

## Directory Structure

The following basic directory structure is required to run our implementation of MGNNs:

```bash
.
├── data
├── src
└── experiments

```

The `./data` folder contains the _dataset folders_, which in turn contain training, validation, and testing data.

The `./src` folder contains the Python scripts that implement and run our system. 

The `./experiments` folder contains the configuration and output of each experiment that has been run. 


## Data Structure

Each benchmark/dataset should consist of a single folder with the following files:

```
dataset_name
├── predicates.csv
├── train_graph.tsv
├── train_pos.tsv
├── valid_graph.tsv
├── valid_pos.tsv
├── valid_neg.tsv
├── test_graph.tsv
├── test_pos.tsv
└── test_neg.tsv
```

The `predicates.csv` file is a comma-separated file where each line corresponds a predicate in the signature. The line format is "[predicate name],[arity]" where the arity is either 1 or 2.

The `train_graph.tsv`, `valid_graph.tsv`, and `test_graph.tsv` files contain the input graph that will be fed into the model for training, validation, and testing, respectively.

The `train_pos.tsv`, `valid_pos.tsv`, and `test_pos.tsv` files contain the positive examples for training, validation, and testing, respectively.

The `valid_neg.tsv`, `test_pos.tsv` files contain the negative examples for validation and testing. Note that there are no negative examples for training: all facts not in train_pos.tsv are assumed to be negative for training purposes.

In each tsv file, each line corresponds to a single fact of the form "[subject]\t[relation]\t[object]". For unary files, the [relation] must be `http://www.w3.org/1999/02/22-rdf-syntax-ns#type`. [relation]s other than the type predicate **must** appear in `predicates.csv` with arity 2. If rdf:type is used as [relation], then the corresponding [object] **must** also appear in `predicates.csv` with arity 1. Please note that it is not necessary to provide a list of entity names, as this system supports the so-called "inductive setting" where testing and validation files can mention entity names not seen during training.

## How to Run the Experiments

First edit `./src/config/config.yaml` to select your preferred configuration. This file has several options:

- data_dir: the path of the folder that the benchmark/dataset for this experiment (usually this is a subfolder of ./data)
- exp_dir: the path of the folder where the folder with experiment results for this benchmar will be stored. We recommend using `./experiments`.
- use_dummy_constants: parameter set to `true` or `false` to introduce dummy constants in the training graph to disincentivise false positives. 
- encoding_scheme: currently only `canonical` or `iclr22` are supported, corresponding to the canonical encoding or the encoding described in our ICLR22 paper [1]
- aggregation_1: aggregation function in the first layer; currently supporting `max` or `sum` only
- aggregation_2: aggregation function in the second layer; currently supporting `max` or `sum` only
- derivation_threshold: threshold \theta from the papers, applied after the last layer to decide which facts are derived
- non_negative_weights: parameter set to `true` or `false` to impose that the learned matrices of the model have non-negative weights. 
- clamping: [FEATURE CURRENTLY UNSUPPORTED] 

To train and apply a GNN model for a given benchmark with the structure described in the previous section, please open a terminal in the root directory and run 

```bash
  python run_experiment.py ./src/config/config.yaml
```

This will create a new folder in `./experiments` labelled by the dataset name (the name of the folder whose path is data_dir) followed by the timestamp of the experiment start. The experiment should create the following files:

You can run the experiment with the option --load_model [folder of a previous experiment]. This will load the model and encoder from a previous experiment. This can be useful to skip training; however, please be mindful that the model parameters in that experiment match those in your current folder. Further support to validate this condition should eventually be added.

```
experiment_name
├── checkpoints
├── config.yaml
├── external_encoder.tsv
├── internal_encoder.tsv
├── model.pt
├── predicted_triples_scored.tsv
├── predicted_triples.tsv
├── test_metrics.txt
└── valid_metrics.txt 
```

- Folder `checkpoints` contains checkpoint saves of the model during training.
- File `config.yaml` is a copy of the configuration file used to run this experiment. 
- Files `external_encoder.tsv` and `internal_encoder.tsv` show the external and internal encoder, respectively, used for the experiment. Please note that we use the framework from our paper [2] where we always consider an external, non-canonical encoder that maps an input dataset to a (col,d)-dataset, and then an internal encoder--which is always the canonical encoder--which maps this (col,d)-dataset to a (col,d)-graph. The ICLR22 is an example of external encoder. If we wish to use only the canonical encoder in an experiment (i.e. by selecting "canonical" in the configuration file), then our code generates an external encoder which is simply the identity.
- File `model.pt` is a saved version of the model. 
- File `predicted_triples_scored.tsv` contains a prediction in each line followed by its "score" (i.e. the corresponding value in the last layer of the GNN application)
- File `predicted_triples.tsv` contains the same but without the scores.
- Files `test_metrics.txt` and `valid_metrics.txt` contain the classification metrics obtained by applying the model on the testing and validation datasets, respectively, over a variety of fixed thresholds.
