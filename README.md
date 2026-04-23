# MGNNs

This repository provides an implementation of **Monotonic Graph Neural Networks (MGNNs)**.

For detailed usage of individual scripts, run them with the `--help` flag.

This project is licensed under the Apache License 2.0 — see the `LICENSE` file for details.

---

## Requirements

This code depends on:

- https://pytorch.org/
- https://github.com/rusty1s/pytorch_geometric

Tested with:
- Python 3.9.6  
- PyTorch v2.8.0  
- PyTorch Geometric v2.6.1  

---

## Directory Structure

The following directory structure is required:

.
├── data
├── src
└── experiments

- ./data — contains dataset folders  
- ./src — contains the implementation and scripts  
- ./experiments — stores configurations and outputs for each run  

---

## Data Format

Each dataset should be structured as follows:

dataset_name/
├── predicates.csv
├── train_graph.tsv
├── train_pos.tsv
├── valid_graph.tsv
├── valid_pos.tsv
├── valid_neg.tsv
├── test_graph.tsv
├── test_pos.tsv
└── test_neg.tsv

### File Descriptions

- predicates.csv  
  A comma-separated file where each line defines a predicate:
  [predicate_name],[arity]
  where arity ∈ {1, 2}.

- *_graph.tsv  
  Graph input for training, validation, and testing.

- *_pos.tsv  
  Positive examples for each split.

- valid_neg.tsv, test_neg.tsv  
  Negative examples for validation and testing.  
  Note: Training assumes all facts not in train_pos.tsv are negative.

### TSV Format

Each .tsv file contains one fact per line:

[subject]\t[relation]\t[object]

- For unary predicates, [relation] must be:
  http://www.w3.org/1999/02/22-rdf-syntax-ns#type

- Non-type relations must appear in predicates.csv with arity 2.

- If rdf:type is used:
  - The [object] must appear in predicates.csv with arity 1.

Note: Entity lists are not required. The system supports the inductive setting, meaning validation and test sets may include unseen entities.

---

## Running Experiments

### Step 1: Configure

Edit:
./src/config.yaml

Available options:

- data_dir — path to dataset folder
- exp_dir — output directory (recommended: ./experiments)
- use_dummy_constants — true / false
- encoding_scheme — canonical or iclr22
- aggregation_1 — max or sum
- aggregation_2 — max or sum
- derivation_threshold — threshold θ applied after the final layer
- non_negative_weights — true / false
- clamping — currently unsupported

---

### Step 2: Run

From the root directory:

python run_experiment.py --config-file ./src/config.yaml

---

## Output

Each run creates a new folder inside ./experiments:

<dataset_name>_<timestamp>/

Contents:

experiment_name/
├── checkpoints/
├── config.yaml
├── external_encoder.tsv
├── internal_encoder.tsv
├── model.pt
├── predicted_triples_scored.tsv
├── predicted_triples.tsv
├── test_metrics.txt
└── valid_metrics.txt

### Output Description

- checkpoints/ — model checkpoints during training  
- config.yaml — copy of the configuration used  
- external_encoder.tsv, internal_encoder.tsv — encoding pipeline  
- model.pt — trained model  
- predicted_triples_scored.tsv — predictions with scores  
- predicted_triples.tsv — predictions without scores  
- test_metrics.txt, valid_metrics.txt — evaluation metrics  

---

## Notes

- Negative training examples are implicitly defined  
- Supports inductive generalisation (unseen entities at test time)

