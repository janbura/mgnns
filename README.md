This file explains how to use our implementation of monotonic GNNs. Please check the help command in each script for further details. Unless otherwise noted, all file and folder names in this document are relative to the root folder of the supplemental material. This project is licensed under the Apache License 2.0 – see the LICENSE file for details.

# MGNNs 

This code requires [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric); it was tested with Python v3.7.7, PyTorch v1.5.0 and PyTorch Geometric v1.5.0. 

## Directory Structure

The following basic directory structure is required to run our implementation of MGNNs:

```bash
.
├── data 
├── encoders 
├── explanations 
├── metrics
├── models
│   └── checkpoints
└── predictions

```

The ```./data``` folder contains the _dataset folders_, which in turn contain training, validation, and testing data.

The ```./encoders``` folder contains the encoding-decoding schemes generated for specific data.

The ```./explanations``` folder contains the extracted rules that explain predicted facts.

The ```./metrics``` folder contains the classification metrics generated using ```./calculate_classification_metrics.py```

The ```./models``` folder is where any models trained will be saved, with the ```./models/checkpoints``` subfolder for snapshots of the model generated mid-training. 

The ```./predictions``` folder contains the facts predicted by the model.


## Data Structure

We recommend that all training and testing data for our application is packaged into **datasets**. A **dataset** consists of a named folder containing the following files:

- predicates
- training
    - graph
    - positive_examples
- validation
    - graph
    - positive_examples
    - negative_examples
-testing
    - graph
    - positive_examples
    - negative_examples
     
The ```predicates''' file is a csv file where each line corresponds a predicate in the signature. The line format is "[predicate name],[arity]" where the arity is written as either 1 or 2.

The graph, positive, and negative examples files for training, validation, and testing should be expressed as a tsv file. Each line corresponds to a single fact of the form "[subject]\t[relation]\t[object]". For unary files, the relation must be ```http://www.w3.org/1999/02/22-rdf-syntax-ns#type```. [relation]s other than the type predicate **must** appear in ```predicates.csv``` with arity 2. If rdf:type is used as [relation], then the corresponding [object] **must** appear in ```predicates.csv``` with arity 1. Please note that it is not necessary to provide a list of entity names, as this system supports the so-called "inductive setting" where testing and validation files can mention entity names not seen during training.

## How to Train Your MGNN 

Please run the script ```src/train.py``` to train your MGNN. You can run the script using the following arguments:

- ```--model-name```: name (not path) of the model that will be stored
- ```--model-folder```: path of the folder where the trained model will be stored 
- ```--encoder-folder```: path of the folder where models are stored 
- ```--train-graph```: path of the folder where the training graph is stored.
- ```--train-examples```: path of the folder where the training positive examples file is stored.
- ```--encoding_scheme```: chosen encoding/decoding scheme (currently supported: ```canonical``` and ```iclr22```)
- ```--predicates```: path of the predicates.csv file to be used.
- ```--aggregation```: choice of aggregation functions (currently supported: ```max-max```, ```max-sum```, ```sum-max```, ```sum-sum```)
- ```--non-negative-weights```: set to True to use only nonnegative weights; set to False to allow negative weights (in addition to nonnegative).

Please run the script with the help option ```-h``` to access an exhaustive link of arguments and options. 

## How to Test Your MGNN

Please run the script ```src/test.py``` to apply your MGNN to a dataset. The script arguments are the following:

- ```--load-model-name```: path of the trained model to load
- ```--canonical-encoder-file```: path of the canonical encoder/decoder used to train this model
- ```--iclr22-encoder-file```: (optional) path of the iclr22 encoder/decoder used to train this model
- ```--encoding-scheme```: chosen encoding/decoding scheme (currently supported: ```canonical``` and ```iclr22```)
- ```--threshold```: threshold parameter to be used for prediction; values above the threshold are classified as positive, all others as negative 
- ```--predicates```: path of the predicates.csv file to be used.
- ```--test-graph```: path of the graph to which the MGNN is applied 
- ```--test-positive-examples```: path of the file with positive test examples (those the MGNN *should* predict)
- ```--test-positive-examples```: path of the file with negative test examples (those the MGNN *should not* predict)
- ```--output```: path of the file where the classification metrics will be printed
- ```--encoding-scheme```: choose the encoder-decoder that will be applied to the data (choices: 'iclr22' or 'canonical')

Please run the script with the help option ```-h``` to access an exhaustive link of arguments and options. 

## Rule extraction

To extract the rules captured by a GNN model, run the script ```./extract_rules.py```.  The ```--dataset-name``` is the name of the benchmark for which the model was trained, which is used to find the corresponding set of predicates in the ```./predicates``` folder. The ```--load-model-name``` is the name of the file containing the model for which we wish to extract rules. The ```--max-atoms-in-body``` argument gives the maximum number of atoms in the body that we consider in the search. Please run the script with option `-h` for a full list of parameters and options. We provide an example used to extract rules with at most 2 atoms in the body for the model trained for the GraIL-BM_WN18RR_v1 benchmark.
```bash
 python extract_rules.py --dataset-name GraIL-BM_WN18RR_v1 --encoding-scheme EC  --load-model-name GraIL-BM_WN18RR_v1_from-data_EC --max-atoms-in-body 2
```
This will create the file of extracted rules in ```./rules/extracted/```.

To check how many rules extracted by AnyBURL on a benchmark are captured (in full generality) by the MGNN model trained for the same benchmark, run the script ```./compare_anyburl.py```. The ```--dataset-name``` is the name of the benchmark for which the MGNN model was trained, which is used to find the corresponding set of predicates in the ```./predicates``` folder. The ```--load-model-name``` is the name of the file containing the MGNN model for which we wish to compare rules.  The ```--rules``` argument is the file with rules extracted by AnyBURL. We provide an example to compare the rules learned by both systems on the GraIL-BM_WN18RR_v1 benchmark. 
```bash
python compare_anyburl.py --dataset-name GraIL-BM_WN18RR_v1 --encoding-scheme EC --load-model-name GraIL-BM_WN18RR_v1_from-data_EC --encoding-scheme EC --threshold 0.00000001 --rules ./AnyBURL/rules/GraIL-BM_WN18RR_v1/alpha-10_filtered.txt 
```
The result is printed in the console.

To check how many rules extracted by AnyBURL on a benchmark are captured _specifically on the benchmark_ by the MGNN model trained for that benchmark, first set up an RDFox sandbox server running locally on port 8080. You can do this by navigating to the directory where the RDFox executable is stored and running
```bash
./RDFox sandbox . 'set endpoint.port 8080' 'endpoint start'
```
See the [RDFox documentation](https://docs.oxfordsemantic.tech/) for more details.
 
Next, run the script ```./count_anyburl_rules_entailed_by_mgnn.py```. This script needs to be run twice, and in between one needs to apply the MGNN model to the output of the first run, and then use the generated MGNN predictions as input for the second run. The `--rules` argument is the file with the rules extracted by AnyBURL. The argument `--dataset` is the incomplete testing dataset S_I of the benchmark. The argument `--queries` refers to a file that is initially empty or nonexistent, but after the first execution of the script will be created and it will contain all facts derived by the rules extracted by AnyBURL on S_I, as computed by RDFox. Finally, the argument `--answers` contains the predictions (scores) for all facts in `--queries` obtained when applying the MGNN model to the incomplete testing dataset, and has to be filled after the first run of the script, but before the second. The `--threshold` argument is the classification threshold to be applied to the scores computed by the MGNN system to determine if a fact has been derived or not.
 

Here we give the example command used for GraIL-BM_WN18RR_v1.
```bash
python count_anyburl_rules_entailed_by_mgnn.py --rules ./AnyBURL/rules/GraIL-BM_WN18RR_v1/alpha-10_filtered.txt --dataset ./data/GraIL-BM_WN18RR_v1/test/test-graph.txt --threshold 0.000000000001 --queries ./rules/captured_in_dataset/GraIL-BM_WN18RR_v1/queries --scores ./rules/captured_in_dataset/GraIL-BM_WN18RR_v1/answers
```
The first time this command is run will throw an error because the ```--answers``` file is nonexistent. However, the script will store in the ```--queries``` file all facts entailed by the AnyBURL rules on the ``incomplete`` test graph S_I. One can then apply the MGNN to S_I using the queries file as a target dataset, and the resulting predictions must be stored in the ```--answers``` file. Then, the command can be run again to produce the number of rules captured by our system on the benchmark, which is printed on the console. The command used in this case for applying the MGNN to the queries and produce the answers, in between the two runs of the previous command, is:

```bash
python test.py --dataset-name GraIL-BM_WN18RR_v1 --load-model-name GraIL-BM_WN18RR_v1_from-data_EC --encoding-scheme EC --test-data --test-graph ./data/GraIL-BM_WN18RR_v1/test/test-graph.txt --test-examples  ./rules/captured_in_dataset/GraIL-BM_WN18RR_v1/queries --print-entailed-facts ./rules/captured_in_dataset/GraIL-BM_WN18RR_v1/answers --get-scores
```

# Evaluation

This section contains the necessary instructions to reproduce the results of the Experiments section of the main paper. We discuss the file structure of the benchmark datasets, and then we provide the necessary information to run AnyBURL and DRUM.

## Benchmarks  

The ```./data``` folder contains all datasets used in the evaluation. There are 12 benchmarks for inductive knowledge graph completion (for the citations of the sources of each benchmark, please see the paper). Data for a benchmark with name {$1} is stored in the folder ```./data/{$1}```.  For a benchmark {$1}, we have in ```./data/{$1}``` three subfolders ```train``` ```test``` and ```validation```, each with the datasets for the corresponding phases of the experiment. 

For a given benchmark {$1}, file ```./data/{$1}/train/train.txt``` contains the full training dataset. Folder ```./data/{$1}/train``` also contains two subfolders: a folder ```mgnn_split``` with the training dataset split used for training MGNNs (for details about the splitting strategy we use for training see the main paper), and a folder ```drum_split``` with the split dataset for training DRUM (we used the splitting strategy described in the DRUM paper, see details below). Within each subfolder, file ```train_graph.txt``` contains the _incomplete_ graph, to be used as input for the model during training, and file ```train_facts.txt```, which contains the dataset that completes it, from where we take positive examples for training.

For a given benchmark {$1}, the folder ```./data/{$1}/test/``` contains the following files:
1) test.txt - the full testing benchmark
2) test-graph.txt - the fragment of the full testing benchmark corresponding to the ``incomplete'' dataset.
3) test-fact.txt  - the fragment of the full testing benchmark corresponding to the completing dataset, which completes the previous fragment.
4) test_all_examples.txt - a dataset that contains the dataset in point 3, plus all negative examples obtained using the strategy described in Section 4 of the paper.
5) test_all_examples_with_truth_values.txt - the same dataset as in point 4, but with an additional column that contains a value 1 if the example is positive, and a value 0 if it is negative. 

Furthermore, the folder ```./data/{$1}/test/``` contains an additional sub-folder ```beta-split``` with the alternative split of the full testing dataset (see point 1), which we used to compute the results of Table 3 in Section 4. 

The folder ```./data/{$1}/valid/``` has an analogous file structure. 


## Using AnyBURL 


We have used the version of [AnyBURL](https://web.informatik.uni-mannheim.de/AnyBURL/) available on the website, dowloaded on June 9, 2021. The system we used is in ```./AnyBURL/AnyBURL-RE.jar```. The file ```./AnyBURL/README.txt``` contains information about how to use the system and also contains the system license.

For each benchmark, we first ran the learning script of AnyBURL by going to the folder ```./AnyBURL``` and running the following command to learn rules from the benchmark:
```bash
java -cp AnyBURL-RE.jar de.unima.ki.anyburl.LearnReinforced config-learn.properties
```
For this command, AnyBURL takes parameters (including input and output file addresses) from the configuration file ```./AnyBURL/config-learn.properties```.  We used the default configuration, as given in the model file that comes with AnyBURL's download. Please see the configuration file included in the supplemental material for details, which corresponds to the configuration used for evaluating benchmark GraIL-BM_WN18RR_v1. Other benchmarks were evaluated by replacing the input and output dataset file names in ```./AnyBURL/config-learn.properties```.
 
Rules extracted by AnyBURL were stored in  ```./AnyBURL/rules/{$1}``` with {$1} the name of the benchmark.

Next, for each benchmark, we ran the rule application script of AnyBURL by going to the folder ```./AnyBURL``` and running the following command to apply the rules extracted in the previous step to the validation and testing datasets.
```bash
java -cp AnyBURL-RE.jar de.unima.ki.anyburl.Apply config-apply.properties
```
For this command, AnyBURL takes parameters (including input and output file addresses) from the configuration file ```./AnyBURL/config-apply.properties```.  We used the default configuration, as given in the model file that comes with AnyBURL's download, with one exception: when applying learned rules to a dataset, to make sure the system outputs the score of each target fact, we increased the number of answers for each query fact to 100 (in practice, the number returned is lower, which ensures that all facts derived by the system are taken into account in our evaluation). Please see the configuration file included in the supplemental material for details, which corresponds to the configuration used for evaluating benchmark GraIL-BM_WN18RR_v1. Other benchmarks were evaluated by replacing the input and output dataset file names in ```./AnyBURL/config-apply.properties```.

The predictions made by AnyBURL are stored in ```./AnyBURL/predictions/classification/{$1}``` with {$1} the name of the benchmark.
 
Finally, for each benchmark, we ran script ```./AnyBURL/anyburl_classification_analyser.py```, which processes the output by the AnyBURL system and computes the classification metrics for several thresholds, which are stored in ```./AnyBURL/metrics/classification```. For example, for GraIL-BM_WN18RR_v1 we can use the following command:
```bash
python ./AnyBURL/anyburl_classification_analyser.py --scores ./AnyBURL/predictions/classification/GraIL-BM_WN18RR_v1/alpha-10 --truths ./data/GraIL-BM_WN18RR_v1/test/test_all_examples_with_truth_values.txt --output ./AnyBURL/metrics/classification/GraIL-BM_WN18RR_v1.txt
```
To compare the rules learned with AnyBURL with those learned by MGNNs in general, we filtered out the rules that include constants, since we already known that MGNN capture rules without constants. For this, we used the script ```./AnyBURL/filer_nonground_rules.py```, where the `--input` argument should be the address of the target rule file. For example:
```bash
python ./AnyBURL/filter_nonground_rules.py --input ./AnyBURL/rules/GraIL-BM_WN18RR_v1/alpha-10
```


## Using DRUM

 
We have used the version of [DRUM](https://github.com/alisadeghian/DRUM) available on GitHub, downloaded on August 19, 2021. The system we used is in ```./DRUM-master/```. No license was found in the GitHub website. The file ```./DRUM-master/README.md``` contains all necessary information about how to use the system. Please note that DRUM uses a previous version of Python. We ran all commands using Python 2.7.

DRUM takes parameters in the command line; we always used the default configurations. DRUM paper mentions that the system was trained by randomly splitting the training dataset into an incomplete dataset T_I and the set that completes it T_C, with proportion 3:1, so we have also performed a random splitting of the training datasets with this proportion, which are stored in ```./data/{$1}/train/drum_split```, with {$1} the name of the benchmark. For each benchmark, we first train DRUM by moving into `./DRUM-master` and running the learning command
```bash
python src/main.py --datadir=datasets/{$1}/train --exps_dir=exps/{$1} --exp_name=train
```
with {$1} the name of the dataset e.g. GraIL-BM_WN18RR_v1.

Since DRUM requires input datasets to have a specific name and address, we have copied all training data to `./DRUM-master/datasets/{$1}/train` and renamed it using
the filenames required by the DRUM system (see DRUM's paper for details). For example, the incomplete training graph `./data/GraIL-BM_WN18RR_v1/train/drum_split/train_graph.txt` must be stored as `./DRUM-master/datasets/GraIL-BM_WN18RR_v1/train/facts.txt`, and the completing training graph `./data/GraIL-BM_WN18RR_v1/train/drum_split/train_facts.txt` must be stored as `./DRUM-master/datasets/GraIL-BM_WN18RR_v1/train/train.txt`. Lists of all predicates and entities in the signature must also be present in the dataset directory. For example, for benchmark GraIL-BM_WN18RR_v1 we store all entities and predicates in the signature in  `./DRUM-master/datasets/GraIL-BM_WN18RR_v1/train/entities.txt` and  `./DRUM-master/datasets/GraIL-BM_WN18RR_v1/train/relations.txt`, respectively. Please see these files for details about their format.

The models learned by DRUM are stored in ```./DRUM-master/exps/{$1}/train/ckpt```, again with {$1} being the name of the benchmark.

For validation and testing, we move into `./DRUM-master` and run DRUM using    

```bash
python src/main.py --datadir=datasets/{$1}/{$2} --exps_dir=exps/{$1} --exp_name={$2} --no_train --from_model_ckpt ./exps/{$1}/train/ckpt/model-10
```

where {$1} is the name of the benchmark, and {$2} is either ``valid`` for validation, or ``test`` for testing. Once again, we copy validation and testing datasets in `./DRUM-master/datasets/{$1}/valid` and  `./DRUM-master/datasets/{$1}/test`, respectively, using the filenames required by the DRUM system. The results of predictions are stored in ```./DRUM-master/exps/{$1}/{$2}/test_preds_and_probs.txt```.

To calculate the classification metrics for the testing results (and analogously for validation), we run the script ```./DRUM-master/drum_classification_analyser.py``` which processes the output by the DRUM system and computes the classification metrics for several thresholds, which it stores in ```./DRUM-master/exps/{$1}/test/ckpt```,with {$1} the name of the benchmark. For example, for GraIL-BM_WN18RR_v1 we can use the following command:
```bash
python ./DRUM-master/drum_classification_analyser.py --scores ./DRUM-master/exps/GraIL-BM_WN18RR_v1/test/test_preds_and_probs.txt --truths ./data/GraIL-BM_WN18RR_v1/test/test_all_examples_with_truth_values.txt --output ./DRUM-master/exps/GraIL-BM_WN18RR_v1/test/metrics.txt

```


To apply a rule's immediate consequence operator to dataset we use RDFox. An [RDFox](https://www.oxfordsemantic.tech/product "RDFox product page") server is required, which is a commercial product available under academic license. It requires a UNIX-based operating system to run (but should be very simple to modify to run on Windows - just change the way directories are referenced).



