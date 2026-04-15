#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ----
"""

import torch
from torch_geometric.data import Data, DataLoader
import os.path
from encoding_schemes import CanonicalEncoderDecoder, ICLREncoderDecoder
import data_parser
from config import ExperimentConfig, EncoderType, ExperimentType, AggregationType
from utils import check, load_predicates
from gnn_architectures import GNN

# TODO: use a "sane" treatment of encoder-decoders, instead of the current 'if-elif-else' mess.
#  e.g. use an `identity' non-canonical encoder and assume there's always a non-canonical encoder

def train(config: ExperimentConfig):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dd = config.get_data_dir()
    ed = config.get_exp_dir()

    train_graph_dataset = data_parser.parse(check(dd / 'train_graph.tsv', "Train graph"))
    # TODO: sanity check - data parser should check this is compatible with encoding/decoding


    # create the cd-dataset that will be passed as input
    # 'cd' is short for (col,\delta), referring to the (col,\delta)-signature
    can_encoder_decoder = CanonicalEncoderDecoder(check(ed / 'canonical_encoder.tsv', "Canonical encoding"))
    if config.encoding_scheme == EncoderType.CANONICAL:
        cd_dataset = train_graph_dataset
    elif config.encoding_scheme == EncoderType.ICLR22:
        iclr_encoder_decoder = ICLREncoderDecoder(check(ed / "iclr22_encoder.tsv", "ICLR encoding"))
        cd_dataset = iclr_encoder_decoder.encode_dataset(train_graph_dataset)
    else:
        raise ValueError(f"Script does not support encoding scheme: {config.encoding_scheme}")

    # train_x : torch.FloatTensor of size i x j, with i the number of graph nodes, j the length of feature vectors
    # train_nodes: dictionary mapping each node in the graph to the corresponding row of train_x
    # train_edge_list : torch.LongTensor with all edges in the graph, each edge is a pair of nodes (integers)
    # train_edge_colour_list : torch.LongTensor where the ith component is the colour of the ith edge in train_edge_list
    (train_x, train_nodes, train_edge_list, train_edge_colour_list) = \
        can_encoder_decoder.encode_dataset(cd_dataset, use_dummy_constants=config.use_dummies)


    # examples are encoded as graphs equal to train_x where all labels are 0 except those corresp to facts in examples
    train_examples_dataset = []
    examples_excluded = 0
    for s, p, o in data_parser.parse(check(dd / 'train_examples.tsv', "Train examples")):
        # TODO: non-uniform enc: we could instead encode all of these in the input, see if that improves performance
        # NOTE: Drop all examples introducing nodes not in the training graph, as no predictions are generated for them
        # This is relevant in ICLR enc for examples of form R(a,b) when a and b do not occur together in train_graph
        if config.encoding_scheme == EncoderType.CANONICAL:
            _, e_nodes, _, _ = can_encoder_decoder.encode_dataset([(str(s), str(p), str(o))])
        elif config.encoding_scheme == EncoderType.ICLR22:
            cd_dataset_examples = iclr_encoder_decoder.encode_dataset([(str(s), str(p), str(o))])
            _, e_nodes, _, _ = can_encoder_decoder.encode_dataset(cd_dataset_examples)
        else:
            raise ValueError(f"Training script does not support encoding scheme: {config.encoding_scheme}")
        exclude_example = False
        for node in e_nodes:
            if node not in train_nodes:
                exclude_example = True
        if exclude_example:
            examples_excluded += 1
        else:
            train_examples_dataset.append((str(s), str(p), str(o)))
    if config.encoding_scheme == EncoderType.CANONICAL:
        cd_dataset_examples = train_examples_dataset
    elif config.encoding_scheme == EncoderType.ICLR22:
        cd_dataset_examples = iclr_encoder_decoder.encode_dataset(train_examples_dataset)
    else:
        raise ValueError(f"Training script does not support encoding scheme: {config.encoding_scheme}")
    (new_y, examples_nodes, _, _) = can_encoder_decoder.encode_dataset(cd_dataset_examples)
    train_y = torch.zeros_like(train_x) #  torch.FloatTensor of the same size as train_x
    for node in examples_nodes:
        train_y[train_nodes[node]] = new_y[examples_nodes[node]]

    # Convert to PyTorch Geometric Data objects
    # Data: "A plain old python object modeling a single graph with various (optional) attributes"
    #        Please note that edge_type is a custom attribute of the function, NOT related to the optional
    #        attribute edge_attr.
    train_data = Data(x=train_x, y=train_y, edge_index=train_edge_list, edge_type=train_edge_colour_list)
    # DataLoader: "Data loader which merges data objects from a torch_geometric.data.dataset to a mini-batch."
    #  Note that list train_data.to(device) is a Dataset. DataLoader only uses two methods within
    #  the dataset argument: __length__, and __getitem__, so it works with a list like this.
    train_loader = DataLoader(dataset=[train_data.to(device)], batch_size=1)

    data_binary_predicates, data_unary_predicates = load_predicates(check(dd / "predicates.csv", "Predicates"))
    if config.encoding_scheme==EncoderType.CANONICAL:
        cd_unary_predicates = data_unary_predicates
        cd_binary_predicates = data_binary_predicates
    elif config.encoding_scheme == EncoderType.ICLR22:
        cd_unary_predicates = iclr_encoder_decoder.canonical_unary_predicates()
        cd_binary_predicates = iclr_encoder_decoder.canonical_binary_predicates()
    else:
        raise ValueError(f"Script does not support encoding scheme: {config.encoding_scheme}")

    model = GNN(feature_dimension=len(cd_unary_predicates), num_edge_colours=len(cd_binary_predicates),
                aggregation_1=config.agg_function_1, aggregation_2=config.agg_function_2).to(device)
    # Select Adam as the optimisation algorithm
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    checkpoints_folder = ed / "checkpoints"
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)

    def train_epoch():
        # Set module in training mode (this method is inherited from torch.nn.Module)
        model.train()

        total_loss = 0

        # Notice how here we are iterating over the elements of train_loader, according to the documentation is
        # a DataLoader, which in turn means that iteration is entirely controlled by the iterable data structure
        # that implements whichever Dataset argument was used on creation on the DataLoader. In our case, the Dataset
        # is a Pytorch Geometric Data object, which provides an iterable method where it simply provides a tuple with
        # attributes, their names and values. In short, a batch here is iterating through 4-tuples of the form
        #  x, y, edge_index, edge_type
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            y = batch.y
            # Construct a weight matrix with weight of 5.0 wherever there is a
            # 1 output in the y vector, 0.5 where there is a 0.
            # This improves the propagation of the gradients.  # 0.1 and 10 gives very good results, for some reason!
            weight = torch.tensor([1.0, 5.0]).to(device)
            # .data is a tensor method that gives you the values; .long() transforms it to long format
            # ALSO: bear in mind that y is going to be a single number, because it is just one element in the batch
            # ALSO: view_as is an operation of tensors to make it look the same size as y: so essentially we are looking
            #       at weight as a tensor of the same size as y.
            weight_ = weight[y.data.long()].view_as(y)
            # Compute GNN output
            # Instances of modules are callable, and what happens on the call depends on whether there are `hooks`.
            # There aren't in this case, in which case the call uses the `forward` method inside the model. And indeed,
            # the forward method extracts named attributes from its input which coincide with the names of the
            # attributes in the object `batch` that we pass as input to the instance `model` of this Module
            output, _ = model(batch) # Ignore activations in intermediate layer
            # Target label
            label = y.to(device)
            lossFunc = torch.nn.BCELoss(reduction='none')
            # Compute loss matrix, to be reduced later
            loss = lossFunc(output[:,0], label[:,0])

            # Double check we're not getting NaNs
            assert(not (loss != loss).any())
            loss = loss * weight_[:,0]
            # Use sum reduction on loss, backpropagate
            loss.sum().backward()
            optimizer.step()
            # Any weight components < 0 are immediately "clamped" to 0, but not the bias
            for name, param in model.named_parameters():
                if 'bias' not in name and config.non_negative_weights:
                    param.data.clamp_(0)
            total_loss += batch.num_graphs * loss.sum().item()

        return total_loss

    divisor = 200 # How often we'll report progress of GNN

    # Implementing a form of early stopping. Keep track of the lowest loss achieved, if we've had max_num_bad epochs
    # only achieving higher losses than the lowest one recorded, then stop early.
    min_loss = None
    num_bad_iterations = 0
    max_num_bad = 1000

    print("Training model")
    # Train for a maximum of 20000 epochs, but expect to stop early
    for epoch in range(20000): # TODO: include these numbers in the experiment configuration
        loss = train_epoch()
        if min_loss is None: min_loss = loss
        if epoch % divisor == 0:
            print('Epoch: {:03d}, Loss: {:.5f}'.
                  format(epoch, loss))
            if epoch % 1000 == 0: # Save checkpoint for each 1000 epochs
                torch.save(model, checkpoints_folder / "{}_Epoch{}.pt".format("model", epoch))
        if loss >= min_loss:
            num_bad_iterations += 1
            if num_bad_iterations > max_num_bad:
                print("Stopping early")
                break
        else:
            num_bad_iterations = 0
            min_loss = loss

    torch.save(model, ed / "model.pt")

