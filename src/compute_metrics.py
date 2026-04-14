#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ----
"""
import torch
from sympy.integrals.risch import derivation
from torch_geometric.data import Data
from numpy import arange
from numpy import trapz
from numpy import nan_to_num
import argparse
import data_parser
import os.path
import rdflib as rdf
from encoding_schemes import CanonicalEncoderDecoder, ICLREncoderDecoder
from src.config import EncoderType
from utils import load_predicates
from config import ExperimentConfig
from pathlib import Path


def precision(tp, fp):
    value = 0
    try:
        value = tp / (tp + fp)
    except:
        value = float("NaN")
    finally:
        return value


def recall(tp, fn):
    value = 0
    try:
        value = tp / (tp + fn)
    except:
        value = float("NaN")
    finally:
        return value


def accuracy(tp, fp, tn, fn):
    value = 0
    try:
        value = (tn + tp) / (tp + fp + tn + fn)
    except:
        value = float("NaN")
    finally:
        return value


def f1score(tp, fp, fn):
    value = 0
    try:
        value = tp / (tp + 0.5 * (fp + fn))
    except:
        value = float("NaN")
    finally:
        return value


def auprc(precision_vector, recall_vector):
    return -1 * trapz(precision_vector, recall_vector)


def parse_triple(line):
    temp_string = line[1:]
    bits = temp_string.split('>')
    ent1 = bits[0]
    print(ent1)
    ent2 = bits[1][2:]
    ent3 = bits[2][2:]
    ent4 = bits[3][1:-2]
    return ent1, ent2, ent3, ent4

def compute_metrics(config, predictions, positive_examples, negative_examples, metrics_file):

    threshold_list = [0.0000000001, 0.000000001, 0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001] \
                     + arange(0.01, 1, 0.01).tolist() + [0.999, 0.9999, 0.99999, 0.999999, 0.9999999, 0.99999999,
                                                         0.999999999, 0.9999999999, 0.99999999999]
    threshold_list = [round(elem, 10) for elem in threshold_list]
    number_of_positives = 0
    number_of_negatives = 0
    counter_all = 0
    counter_scored = 0
    # Each threshold is mapped to a 4-tuple containing true and false positives and negatives.
    threshold_to_counter = {0: [0, 0, 0, 0]}
    for threshold in threshold_list:
        threshold_to_counter[threshold] = [0, 0, 0, 0]
    entry_for = {"true_positives": 0, "false_positives": 1, "true_negatives": 2, "false_negatives": 3}

    print("Loading examples data from {}".format(positive_examples))
    assert os.path.exists(positive_examples), f"Positive examples file not found: {positive_examples}"
    test_positive_examples_dataset = data_parser.parse(positive_examples)

    print("Loading examples data from {}".format(negative_examples))
    assert os.path.exists(negative_examples), f"Negative examples file not found: {negative_examples}"
    test_negative_examples_dataset = data_parser.parse(negative_examples)

    test_examples_dataset = [(ex, '1') for ex in test_positive_examples_dataset] + \
                            [(ex, '0') for ex in test_negative_examples_dataset]

    # Score each individual fact
    for ((s, p, o), score) in test_examples_dataset:
        counter_all += 1
        # Check that the target fact has a score
        if (s, p, o) in predictions:
            counter_scored += 1
        if score == '1':
            # Positive example
            number_of_positives += 1
            # First consider threshold 0
            # True positive
            if predictions.get((s, p, o), 0) > 0:
                threshold_to_counter[0][entry_for["true_positives"]] += 1
            # False negative
            else:
                threshold_to_counter[0][entry_for["false_negatives"]] += 1
            # Consider all other thresholds
            for threshold in threshold_list:
                # True positive
                if predictions.get((s, p, o), 0) > threshold:
                    threshold_to_counter[threshold][entry_for["true_positives"]] += 1
                # False negative
                else:
                    threshold_to_counter[threshold][entry_for["false_negatives"]] += 1
        # Negative example
        else:
            assert score == '0'
            number_of_negatives += 1
            # First consider threshold 0
            # False positive
            if predictions.get((s, p, o), 0) > 0:
                threshold_to_counter[0][entry_for["false_positives"]] += 1
            # True negative
            else:
                threshold_to_counter[0][entry_for["true_negatives"]] += 1
            # Consider all other thresholds
            for threshold in threshold_list:
                # False positive
                if predictions.get((s, p, o), 0) > threshold:
                    threshold_to_counter[threshold][entry_for["false_positives"]] += 1
                # True negative
                else:
                    threshold_to_counter[threshold][entry_for["true_negatives"]] += 1

    #  Compute and print result
    recall_vector = []
    precision_vector = []
    print("Total examples: {}".format(counter_all))
    print("Scored examples: {}".format(counter_scored))

    with open(metrics_file, 'w') as f:
        f.write("Threshold" + '\t' + "Precision" + '\t' + "Recall" + '\t' + "Accuracy" + '\t' + "F1 Score" + '\n')
        for threshold in threshold_to_counter:
            tp, fp, tn, fn = threshold_to_counter[threshold]
            f.write("{}\t{}\t{}\t{}\t{}\n".format(threshold, precision(tp, fp),
                                                  recall(tp, fn), accuracy(tp, fp, tn, fn),
                                                  f1score(tp, fp, fn)))
            recall_vector.append(recall(tp, fp))
            precision_vector.append(precision(tp, fp))
        # Add extremal points for AUC. This ensures a perfect classifier has AUC 1, a random classifier has AUC 0.5,
        # and an `always wrong' classifier has an AUC 0.
        # Without this, a perfect classifier would have a score of 0!!
        precision_vector.insert(0, 0)
        precision_vector.append(1)
        recall_vector.insert(0, 1)
        recall_vector.append(0)
        # Get rid of NaNs
        recall_vector = nan_to_num(recall_vector)
        precision_vector = nan_to_num(precision_vector)
        f.write("Area under precision recall curve: {}\n".format(auprc(precision_vector, recall_vector)))
    f.close()
