# File: parser_util.py
# coding=utf-8
import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-root', '--dataset_root',
                        type=str,
                        help='Path to dataset',
                        default='C:\\work\\few_shot_clothing_detection\\data\\clean_data')

    parser.add_argument('-exp', '--experiment_root',
                        type=str,
                        help='Root where to store models, losses, and accuracies',
                        default='C:\\work\\few_shot_clothing_detection\\output')

    parser.add_argument('-nep', '--epochs',
                        type=int,
                        help='Number of epochs to train for',
                        default=10)

    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        help='Learning rate for the model, default=0.001',
                        default=0.001)

    parser.add_argument('-lrS', '--lr_scheduler_step',
                        type=int,
                        help='StepLR learning rate scheduler step, default=20',
                        default=20)

    parser.add_argument('-lrG', '--lr_scheduler_gamma',
                        type=float,
                        help='StepLR learning rate scheduler gamma, default=0.5',
                        default=0.5)

    parser.add_argument('-its', '--iterations',
                        type=int,
                        help='Number of episodes per epoch, default=100',
                        default=100)

    parser.add_argument('-cTr', '--classes_per_it_tr',
                        type=int,
                        help='Number of random classes per episode for training, default=5',
                        default=5)

    parser.add_argument('-nsTr', '--num_support_tr',
                        type=int,
                        help='Number of samples per class to use as support for training, default=5',
                        default=5)

    parser.add_argument('-nqTr', '--num_query_tr',
                        type=int,
                        help='Number of samples per class to use as query for training, default=5',
                        default=5)

    parser.add_argument('-cVa', '--classes_per_it_val',
                        type=int,
                        help='Number of random classes per episode for validation, default=5',
                        default=5)

    parser.add_argument('-nsVa', '--num_support_val',
                        type=int,
                        help='Number of samples per class to use as support for validation, default=5',
                        default=5)

    parser.add_argument('-nqVa', '--num_query_val',
                        type=int,
                        help='Number of samples per class to use as query for validation, default=5',
                        default=5)

    parser.add_argument('-seed', '--manual_seed',
                        type=int,
                        help='Input for the manual seed initializations',
                        default=7)

    parser.add_argument('--cuda',
                        action='store_true',
                        help='Enables CUDA')

    return parser
