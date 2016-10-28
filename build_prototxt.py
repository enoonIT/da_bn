#!/usr/bin/env python
from collections import namedtuple
from argparse import ArgumentParser
import prototxt_builder


def get_arguments():
    parser = ArgumentParser("")
    parser.add_argument(
        "output_directory", help="Where to build the directory tree for experiments")
    return parser.parse_args()


class Keys:
    def __init__(self):
        self.A = "amazon10"
        self.D = "dslr10"
        self.W = "webcam10"
        self.C = "caltech10"


class Settings:
    def __init__(self):
        self.base = "base"
        self.dual_shared_bn = "dual_shared_bn"
        self.dual_separated_bn = "dual_separated_bn"
        self.dual = "dual"

S = Settings()
K = Keys()
Dataset = namedtuple('Dataset', ['size', 'image_list_path'])

train_defaults = {
    "SOURCE_BSIZE": 128,
    "TARGET_BSIZE": 128,
    "TEST_BSIZE": 128,
    "SOURCE_LIST_PATH": None,
    "TARGET_LIST_PATH": None,
    "ENTROPY_LOSS_WEIGHT": 0.6,
    "MEAN_FILE": "../../../datasets/imagenet_mean.binaryproto"
}

solver_defaults = {
    "TRAIN_PROTOTXT": "train_prototxt_name",
    "TEST_ITER": 10,
    "BASE_LR": 0.001,
    "STEPSIZE": 1000,
    "MAX_ITER": 1000,
    "SNAPSHOT_PREFIX": "snapshot_"
}

settings = [(K.A, K.C), (K.W, K.C), (K.D, K.C), (K.C, K.A), (K.C, K.W), (K.C, K.D)]
exp_settings = [S.base, S.dual_shared_bn, S.dual_separated_bn, S.dual]


def build_all(args):
    amazon10 = Dataset(958, "../../../datasets/amazon10/train.txt")
    webcam10 = Dataset(295, "../../../datasets/webcam10/train.txt")
    caltech10 = Dataset(1123, "../../../datasets/caltech10/train.txt")
    dslr10 = Dataset(157, "../../../datasets/dslr10/train.txt")
    data_info = {K.A: amazon10, K.W: webcam10, K.C: caltech10, K.D: dslr10}
    with open('templates/solver_template.prototxt', 'rt') as solver_file:
        builder = prototxt_builder.BuilderHelper(data_info, solver_defaults, train_defaults,
                                                 solver_file.read())
    builder.build_all(settings, exp_settings, args.output_directory)


if __name__ == '__main__':
    args = get_arguments()
    build_all(args)
