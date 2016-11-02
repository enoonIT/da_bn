#!/usr/bin/env python
from collections import namedtuple
from argparse import ArgumentParser
import prototxt_builder


def get_arguments():
    parser = ArgumentParser("")
    parser.add_argument(
        "output_directory", help="Where to build the directory tree for experiments")
    parser.add_argument("--loss_weight", default=0.6, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    return parser.parse_args()


class Keys:
    def __init__(self):
        self.A = "amazon31"
        self.D = "dslr31"
        self.W = "webcam31"


class Settings:
    def __init__(self):
        self.base = "base"
        self.dual_shared_bn = "dual_shared_bn"
        self.dual_separated_bn = "dual_separated_bn"
        self.dual = "dual"

S = Settings()
K = Keys()
Dataset = namedtuple('Dataset', ['size', 'image_list_path'])

settings = [(K.A, K.W), (K.D, K.W), (K.W, K.D), (K.A, K.D), (K.D, K.A), (K.W, K.A)]
exp_settings = [S.base, S.dual_shared_bn, S.dual_separated_bn, S.dual]


def build_all(args):
    amazon31 = Dataset(2817, "/home/enoon/libs/DIGITS/digits/jobs/20161021-150521-4c31/train.txt")
    webcam31 = Dataset(795, "/home/enoon/libs/DIGITS/digits/jobs/20161021-150642-b588/train.txt")
    dslr31 = Dataset(498, "/home/enoon/libs/DIGITS/digits/jobs/20161021-172111-d155/train.txt")
    data_info = {K.A: amazon31, K.W: webcam31, K.D: dslr31}
    batch_size = args.batch_size
    train_defaults = {"SOURCE_BSIZE": batch_size/2,
                      "TARGET_BSIZE": batch_size/2,
                      "TEST_BSIZE": 128,
                      "SOURCE_LIST_PATH": None,
                      "TARGET_LIST_PATH": None,
                      "ENTROPY_LOSS_WEIGHT": args.loss_weight,
                      "MEAN_FILE": "../../../datasets/imagenet_mean.binaryproto",
                      "N_CLASSES": 31,
                      "BSIZE": batch_size}
    solver_defaults = {"TRAIN_PROTOTXT": "train_prototxt_name",
                       "TEST_ITER": 10,
                       "BASE_LR": 0.001,
                       "STEPSIZE": 1000,
                       "MAX_ITER": 1000,
                       "SNAPSHOT_PREFIX": "snapshot_"}
    with open('templates/solver_template.prototxt', 'rt') as solver_file:
        builder = prototxt_builder.BuilderHelper(data_info, solver_defaults, train_defaults,
                                                 solver_file.read())
    builder.build_all(settings, exp_settings, args.output_directory)


if __name__ == '__main__':
    args = get_arguments()
    build_all(args)
