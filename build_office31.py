#!/usr/bin/env python
from collections import namedtuple
from argparse import ArgumentParser
import prototxt_builder

MEAN_FILE_PATH = "/home/enoon/libs/bvlc_caffe/data/ilsvrc12/imagenet_mean.binaryproto"

AMAZON31_PATH = "/home/enoon/code/CNN/DA_BN/da_bn/script/splits_files/amazon/full.txt"
AMAZON10_PATH = "/home/enoon/code/CNN/DA_BN/da_bn/script/datasets/amazon10/train.txt"
DSLR31_PATH = "/home/enoon/code/CNN/DA_BN/da_bn/script/splits_files/dslr/full.txt"
DSLR10_PATH = "/home/enoon/code/CNN/DA_BN/da_bn/script/datasets/dslr10/train.txt"
WEBCAM31_PATH = "/home/enoon/code/CNN/DA_BN/da_bn/script/splits_files/webcam/full.txt"
WEBCAM10_PATH = "/home/enoon/code/CNN/DA_BN/da_bn/script/datasets/webcam10/train.txt"
CALTECH10_PATH = "/home/enoon/code/CNN/DA_BN/da_bn/script/datasets/caltech10/train.txt"
#
CALTECH12_PATH = "/home/enoon/code/CNN/DA_BN/da_bn/script/datasets/caltech12_train.txt"
ILSVRC12_PATH = "/home/enoon/code/CNN/DA_BN/da_bn/script/datasets/ilsvrc12_12_train.txt"
BING12_PATH = "/home/enoon/code/CNN/DA_BN/da_bn/script/datasets/bing_12_train.txt"
PASCAL12_PATH = "/home/enoon/code/CNN/DA_BN/da_bn/script/datasets/pascal_12_train.txt"


def get_arguments():
    parser = ArgumentParser("")
    parser.add_argument(
        "output_directory", help="Where to build the directory tree for experiments")
    parser.add_argument("--loss_weight", default=0.6, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--cross_validate", default=None, type=int)
    parser.add_argument("--num_epochs", default=40, type=int)
    parser.add_argument("--sampling_protocol", action="store_true")
    return parser.parse_args()


class Keys:
    def __init__(self):
        # Office31
        self.A = "amazon31"
        self.D = "dslr31"
        self.W = "webcam31"
        # Office-Caltech
        self.A10 = "amazon10"
        self.D10 = "dslr10"
        self.W10 = "webcam10"
        self.C10 = "caltech10"
        # Image CLEF
        self.C12 = "caltech12"
        self.I12 = "imagenet12"
        self.B12 = "bing12"
        self.P12 = "pascal12"


class Settings:
    def __init__(self):
        self.base = "base"
        self.dual_shared_bn = "dual_shared_bn"
        self.dual_separated_bn = "dual_separated_bn"
        self.dual_separated_bn_eps = "dual_separated_bn_eps"
        self.dual_separated_bn_l1 = "dual_separated_bn_l1"
        self.dual_separated_bn_l1_scale = "dual_separated_bn_l1_scale"
        self.dual_separated_bn_eps_l1 = "dual_separated_bn_eps_l1"
        self.dual_separated_bn_scale = "dual_separated_bn_scale"
        self.dual = "dual"
        self.inception = "dual_separated_bn_inception"
        self.inception_l1 = "dual_separated_bn_inception_l1"
        self.inception_laplace = "dual_separated_bn_inception_laplace"
        self.alexnet_bn = "dual_separated_bn_all"

S = Settings()
K = Keys()
Dataset = namedtuple('Dataset', ['size', 'image_list_path', 'sampling_size'])

settings31 = [(K.A, K.W), (K.D, K.W), (K.W, K.D), (K.A, K.D), (K.D, K.A), (K.W, K.A)]
settingsD31 = [(K.D, K.W), (K.W, K.D), (K.A, K.D), (K.D, K.A)]
settings10 = [(K.A10, K.C10), (K.W10, K.C10), (K.D10, K.C10),
              (K.C10, K.A10), (K.C10, K.W10), (K.C10, K.D10)]
settingsIC = [(K.C12, K.I12), (K.C12, K.P12), (K.C12, K.B12), (K.I12, K.C12), (K.I12, K.P12), (K.I12, K.B12),
              (K.B12, K.I12), (K.B12, K.P12), (K.B12, K.C12), (K.P12, K.C12), (K.P12, K.I12), (K.P12, K.B12)]

# exp_settings = [S.dual_shared_bn, S.dual_separated_bn, S.dual]
# exp_settings = [S.base, S.dual_shared_bn, S.dual_separated_bn, S.dual]
# exp_settings = [S.dual_separated_bn_scale]
# exp_settings = [S.dual_separated_bn_l1, S.dual_separated_bn]
exp_settings = [S.inception]


def build_all(args):
    amazon31 = Dataset(2817, AMAZON31_PATH, 20)
    webcam31 = Dataset(795, WEBCAM31_PATH, 8)
    dslr31 = Dataset(498, DSLR31_PATH, 8)
    amazon10 = Dataset(958, AMAZON10_PATH, 20)
    webcam10 = Dataset(295, WEBCAM10_PATH, 8)
    caltech10 = Dataset(1123, CALTECH10_PATH, 20)
    dslr10 = Dataset(157, DSLR10_PATH, 8)
    #
    bing12 = Dataset(3523, BING12_PATH, 0)
    caltech12 = Dataset(2927, CALTECH12_PATH, 0)
    ilsvrc12 = Dataset(41630, ILSVRC12_PATH, 0)
    pascal12 = Dataset(11735, PASCAL12_PATH, 0)
    data_info = {K.A10: amazon10, K.W10: webcam10, K.C10: caltech10, K.D10: dslr10,
                 K.A: amazon31, K.W: webcam31, K.D: dslr31,
                 K.B12: bing12, K.C12: caltech12, K.I12: ilsvrc12, K.P12: pascal12}
    batch_size = args.batch_size
    train_defaults = {"SOURCE_BSIZE": batch_size/2,
                      "TARGET_BSIZE": batch_size/2,
                      "TEST_BSIZE": 64,
                      "SOURCE_LIST_PATH": "train_source.txt",
                      "TARGET_LIST_PATH": "train_target.txt",
                      "SOURCE_TEST_LIST_PATH": "test_source.txt",
                      "TARGET_TEST_LIST_PATH": "test_target.txt",
                      "SOURCE_TEST_BSIZE": 0,
                      "ENTROPY_LOSS_WEIGHT": args.loss_weight,
                      "MEAN_FILE": MEAN_FILE_PATH,
                      "N_CLASSES": 31,
                      "BSIZE": batch_size}
    solver_defaults = {"TRAIN_PROTOTXT": "train_prototxt_name",
                       "TEST_ITER": 10,
                       "BASE_LR": 0.001,
                       "STEPSIZE": 1000,
                       "MAX_ITER": 1000,
                       "TEST_INTERVAL": 90,
                       "SNAPSHOT_PREFIX": "snapshot_"}
    with open('templates/solver_template.prototxt', 'rt') as solver_file:
        builder = prototxt_builder.BuilderHelper(data_info, solver_defaults, train_defaults,
                                                 solver_file.read())
    builder.cross_validate = args.cross_validate
    builder.sampling_protocol = args.sampling_protocol
    builder.NUM_EPOCHS = args.num_epochs
    builder.build_all(settingsIC, exp_settings, args.output_directory)


if __name__ == '__main__':
    args = get_arguments()
    build_all(args)
