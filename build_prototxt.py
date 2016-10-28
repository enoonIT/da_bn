#!/usr/bin/env python
from collections import namedtuple
from math import floor
from argparse import ArgumentParser
from os.path import join, exists
from os import makedirs


class Keys:
    def __init__(self):
        self.A = "amazon10"
        self.D = "dslr10"
        self.W = "webcam10"
        self.C = "caltech10"

K = Keys()
Dataset = namedtuple('Dataset', ['name', 'size', 'images_path', 'image_list_path'])


def multipleReplace(text, wordDict):
    for key in wordDict:
        text = text.replace(key, wordDict[key])
    return text


def get_arguments():
    parser = ArgumentParser("")
    parser.add_argument(
        "output_directory", help="Where to build the directory tree for experiments")
    return parser.parse_args()


def build_dirtree(data_settings, exp_settings, basedir):
    for setting in data_settings:
        new_dirs = []
        setting_dir = join(basedir, "{}_to_{}".format(setting[0], setting[1]))
        new_dirs.append(setting_dir)
        new_dirs += [join(setting_dir, exp) for exp in exp_settings]
        for new_dir in new_dirs:
            if not exists(new_dir):
                makedirs(new_dir)


train_params = {
    "SOURCE_BSIZE": 128,
    "TARGET_BSIZE": 128,
    "TEST_BSIZE": 128,
    "SOURCE_LIST_PATH": None,
    "$TARGET_LIST_PATH$": None,
    "ENTROPY_LOSS_WEIGHT": 0.6,
    "MEAN_FILE": "../../datasets/imagenet_mean.binaryproto"
    }

solver_defaults = {
    "TRAIN_PROTOTXT": "train_prototxt_name",
    "TEST_ITER": 10,
    "BASE_LR": 0.001,
    "STEPSIZE": 1000,
    "MAX_ITER": 1000,
    "SNAPSHOT_PREFIX": "snapshot_"
}

data_info = {K.A: 958, K.W: 295, K.C: 1123, K.D: 157}
batch_size = 256
settings = [(K.A, K.C), (K.W, K.C), (K.D, K.C),
            (K.C, K.A), (K.C, K.W), (K.C, K.D)]
exp_settings = ["base", "dual_shared_bn", "dual_separated_bn", "dual"]

if __name__ == '__main__':
    args = get_arguments()
    for setting in settings:
        source_size = data_info[setting[0]]
        target_size = data_info[setting[1]]
        source_bsize = floor(batch_size * source_size /
                             (source_size + target_size))
        target_bsize = batch_size - source_bsize
        print "%s to %s (%d,%d) -> %d, %d" % (
            setting[0], setting[1], source_size, target_size, source_bsize,
            target_bsize)
    build_dirtree(settings, exp_settings, args.output_directory)
