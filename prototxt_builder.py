#!/usr/bin/env python
from math import floor, ceil
from os.path import join, exists
from os import makedirs
from build_office31 import S
from collections import namedtuple
from random import shuffle, seed
params_functions = {}
Setting = namedtuple('Setting', ['name', 'source', 'target', 'cross_validate', 'epochs', 'sampling_protocol'])


class BuilderHelper:
    def __init__(self, database_info, solver_defaults, train_defaults, solver_template):
        params_functions[S.base] = build_dual_shared_bn
        params_functions[S.dual_separated_bn] = build_dual_separated_bn
        params_functions[S.alexnet_bn] = build_dual_separated_bn
        params_functions[S.inception] = build_inception
        params_functions[S.inception_l1] = build_inception
        params_functions[S.inception_laplace] = build_inception
        params_functions[S.dual_separated_bn_scale] = build_dual_separated_bn
        params_functions[S.dual_separated_bn_eps] = build_dual_separated_bn
        params_functions[S.dual_separated_bn_l1] = build_dual_separated_bn
        params_functions[S.dual_separated_bn_l1_scale] = build_dual_separated_bn
        params_functions[S.dual_separated_bn_eps_l1] = build_dual_separated_bn
        params_functions[S.dual_shared_bn] = build_dual_shared_bn
        params_functions[S.dual] = build_dual_separated_bn
        self.data_info = database_info
        self.solver_defaults = solver_defaults
        self.train_defaults = train_defaults
        self.solver_template = solver_template
        self.cross_validate = False
        self.NUM_EPOCHS = 120
        self.sampling_protocol = False

    def build_all(self, data_settings, exp_settings, basedir):
        if self.sampling_protocol:
            print "Will do sampling protocol"
        for setting in data_settings:
            setting_dir = join(basedir, "{}_to_{}".format(setting[0], setting[1]))
            for exp in exp_settings:
                new_dir = join(setting_dir, exp)
                if not exists(new_dir):
                    makedirs(new_dir)
                exp_setting = Setting(exp, self.data_info[setting[0]], self.data_info[setting[1]],
                                      self.cross_validate, self.NUM_EPOCHS, self.sampling_protocol)
                with open(join('templates', '%s_train_template.prototxt' % exp), 'rt') as train_f:
                    train_template = train_f.read()
                self.write_templates(self.solver_defaults.copy(), self.train_defaults.copy(),
                                     self.solver_template, train_template,
                                     exp_setting, new_dir)

    def write_templates(self, solver, train, solver_template, train_template, setting, outpath):
        write_specific_params = params_functions[setting.name]
        write_specific_params(solver, train, setting)
        fill_general_params(solver, train, setting)
        train_text = multipleReplace(train_template, train)
        solver_text = multipleReplace(solver_template, solver)
        source_files = open(setting.source.image_list_path).readlines()
        target_files = open(setting.target.image_list_path).readlines()
        if setting.sampling_protocol:
            n_splits = 5
            seed(0)
            for k in range(n_splits):
                new_dir = join(outpath, "k_" + str(k))
                if not exists(new_dir):
                    makedirs(new_dir)
                shuffle(source_files)
                splits = get_sampling_split(source_files, setting.source.sampling_size)
                write_lines(new_dir, "source", splits, None)
                write_lines(new_dir, "target", target_files, target_files)
                with open(join(new_dir, "solver.prototxt"), "wt") as solver_file:
                    solver_file.write(solver_text)
                with open(join(new_dir, solver["TRAIN_PROTOTXT"]), "wt") as train_file:
                    train_file.write(train_text)

        elif setting.cross_validate:
            for k in range(setting.cross_validate):
                new_dir = join(outpath, "k_" + str(k))
                if not exists(new_dir):
                    makedirs(new_dir)
                seed(0)
                shuffle(source_files)
                #shuffle(target_files)
                splits_s = chunkify(source_files, setting.cross_validate)
                #splits_t = chunkify(target_files, setting.cross_validate)
                write_splits(new_dir, "source", splits_s, k)
                #write_splits(new_dir, "target", splits_t, k)
                write_lines(new_dir, "target", target_files, target_files)
                with open(join(new_dir, "solver.prototxt"), "wt") as solver_file:
                    solver_file.write(solver_text)
                with open(join(new_dir, solver["TRAIN_PROTOTXT"]), "wt") as train_file:
                    train_file.write(train_text)
        else:
            write_lines(outpath, "source", source_files, None)
            write_lines(outpath, "target", target_files, target_files)
            with open(join(outpath, "solver.prototxt"), "wt") as solver_file:
                solver_file.write(solver_text)
            with open(join(outpath, solver["TRAIN_PROTOTXT"]), "wt") as train_file:
                train_file.write(train_text)


def multipleReplace(text, wordDict):
    for key in wordDict.keys():
        text = text.replace("${}$".format(key), str(wordDict[key]))
    return text


def get_batch_sizes(source_size, target_size, batch_size):
    source_bsize = int(floor(batch_size * source_size /
                             (source_size + target_size)))
    target_bsize = int(batch_size - source_bsize)
    return (source_bsize, target_bsize)


def fill_general_params(solver, train, setting):
    if setting.cross_validate:
        source_size = int(setting.source.size / setting.cross_validate)
        target_size = setting.target.size
        train["SOURCE_TEST_BSIZE"] = train["SOURCE_BSIZE"]  # must be the same or slice will not work
        train["TARGET_TEST_BSIZE"] = int((train["SOURCE_BSIZE"] * target_size) / source_size)
        solver["TEST_ITER"] = int(ceil(float(target_size) / train["TARGET_TEST_BSIZE"]))
    else:
        if setting.name in [S.inception, S.inception_laplace, S.inception_l1]:
            train["TEST_BSIZE"] = 8
        else:
            train["TEST_BSIZE"] = 64
        solver["TEST_ITER"] = int(ceil(float(setting.target.size) / train["TEST_BSIZE"]))


def build_inception(solver, train, setting):
    train_prototxt_name = "dual_separated_bn_inception_train.prototxt"
    MAX_ITER = int((setting.epochs * setting.source.size) / 32)
    solver["STEPSIZE"] = int(MAX_ITER * 0.333)
    solver["MAX_ITER"] = MAX_ITER
    solver["TRAIN_PROTOTXT"] = train_prototxt_name
    solver["SNAPSHOT_PREFIX"] = "snapshot_dual_separated_bn_inception"
    solver["TEST_INTERVAL"] = 120


def build_dual_separated_bn(solver, train, setting):
    train_prototxt_name = "dual_separated_bn_train.prototxt"
    source_size = setting.source.size
    target_size = setting.target.size
    if setting.cross_validate:
        kval = setting.cross_validate
        source_size = int((kval-1) * setting.source.size) / kval
    elif setting.sampling_protocol:
        source_size = int(setting.source.sampling_size * train["N_CLASSES"])

    (source_bsize, target_bsize) = get_batch_sizes(source_size, target_size, train["BSIZE"])
    MAX_ITER = int((setting.epochs * source_size) / source_bsize)
    # solver params
    solver["STEPSIZE"] = int(MAX_ITER * 0.9)
    solver["MAX_ITER"] = MAX_ITER
    solver["TRAIN_PROTOTXT"] = train_prototxt_name
    solver["SNAPSHOT_PREFIX"] = "snapshot_dual_separated_bn"

    # train params
    train["SOURCE_BSIZE"] = source_bsize
    train["TARGET_BSIZE"] = target_bsize


def build_dual_shared_bn(solver, train, setting):
    train_prototxt_name = "dual_shared_bn_train.prototxt"
    source_bsize = train["SOURCE_BSIZE"]
    MAX_ITER = int((setting.epochs * setting.source.size) / source_bsize)
    # solver params
    solver["STEPSIZE"] = int(MAX_ITER * 0.9)
    solver["MAX_ITER"] = MAX_ITER
    solver["TRAIN_PROTOTXT"] = train_prototxt_name
    solver["SNAPSHOT_PREFIX"] = "snapshot_dual_shared_bn"


def chunkify(lst, n):
    return [lst[i::n] for i in xrange(n)]


def write_splits(path, append, splits, i):
    train = splits[0:i] + splits[i+1:]
    test = splits[i]
    write_lines(path, append, sum(train, []), test)
        

def write_lines(path, append, train_lines, test_lines):
    if train_lines:
        with open(join(path, "train_" + append + ".txt"), 'wt') as tmp:
            tmp.writelines(train_lines)
    if test_lines:
        with open(join(path, "test_" + append + ".txt"), 'wt') as tmp:
            tmp.writelines(test_lines)


def get_sampling_split(split_lines, sampling):
    subset = []
    class_counter = {}
    for line in split_lines:
        [path, label] = line.strip().split()
        items = class_counter.get(label, 0)
        if items < sampling:
            subset.append(path + " " + label + "\n")
            class_counter[label] = items + 1
    return subset
