#!/usr/bin/env python
from math import floor, ceil
from os.path import join, exists
from os import makedirs
from build_prototxt import S
from collections import namedtuple
BATCH_SIZE = 256
params_functions = {}
Setting = namedtuple('Setting', ['name', 'source', 'target'])


class BuilderHelper:
    def __init__(self, database_info, solver_defaults, train_defaults, solver_template):
        params_functions[S.base] = build_dual_shared_bn
        params_functions[S.dual_separated_bn] = build_dual_separated_bn
        params_functions[S.dual_shared_bn] = build_dual_shared_bn
        params_functions[S.dual] = build_dual_separated_bn
        self.data_info = database_info
        self.solver_defaults = solver_defaults
        self.train_defaults = train_defaults
        self.solver_template = solver_template

    def build_all(self, data_settings, exp_settings, basedir):
        for setting in data_settings:
            setting_dir = join(basedir, "{}_to_{}".format(setting[0], setting[1]))
            for exp in exp_settings:
                new_dir = join(setting_dir, exp)
                if not exists(new_dir):
                    makedirs(new_dir)
                exp_setting = Setting(exp, self.data_info[setting[0]], self.data_info[setting[1]])
                with open(join('templates', '%s_train_template.prototxt' % exp), 'rt') as train_f:
                    train_template = train_f.read()
                self.write_templates(self.solver_defaults.copy(), self.train_defaults.copy(),
                                     self.solver_template, train_template,
                                     exp_setting, new_dir)

    def write_templates(self, solver, train, solver_template, train_template, setting, outpath):
        write_specific_params = params_functions[setting.name]
        fill_general_params(solver, train, setting)
        write_specific_params(solver, train, setting)
        solver_text = multipleReplace(solver_template, solver)
        train_text = multipleReplace(train_template, train)
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
    solver["TEST_ITER"] = int(ceil(float(setting.target.size) / train["TEST_BSIZE"]))
    train["SOURCE_LIST_PATH"] = setting.source.image_list_path
    train["TARGET_LIST_PATH"] = setting.target.image_list_path


def build_dual_separated_bn(solver, train, setting):
    train_prototxt_name = "dual_separated_bn_train.prototxt"
    (source_bsize, target_bsize) = get_batch_sizes(
        setting.source.size, setting.target.size, BATCH_SIZE)
    MAX_ITER = int((200 * setting.source.size) / source_bsize)
    # solver params
    solver["STEPSIZE"] = MAX_ITER
    solver["MAX_ITER"] = MAX_ITER
    solver["TRAIN_PROTOTXT"] = train_prototxt_name
    solver["SNAPSHOT_PREFIX"] = "snapshot_dual_separated_bn"

    # train params
    train["SOURCE_BSIZE"] = source_bsize
    train["TARGET_BSIZE"] = target_bsize


def build_dual_shared_bn(solver, train, setting):
    train_prototxt_name = "dual_shared_bn_train.prototxt"
    (source_bsize, target_bsize) = (128, 128)
    MAX_ITER = int((200 * setting.source.size) / source_bsize)
    # solver params
    solver["STEPSIZE"] = MAX_ITER
    solver["MAX_ITER"] = MAX_ITER
    solver["TRAIN_PROTOTXT"] = train_prototxt_name
    solver["SNAPSHOT_PREFIX"] = "snapshot_dual_shared_bn"

    # train params
    train["SOURCE_BSIZE"] = source_bsize
    train["TARGET_BSIZE"] = target_bsize
