#!/usr/bin/env python
from argparse import ArgumentParser
import pandas as pd
from glob import glob
from os.path import join
import numpy as np


def get_arguments():
    parser = ArgumentParser("")
    parser.add_argument(
        "test_folder", help="Where to find the test output files")
    parser.add_argument("--num_iter", default=5, type=int, help="Number of last epochs to average")
    parser.add_argument("--kval", action="store_true")
    parser.add_argument("--sampling", action="store_true")
    parser.add_argument("--n_mean", default=5, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    folders = sorted(glob(join(args.test_folder, '*/*/')))
    if len(folders) is 0:
        print "No folders"
        exit()
    # print folders
    N_VALS = 1
    if not args.kval:
        tmp = folders[-1]
        #folders[-1] = folders[-2]
        #folders[-2] = tmp
        N_VALS = 1
    for setting in folders:
        #import ipdb; ipdb.set_trace()
        # print setting.split('/')[-2]
        file_reg = join(setting, "output_dir_*", "wtf_caffe_output*.test")
        if args.kval or args.sampling:
            file_reg = join(setting, '*/*/*.test')
        experiments = glob(file_reg)
        results = np.zeros((N_VALS, len(experiments)))
        if len(experiments) is 0:
            continue
        for k, ex in enumerate(experiments):
            test_log = pd.read_csv(ex)
            mean_acc = 100*test_log["accuracy"][-args.n_mean:].mean()
            last_acc = 100*test_log["accuracy"].iloc[-1]
            results[0, k] = mean_acc
            if args.kval:
                results[3, k] = 100*test_log["accuracy-source"][-5:].mean()
                results[4, k] = test_log["loss-source"][-5:].mean()
                results[1, k] = test_log["loss-target"][-5:].mean()
                results[2, k] = test_log["loss"][-5:].mean()
        if args.kval:
            print "%f, %f, %s %d" % (str(results.mean()), str(results.std()), setting, len(experiments))
        else:
            print "%.2f, %.2f, %s %d" % (results.mean(), results.std(), setting, len(experiments))
