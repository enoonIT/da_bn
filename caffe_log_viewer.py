from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('train')
    parser.add_argument('test')
    parser.add_argument('image_output')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    train_log = pd.read_csv(args.train)
    test_log = pd.read_csv(args.test)
    _, ax1 = plt.subplots(figsize=(15, 10))
    ax2 = ax1.twinx()
    ax1.plot(train_log["NumIters"], train_log["loss-source"], alpha=0.5, label='train_loss_source')
    if 'loss-target' in train_log:
        ax1.plot(train_log["NumIters"], train_log["loss-target"], alpha=0.3, label='train_loss_target')
    test_log["loss"][0] = test_log["loss"][1]
    ax1.plot(test_log["NumIters"], test_log["loss"], 'g', label='test_loss')
    ax2.plot(test_log["NumIters"], test_log["accuracy"], 'r', label='test_accuracy')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss')
    ax2.set_ylabel('test accuracy')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc=2)
#    plt.legend()
    plt.savefig(args.image_output, bbox_inches='tight')
    plt.show()
    print 'Accuracy max %f' % test_log["accuracy"].max()
