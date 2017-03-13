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
    ax1.plot(test_log["NumIters"], test_log["loss"], 'g', label='test_loss')
    source_test = 'accuracy-source'  # loss-source
    if source_test in test_log:
        ax2.plot(test_log["NumIters"], test_log[source_test], 'b', label='source-test-accuracy')
    source_test_loss = 'loss-source'
    if source_test_loss in test_log:
        test_log[source_test_loss][0] = min(test_log[source_test_loss][0], 5.0)
        ax1.plot(test_log["NumIters"], test_log[source_test_loss], 'g', alpha=0.5, label='source-test-loss')
    ax2.plot(test_log["NumIters"], test_log["accuracy"], 'r', label='test_accuracy')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss')
    ax2.set_ylabel('test accuracy')
    ax1.set_ylim([0,5])
    ax2.set_ylim([0,1])
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc=2)
#    plt.legend()
    plt.savefig(args.image_output, bbox_inches='tight')
    plt.show()
    print 'Accuracy max %f' % test_log["accuracy"].max()
