""" Modified from fasttext facebook example
"""
import fasttext
import argparse
from fasttext import train_supervised


def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate a fasttext supervised model')
    parser.add_argument("--training_data", type=str, default="train.txt", help='The training file to use')
    parser.add_argument("--validation_data", type=str, default="val.txt", help='The validation file to use')
    parser.add_argument("--epoch", type=str, default=25, help='Number of epoch')
    parser.add_argument("--lr", type=str, default=0.5, help='Learning rate')

    args = parser.parse_args()
    print ("Start training model...")

    model = train_supervised(input=args.training_data,
                             epoch=args.epoch,
                             lr=args.lr,
                             wordNgrams=2,
                             verbose=2,
                             loss='ova',
                             bucket=200000,
                             minCount=1)

    print_results(*model.test(args.validation_data))
    model.save_model("model.bin")

    model.quantize(input=args.training_data, qnorm=True, retrain=True, cutoff=100000)
    print_results(*model.test(args.validation_data))
    model.save_model("model.ftz")
