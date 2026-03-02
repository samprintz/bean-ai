import argparse

from .preprocess import preprocess
from .train import train
from .predict import predict


def main():
    parser = argparse.ArgumentParser(prog="bean-ai", description="AI-powered account prediction for beancount")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # preprocess subcommand
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess beancount data")
    preprocess_parser.add_argument("input", help="Input beancount file")
    preprocess_parser.add_argument("--output", default="./data.csv", help="Output CSV file (default: ./data.csv)")
    preprocess_parser.add_argument("--source-account-prefix", action="append", dest="source_account_prefixes", metavar="PREFIX", default=None, help="Source account prefix to skip (default: Assets). Can be specified multiple times.")

    # train subcommand
    train_parser = subparsers.add_parser("train", help="Train the prediction model")
    train_parser.add_argument("input", help="Input CSV file")
    train_parser.add_argument("--dir", default="./models/", help="Model output directory (default: ./models/)")

    # predict subcommand
    predict_parser = subparsers.add_parser("predict", help="Predict account for a transaction")
    predict_parser.add_argument("text", help="Transaction description text")
    predict_parser.add_argument("--dir", default="./models/", help="Model directory (default: ./models/)")

    args = parser.parse_args()

    if args.command == "preprocess":
        source_account_prefixes = tuple(args.source_account_prefixes) if args.source_account_prefixes else ("Assets",)
        preprocess(args.input, args.output, source_account_prefixes)
    elif args.command == "train":
        train(args.input, args.dir)
    elif args.command == "predict":
        result = predict(args.text, args.dir)
        print(result)
