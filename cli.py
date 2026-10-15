"""Small command line interface for the issue triage pipeline."""
import argparse
import json
from pathlib import Path

import joblib

from pipeline import evaluate, predict, sample_input, train_model


def main():
    parser = argparse.ArgumentParser(description="Train, evaluate, or use the synthetic issue triage model.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("evaluate", help="evaluate on a deterministic holdout")
    train = commands.add_parser("train", help="fit and save both models")
    train.add_argument("--model", default="model.joblib")
    inference = commands.add_parser("predict", help="predict from a JSON object")
    inference.add_argument("--model", default="model.joblib")
    inference.add_argument("--input", help="JSON with issue_a, issue_b, and issue")
    args = parser.parse_args()
    if args.command == "evaluate":
        result = evaluate()
    elif args.command == "train":
        joblib.dump(train_model(), args.model)
        result = {"saved_model": str(Path(args.model))}
    else:
        model = joblib.load(args.model)
        payload = json.loads(args.input) if args.input else sample_input()
        result = predict(model, payload)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
