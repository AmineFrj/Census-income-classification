# utils/train.py

import argparse
import yaml
import pandas as pd
from utils.preprocessing import preprocess_data
from src.model import CatBoostPipeline
import json

def main(args):
    """
    Main pipeline: load configs and data, preprocess, train CatBoost, evaluate and save report.
    """
    # Load config files
    with open(args.data_config, "r") as f:
        data_config = yaml.safe_load(f)
    with open(args.model_config, "r") as f:
        model_config = yaml.safe_load(f)

    # Load data
    print("[INFO] Loading training data...")
    train_df = pd.read_csv(args.train, header=None)
    if args.test:
        print("[INFO] Loading test data...")
        test_df = pd.read_csv(args.test, header=None)
        # Preprocess data
        print("[INFO] Preprocessing data...")
        X_train, X_test, y_train, y_test, cat_features = preprocess_data(
            train_df, test_df, config=data_config, verbose=True
        )
    else:
        # Preprocess data
        print("[INFO] Preprocessing data...")
        X_train, _, y_train, _, cat_features = preprocess_data(
            train_df, config=data_config, verbose=True
        )


    # Train and save model
    print("[INFO] Initializing and training CatBoost pipeline...")
    pipe = CatBoostPipeline(config_path=args.model_config)
    pipe.fit(X_train, y_train)
    pipe.save()
    print("[INFO] Model trained and saved.")

    # Evaluate on test set if provided
    if args.test:
        print("[INFO] Evaluating on test set...")
        report = pipe.evaluate(X_test, y_test, save_report=True)
        print("[INFO] Classification report (test set):")
        print(json.dumps(report, indent=2))
    else:
        print("[INFO] No test set provided. Skipping evaluation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate CatBoost model pipeline.")
    parser.add_argument('--train', type=str, required=True, help='Path to train data CSV')
    parser.add_argument('--test', type=str, required=False, help='Path to test data CSV (optional)')
    parser.add_argument('--data-config', type=str, default='config/data_config.yml', help='Path to data config YAML')
    parser.add_argument('--model-config', type=str, default='config.yml', help='Path to model config YAML')
    args = parser.parse_args()
    main(args)
