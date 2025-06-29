# utils/predict.py

import argparse
import yaml
import pandas as pd
import os
from src.model import CatBoostPipeline
from utils.preprocessing import preprocess_data
import json

def main(args):
    """
    Batch scoring and (optionally) evaluation with a trained CatBoost model.
    """
    # Load data config
    with open(args.data_config, "r") as f:
        data_config = yaml.safe_load(f)

    # Load trained CatBoost pipeline
    print("[INFO] Loading CatBoost model...")
    pipe = CatBoostPipeline(config_path=args.model_config)
    pipe.load()

    # Load data to score
    print("[INFO] Loading input data...")
    df = pd.read_csv(args.input, header=None)
    df.columns = data_config["col_names"]

    # Preprocessing
    print("[INFO] Preprocessing input data...")
    X_pred, _, y_true, _, cat_features = preprocess_data(
        df, None,
        config=data_config,
        verbose=True
    )

    # Prediction
    print("[INFO] Running predictions...")
    y_pred = pipe.predict(X_pred)
    y_proba = pipe.predict_proba(X_pred)

    # Save predictions
    print("[INFO] Saving predictions...")
    result = X_pred.copy()
    result['prediction'] = y_pred
    result['proba'] = y_proba
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"✅ Predictions saved to {args.output}")

    # If ground truth labels are available, save the classification report
    if y_true is not None:
        print("[INFO] Saving performance report...")
        from sklearn.metrics import classification_report
        report = classification_report(y_true, y_pred, output_dict=True)
        report_path = args.report or "reports/predict_report.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✅ Performance report saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch prediction and evaluation with CatBoost model")
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV')
    parser.add_argument('--output', type=str, required=True, help='Path to save predictions')
    parser.add_argument('--model-config', type=str, default='config.yml', help='Path to model config YAML')
    parser.add_argument('--data-config', type=str, default='utils/data_config.yml', help='Path to data config YAML')
    parser.add_argument('--report', type=str, help='(Optional) Path to save performance report if labels present')
    args = parser.parse_args()
    main(args)
