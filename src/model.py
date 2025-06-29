import os
import yaml
import json
import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report

class CatBoostPipeline:
    """
    Pipeline for CatBoost classification:
    - Model training and saving
    - Loading and predicting
    - Evaluation and feature tracking
    - Configurable via YAML file
    """

    def __init__(self, config_path="config.yml"):
        """
        Initialize the pipeline from a YAML config file.
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.model_path = self.config.get("model_path", "models/catboost_final.cbm")
        self.features_path = self.config.get("features_path", "models/features_final.pkl")
        self.report_path = self.config.get("report_path", "reports/final_metrics.json")
        self.cat_features = self.config.get("cat_features", [])
        self.params = self.config.get("catboost_params", {})
        self.model = None
        self.feature_names = None
        self._log(f"Pipeline initialized with config: {config_path}")

    def fit(self, X, y):
        """
        Train CatBoost model on the provided data.
        """
        self._log("Start training CatBoost model...")
        self.model = CatBoostClassifier(**self.params, train_dir='reports/catboost_train_report')
        cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.model.fit(X, y, cat_features=cat_features, verbose=100)
        self.feature_names = list(X.columns)
        self._log("Model training complete.")

    def save(self):
        """
        Save the trained model and feature names to disk.
        """
        self.model.save_model(self.model_path)
        joblib.dump(self.feature_names, self.features_path)
        self._log(f"Model saved to {self.model_path}")
        self._log(f"Features saved to {self.features_path}")

    def load(self):
        """
        Load the model and feature names from disk.
        """
        self.model = CatBoostClassifier()
        self.model.load_model(self.model_path)
        self.feature_names = joblib.load(self.features_path)
        self._log(f"Model and features loaded from {self.model_path} and {self.features_path}")

    def predict(self, X):
        """
        Predict class labels for X (expects same features/order as training).
        """
        X = X[self.feature_names]
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        """
        X = X[self.feature_names]
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X, y, save_report=True):
        """
        Evaluate the model and return/save classification report as JSON.
        """
        X = X[self.feature_names]
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]
        report = classification_report(y, y_pred, output_dict=True)
        report["model_path"] = self.model_path
        report["features_path"] = self.features_path
        report["params"] = self.params
        if save_report:
            os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
            with open(self.report_path, "w") as f:
                json.dump(report, f, indent=2)
            self._log(f"Report saved to {self.report_path}")
        return report

    def save_config(self, path="config_saved.yml"):
        """
        Save current config to YAML file.
        """
        with open(path, "w") as f:
            yaml.dump(self.config, f)
        self._log(f"Config saved to {path}")

    def _log(self, msg):
        print(f"[CatBoostPipeline] {msg}")
