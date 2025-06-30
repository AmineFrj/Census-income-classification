# Census Income Prediction – ML Project

A modular, production-ready pipeline for predicting high-income individuals using US Census data.  
Includes robust data preprocessing, model selection, interpretability, and a Dockerized API for deployment.

---

## Project Structure

```
.
├── models/             # Trained models and artifacts
│   ├── catboost_final.cbm
│   └── features_final.pkl
├── reports/            # Evaluation reports, plots, predictions
├── scripts/            # Main scripts: train, predict, serve (API)
│   ├── train.py
│   ├── predict.py
│   └── serve.py
├── src/                # Core modeling CatBoostPipeline class
│   ├── __init__.py
│   └── model.py
├── utils/              # Preprocessing, evaluation, config
│   ├── __init__.py
│   ├── data_config.yml
│   ├── evaluation.py
│   └── preprocessing.py
├── 01_EDA.ipynb                    # EDA Notebook
├── 02_Baseline experiments.ipynb   # Experiments Notebook
├── 03_Final model evaluation.ipynb # Final evaluation Notebook
├── config.yml          # Model/serving configuration
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Quickstart

1. **Install dependencies**

```console
pip install -r requirements.txt
```

Or use Docker for full reproducibility

```bash
docker-compose up --build
```

- This will start a Jupyter Lab server accessible at http://localhost:8888.
  Default token and password: 1234567890
- It will also start the REST API, available at http://localhost:8000 (Swagger docs at /docs).

2. **Train the model**

   ```bash
   python -m scripts.train \
     --train data/census_income_learn.csv \
     --test data/census_income_test.csv \
     --data-config utils/data_config.yml \
     --model-config config.yml
   ```

   - Trained model and feature list saved in `models/`
   - Evaluation report written to `reports/`

   Tip: You can consult the helper to see all available options and arguments by running:

   ```bash
   python -m scripts.train --help
   ```

3. **Predict on new data**

   ```bash
   python -m scripts.predict \
    --input data/census_income_test.csv \
    --output reports/predictions.csv \
    --model-config config.yml \
    --data-config utils/data_config.yml
   ```

   - If the input file contains the target, a classification report is also generated in `reports/`

   Tip: You can consult the helper to see all available options and arguments by running:

   ```bash
   python -m scripts.predict --help
   ```

4. **Serve the model via API**

   uvicorn scripts.serve:app --host 0.0.0.0 --port 8000 --reload

   - Default: http://localhost:8000
   - API docs (Swagger): http://localhost:8000/docs

---

## Configuration

- `config.yml`: Model hyperparameters, file paths, categorical features.
- `utils/data_config.yml`: Preprocessing steps, column names/types, handling of missing/categorical values.

---

## Main Scripts

| Script                 | Purpose                                  |
| ---------------------- | ---------------------------------------- |
| scripts/train.py       | Train and evaluate model, save artifacts |
| scripts/predict.py     | Batch predictions & evaluation           |
| scripts/serve.py       | Start FastAPI server (real-time scoring) |
| utils/preprocessing.py | Data cleaning & transformation pipeline  |
| utils/evaluation.py    | Metrics, plots, SHAP/feature analysis    |
| src/model.py           | Modeling class (CatBoostPipeline)        |

---

## Notebooks

- 01_EDA.ipynb: Exploratory Data Analysis
- 02_Baseline experiments.ipynb: Model benchmarking, CV, and selection
- 03_Final model evaluation.ipynb: Fine-tuning, interpretability, results

---

## Usage Examples

- **Retrain on new data:**  
  Update files in `data/`, rerun `scripts/train.py`.

- **Predict on new data:**  
  Place file in `data/`, run `scripts/predict.py` with `--input ...`.

- **Adjust preprocessing:**  
  Edit `utils/data_config.yml` – fully customizable pipeline.

---

## Features

- Modular & reusable: Separate modules for preprocessing, modeling, API.
- Config-driven: All settings in YAML for easy experiments and deployment.
- Robust to missing/categorical data: CatBoost manages mixed types natively.
- Batch & real-time: Predict via script or REST API.
- Automated reporting: All metrics, curves, and predictions are saved.
- Docker-ready: Full reproducibility, easy deployment (Dockerfile + docker-compose).
- Explainability: Feature importance and SHAP values integrated.
- CI/CD & cloud-ready: Structure is ready for automation or MLOps platforms.

---

## License

MIT – see `LICENSE`

---

## Author

Amine Ferdjaoui, Data Scientist, PhD
