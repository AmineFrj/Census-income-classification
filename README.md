# Census Income Prediction – ML Project

## Project Structure

```
.
├── data/           # Raw, metadata & processed data files
├── models/         # Trained models and artefacts
├── reports/        # Evaluation metrics, predictions, logs
├── scripts/        # Executable scripts (training, prediction, API)
├── src/            # Core modeling classes (CatBoostPipeline, etc)
├── utils/          # Utilities: preprocessing, evaluation, config
├── notebooks/      # (if present) Prototyping & experiments
├── config.yml      # Model/serving configuration
├── utils/data_config.yml # Preprocessing/data configuration
├── Dockerfile, docker-compose.yml
├── requirements.txt
└── README.md

```

---

## Quickstart

### 1. **Install dependencies**

```bash
pip install -r requirements.txt
```

(or use Docker for full reproducibility)

### 2. **Train the model**

```bash
python scripts/train.py \
    --train data/census_income_learn.csv \
    --test data/census_income_test.csv \
    --data-config utils/data_config.yml \
    --model-config config.yml
```

- The trained model and feature list will be saved in `models/`
- Evaluation report will be written in `reports/`

### 3. **Predict on new data**

```bash
python scripts/predict.py \
    --input data/census_income_test.csv \
    --output reports/predictions.csv \
    --model-config config.yml \
    --data-config utils/data_config.yml
```

- If the input file contains the target, a classification report is also generated in `reports/`

### 4. **Serve the model (API)**

```bash
uvicorn serve:app --reload
```

- By default, serves on http://localhost:8000
- Swagger UI: http://localhost:8000/docs

---

## Configuration

- **`config.yml`**: model hyperparameters, paths, categorical features
- **`utils/data_config.yml`**: columns, categorical columns, missing values, and preprocessing steps (see `utils/preprocessing.py` for details)

---

## Key scripts

| File                     | Description                              |
| ------------------------ | ---------------------------------------- |
| `scripts/train.py`       | Train and evaluate model, save artefacts |
| `scripts/predict.py`     | Predict and (optionally) evaluate batch  |
| `scripts/serve.py`       | Run FastAPI server for real-time scoring |
| `utils/preprocessing.py` | All data cleaning & preprocessing        |
| `utils/evaluation.py`    | Metrics, plots, SHAP/feature importance  |
| `src/model.py`           | Modeling class (`CatBoostPipeline`)      |

---

## Main Notebooks

- `EDA.ipynb`: data exploration and understanding
- `Baseline experiments.ipynb`: initial modeling, cross-validation, model selection

---

## Usage examples

- **Retrain on new data**:  
  Update the CSVs in `data/`, rerun `scripts/train.py`

- **Predict on new batches**:  
  Place files in `data/`, run `scripts/predict.py` with `--input ...`

- **Add/adjust preprocessing**:  
  Edit `utils/data_config.yml` (preprocessing steps are fully configurable)

---

## Highlights & Features

- Modular codebase, clear separation of preprocessing, modeling, and serving
- All config in YAML for reproducibility and flexibility
- Robust to missing/categorical data (CatBoost native)
- Batch and real-time inference supported
- Automatic reports and metrics saving
- Ready for Docker, CI/CD, Dataiku or cloud deployment

---

## Development & Contributing

- All code under `src/` and `utils/` is importable from scripts and notebooks
- To add new models: create a class in `src/`, update training script and config
- To extend preprocessing: modify `utils/preprocessing.py` and `data_config.yml`

---

## License

MIT (or your choice) – see `LICENSE` for details

---

## Author

- Amine Ferdjaoui, Data Scientist PhD

---
