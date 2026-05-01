# 🏛️ EstateIQ — AI House Price Prediction System

> An end-to-end machine learning pipeline for intelligent house price estimation, featuring a premium Streamlit dashboard with dark-gold aesthetics.

---

## 📸 Dashboard Preview

| Feature | Screenshot |
|---|---|
| Price Predictor | Sidebar sliders → live price estimate |
| Correlation Heatmap | Interactive top-N feature selector |
| Feature Importance | Auto-generated after training |
| Dataset Explorer | Searchable, filterable raw data view |

---

## 🗂️ Project Structure

```
house-price-predictor/
│
├── app.py                   # Streamlit dashboard (EstateIQ UI)
├── main.py                  # Entry point — runs training pipeline
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py        # Data cleaning + encoding (fit/transform)
│   ├── train.py             # Model training + evaluation + saving
│   └── predict.py           # Inference module (dict-based input)
│
├── data/
│   └── train.csv            # Kaggle Ames Housing dataset
│
├── models/
│   ├── model.pkl            # Best trained model (auto-saved)
│   ├── encoder.pkl          # OrdinalEncoder (auto-saved)
│   └── feature_columns.pkl  # Column order for inference (auto-saved)
│
├── images/
│   ├── heatmap.png
│   ├── price_distribution.png
│   ├── feature_importance.png
│   └── actual_vs_predicted.png
│
├── outputs/
│   └── model_results.csv    # MAE / RMSE / R² for all models
│
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/estateiq.git
cd estateiq
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Get the Ames Housing dataset from [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) and place `train.csv` inside the `data/` folder.

```
data/
└── train.csv
```

---

## 🚀 Usage

### Train the Model
```bash
python main.py
```

This will:
- Load and preprocess `data/train.csv`
- Fit and save the `OrdinalEncoder`
- Train **Linear Regression**, **Random Forest**, and **XGBoost**
- Evaluate all models and print MAE / RMSE / R²
- Auto-select and save the best model (by R²)
- Generate EDA and performance charts in `images/`

### Launch the Dashboard
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

### Run Inference Programmatically
```python
from src.predict import predict

price = predict({
    "GrLivArea":    1800,
    "OverallQual":  8,
    "GarageCars":   2,
    "TotalBsmtSF":  900,
    "YearBuilt":    2005,
})

print(f"Estimated Price: ${price:,.0f}")
```

---

## 🤖 Models Trained

| Model | Description |
|---|---|
| Linear Regression | Baseline linear model |
| Random Forest | 200-tree ensemble, `n_jobs=-1` |
| XGBoost | 300 estimators, `lr=0.05`, `max_depth=6`, histogram method |

The model with the highest **R² score** on the test set is automatically saved as `models/model.pkl`.

---

## 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error — average dollar error |
| **RMSE** | Root Mean Squared Error — penalises large errors |
| **R²** | Coefficient of determination — how much variance is explained |

Results are saved to `outputs/model_results.csv` after every training run.

---

## 🏛️ Dashboard Features

| Section | Description |
|---|---|
| **Metric Cards** | Live dataset stats (properties, avg price, median, features) |
| **Prediction Panel** | Sidebar sliders → instant price estimate |
| **Price Distribution** | Histogram + log-scale distribution |
| **Correlation Heatmap** | Interactive, adjustable top-N feature heatmap |
| **Feature Intelligence** | ML feature importance (post-training) or correlation fallback |
| **Dataset Explorer** | Searchable preview of raw data |

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
xgboost
joblib
matplotlib
seaborn
streamlit
```

Create `requirements.txt`:
```bash
pip freeze > requirements.txt
```

---

## 🔧 Key Design Decisions

**Why `OrdinalEncoder` over `LabelEncoder`?**
`LabelEncoder` must be fit separately per column and cannot handle unseen categories at inference time. `OrdinalEncoder` handles multiple columns at once and supports `unknown_value=-1` for unseen labels.

**Why save `feature_columns.pkl`?**
The model expects features in the exact same order as training. Saving the column list ensures inference always matches, regardless of how the input dict is ordered.

**Why dict-based input in `predict.py`?**
Passing a named `pd.DataFrame` (instead of raw `np.array`) prevents XGBoost's feature name mismatch warning and makes the API intuitive — just pass the features you know; the rest default to zero.

---

## 📁 Dataset

This project uses the **Ames Housing Dataset** from the Kaggle competition:
[House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

- **1,460** training samples
- **79** feature columns (structural, location, condition)
- Target: `SalePrice` (USD)

---

## 🙋 Author

**Rakshitha A S**


---

## 📄 License

This project is for educational and portfolio purposes.
Feel free to fork, modify, and build upon it.
```

---

*Built with Python · scikit-learn · XGBoost · Streamlit*