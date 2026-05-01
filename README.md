# Student Performance Early-Warning System

**Predictive Analytics for Student Academic Performance Using Machine Learning**  
CMP600 – Dissertation | BSc (Hons) Computing [Top-up]


---

## Setup & Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Place the Dataset

Download the xAPI-Edu-Data dataset from Kaggle:  
https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data

Place `xAPI-Edu-Data.csv` in the project root directory.

### 3. Run the Jupyter Notebook

```bash
cd notebook
jupyter notebook ML_Pipeline.ipynb
```

Run **all cells** in order. The notebook will:
- Load and clean the dataset (480 → 478 records after deduplication)
- Perform exploratory data analysis with visualisations
- Preprocess data (one-hot encoding, stratified train-test split)
- Train 4 models: Logistic Regression, SVM, Random Forest, XGBoost
- Evaluate with stratified 10-fold cross-validation
- Analyse feature importance (Gini importance)
- Conduct fairness testing (gender subgroup F1 comparison)
- Run sensitivity analysis (performance with/without gender features)
- **Export trained model artefacts to `./models/`**

The exported files are what the Flask app loads to serve predictions.

### 4. Run the Flask App

```bash
cd ..
python app.py
```

Open in browser: **http://localhost:5000**

---

## Technologies

- **Python 3.10+**
- **scikit-learn** — ML models, preprocessing, evaluation
- **XGBoost** — Gradient boosting classifier
- **pandas / numpy** — Data manipulation
- **matplotlib / seaborn** — Visualisation (notebook)
- **Flask** — Web application framework
- **joblib** — Model serialisation
- **Jupyter** — Interactive notebook
