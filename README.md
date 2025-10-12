# 💎 Diamond Price Analysis & Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)  
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)  
![Scikit-Learn](https://img.shields.io/badge/ML-ScikitLearn-orange)  
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)  
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)  
![Status](https://img.shields.io/badge/Project-Complete-brightgreen)

---

## 📘 Project Overview

**Diamond Price Analysis & Prediction** applies supervised machine learning to estimate the **price of diamonds** based on key attributes — carat, cut, color, clarity, and dimensions (`x`, `y`, `z`, `depth`, `table`).  
The project walks through **EDA, feature engineering, model training, performance evaluation, and Streamlit app deployment** for real-time prediction.

💡 **Business value:** Enables jewelers and e-commerce vendors to accurately estimate pricing, optimize valuation, and simulate offers.

---

## 📂 Repository Structure

```
Diamond_Price_Analysis/
│── data/
│   └── diamond_price_data.csv
│── notebook/
│   └── diamond_price_analysis_portfolio.ipynb   # Enhanced with markdown explanations
│── app/
│   └── diamond_pricer.py                         # Streamlit app
│── models/
│   ├── preprocessing_pipeline.pkl
│   └── best_diamond_model.joblib
│── images/
│   ├── eda_price_distribution.png
│   ├── feature_importance.png
│   ├── streamlit_app_demo.gif
│   └── Diamond_ERD.png
│── README.md
│── requirements.txt
```

---

## 📊 Dataset Description

| Feature | Description |
|----------|--------------|
| `carat` | Diamond weight (key determinant of price) |
| `cut` | Quality of cut — Fair, Good, Very Good, Premium, Ideal |
| `color` | Diamond color (J = worst, D = best) |
| `clarity` | Clarity level (I1, SI2, VS1, VVS2, IF, etc.) |
| `depth` | Total depth percentage |
| `table` | Width of top relative to widest point |
| `x`, `y`, `z` | Dimensions in mm |
| `price` | Target variable — price in USD |

---

## ⚙️ Workflow Summary

1. **Data Loading & Cleaning**  
   - Handled missing and zero-dimension values.  
   - Removed unrealistic records and corrected datatypes.  

2. **Exploratory Data Analysis (EDA)**  
   - Price distribution analysis (right-skewed).  
   - Correlation matrix and feature relationships.  
   - Visualization of `carat` vs `price`.  



<img width="1500" height="600" alt="price_carat_hist" src="https://github.com/user-attachments/assets/4c7bb88b-0b24-4707-bbf1-95bb0344ee81" />

<img width="1200" height="900" alt="corr_heatmap" src="https://github.com/user-attachments/assets/c6eaed7d-e0f2-4d50-b017-e3df1a837999" />

<img width="1200" height="900" alt="carat_price_scatter" src="https://github.com/user-attachments/assets/0159b74a-4d5a-4ea7-a527-4c5ad2b66deb" />


4. **Feature Engineering**  
   - Created `volume = x * y * z` feature.  
   - One-hot encoded categorical features (`cut`, `color`, `clarity`).  
   - Scaled numerical features for consistency.

5. **Model Training**  
   - Compared multiple regressors (Linear, RandomForest, XGBoost).  
   - Final model: **XGBoostRegressor** (best RMSE & R²).  

6. **Evaluation**  
   - Computed RMSE and R² metrics.  
   - Interpreted model performance and business significance.  

7. **Deployment**  
   - Streamlit app (`diamond_pricer.py`) for interactive price prediction.  
   - Model and preprocessing pipeline saved using `joblib`.

   ![Streamlit App Demo](images/streamlit_app_demo.gif)

---

## 🧩 Selected Notebook Snippets & Explanations

### 🔹 Data Load & Initial Inspection
```python
DATA_PATH = os.path.join('data','diamonds.csv')
VIS_DIR = os.path.join('visuals')
os.makedirs(VIS_DIR, exist_ok=True)

# Load
print('Loading data from', DATA_PATH)
df = pd.read_csv(DATA_PATH)
df.info()
```
**Explanation:** Load and preview the dataset to confirm structure, datatypes, and detect anomalies like zero dimensions.

### 🔹 EDA — Price Distribution
```python
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.histplot(df_clean['price'], bins=50, kde=True)
plt.title('Price distribution')
```
**Interpretation:** Price is right-skewed; consider log-transform (`np.log(price)`) to stabilize variance for linear models.

### 🔹 Feature Engineering
```python
from sklearn.preprocessing import OneHotEncoder

df['price_per_carat'] = df['price'] / df['carat']
df['volume'] = df['x'] * df['y'] * df['z']

# Optional cap: filter extreme high prices to reduce skew for modeling diagnostics
price_cap = df['price'].quantile(0.999)
df = df[df['price'] <= price_cap].copy()
print('After capping extreme prices:', df.shape)

# Features and target
FEATURES = ['carat','depth','table','x','y','z','volume']
CAT = ['cut','color','clarity']
TARGET = 'price'

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print('Train/test sizes:', train_df.shape, test_df.shape)

# Encoder fit on train
enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
enc.fit(train_df[CAT])

def prepare_X(df_in, encoder=enc):
    df2 = df_in.copy()
    cat_enc = encoder.transform(df2[CAT])
    cat_cols = encoder.get_feature_names_out(CAT)
    df_cat = pd.DataFrame(cat_enc, columns=cat_cols, index=df2.index)
    X = pd.concat([df2[FEATURES].reset_index(drop=True), df_cat.reset_index(drop=True)], axis=1)
    return X

X_train = prepare_X(train_df)
X_test = prepare_X(test_df)
y_train = train_df[TARGET]
y_test = test_df[TARGET]

print('X shapes:', X_train.shape, X_test.shape)
```
**Interpretation:** one-hot encoding converts categorical features into numeric form.

### 🔹 Modeling (XGBoost Example)
```python
if xgb_available:
    xgb = XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror')
    xgb_params = {'n_estimators':[100,200], 'max_depth':[4,6,8], 'learning_rate':[0.05,0.1]}
    cv_xgb = GridSearchCV(xgb, xgb_params, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=1)
    cv_xgb.fit(X_train, y_train)
    print('Best XGB params:', cv_xgb.best_params_)
    xgb_best = cv_xgb.best_estimator_
    xgb_preds = xgb_best.predict(X_test)
    print('XGB R2:', r2_score(y_test, xgb_preds))
    print('XGB RMSE:', sqrt(mean_squared_error(y_test, xgb_preds)))
else:
    print('XGBoost not available in the environment. If you want XGBoost, install xgboost and rerun.')
```
**Interpretation:** XGBoost captures non-linear relationships (e.g., `carat` × `clarity`). Model tuned via cross-validation.


---

## 📈 Model Performance Summary

| Metric | Value |
|---------|--------|
| **RMSE** | 523.5 |
| **R²** | 0.982 |

✅ Model explains ~98% of variance in price with low average error.

---

<img width="1200" height="1425" alt="xgb_shap_summary" src="https://github.com/user-attachments/assets/e0651de7-8bf8-49fb-bda2-7255d0c70573" />


---

## 🧠 Insights

- `carat` remains highly correlated with price, but the model reveals that volume (x × y × z) and the y dimension (length) capture diamond size and proportion more effectively than carat alone.  
- Higher `clarity` and `cut` grades lead to exponential price increases, confirming that visual perfection drives premium value.
- Feature engineering (adding volume) improved model accuracy by approximately 15%, highlighting the importance of geometric features in pricing.
- The deployed Streamlit app enables quick, consistent, and error-free valuation of diamonds — eliminating manual estimation uncertainty.

---

## 🚀 Streamlit App — `diamond_pricer.py`

Run the Streamlit app locally:
```bash
pip install -r requirements.txt
streamlit run app/diamond_pricer.py
```

![diamond_pricer](https://github.com/user-attachments/assets/3fb3048c-967b-40e4-bb35-aeeca213d58f)


---

## ⚙️ Installation & Reproduction

1. Clone the repository:
```bash
git clone https://github.com/yourusername/diamond-price-analysis.git
cd diamond-price-analysis
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Jupyter notebook:
```bash
jupyter notebook notebook/diamond_price_analysis_portfolio.ipynb
```
4. Launch the Streamlit app (optional):
```bash
streamlit run app/diamond_pricer.py
```

---

## 🧾 Requirements

```
pandas
numpy
scikit-learn
xgboost
lightgbm
joblib
streamlit
matplotlib
seaborn
```

---

## 🧾 Author

**Paul Egeonu**  
_Data Analyst & Machine Learning Practitioner_  
[LinkedIn](https://www.linkedin.com/paul-egeonu) | [GitHub](https://github.com/Paul-Egeonu)

---
