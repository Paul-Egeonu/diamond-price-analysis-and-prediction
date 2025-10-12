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

<img width="1200" height="900" alt="carat_price_scatter" src="https://github.com/user-attachments/assets/0159b74a-4d5a-4ea7-a527-4c5ad2b66deb" />


4. **Feature Engineering**  
   - Created `volume = x * y * z` feature.  
   - One-hot encoded categorical features (`cut`, `color`, `clarity`).  
   - Scaled numerical features for consistency.

5. **Model Training**  
   - Compared multiple regressors (Linear, RandomForest, XGBoost).  
   - Final model: **XGBoostRegressor** (best RMSE & R²).  

6. **Evaluation**  
   - Computed RMSE, MAE, and R² metrics.  
   - Interpreted model performance and business significance.  

7. **Deployment**  
   - Streamlit app (`diamond_pricer.py`) for interactive price prediction.  
   - Model and preprocessing pipeline saved using `joblib`.

   ![Streamlit App Demo](images/streamlit_app_demo.gif)

---

## 🧩 Selected Notebook Snippets & Explanations

### 🔹 Data Load & Initial Inspection
```python
import pandas as pd
df = pd.read_csv('data/diamond_price_data.csv')
df.head()
```
**Explanation:** Load and preview the dataset to confirm structure, datatypes, and detect anomalies like zero dimensions.

### 🔹 EDA — Price Distribution
```python
import seaborn as sns
sns.histplot(df['price'], bins=60, kde=True)
```
**Interpretation:** Price is right-skewed; consider log-transform (`np.log(price)`) to stabilize variance for linear models.

### 🔹 Feature Engineering
```python
df['volume'] = df['x'] * df['y'] * df['z']
df = pd.get_dummies(df, columns=['cut','color','clarity'], drop_first=True)
```
**Interpretation:** Derived volume captures 3D size, while one-hot encoding converts categorical features into numeric form.

### 🔹 Modeling (XGBoost Example)
```python
from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```
**Interpretation:** XGBoost captures non-linear relationships (e.g., `carat` × `clarity`). Model tuned via cross-validation.

### 🔹 Evaluation
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.3f}')
```
**Interpretation:** RMSE and MAE quantify prediction error in USD; R² shows variance explained. Compare RMSE to average price to estimate relative error.

---

## 📈 Model Performance Summary

| Metric | Value |
|---------|--------|
| **RMSE** | 541.23 |
| **MAE** | 312.45 |
| **R²** | 0.982 |

✅ Model explains ~98% of variance in price with low average error.

![Feature Importance](images/feature_importance.png)

---

## 🧠 Insights

- `carat` has the strongest correlation with price.  
- Higher `clarity` and `cut` grades drive exponential increases in price.  
- Feature engineering (adding volume) improved model accuracy by ~15%.  
- The app enables quick valuation without manual estimation errors.

---

## 🚀 Streamlit App — `diamond_pricer.py`

Run the Streamlit app locally:
```bash
pip install -r requirements.txt
streamlit run app/diamond_pricer.py
```

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

![ERD](images/Diamond_ERD.png)
