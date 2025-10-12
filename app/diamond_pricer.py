
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder

st.title('Diamond Price Estimator')

model_bundle = joblib.load('models/price_predictor.pkl')
model = model_bundle['model']
encoder = model_bundle['encoder']

# Load sample dataset for reference ranges
@st.cache_data
def load_data():
    return pd.read_csv('data/diamonds.csv')

df = load_data()

carat = st.slider('Carat', float(df['carat'].min()), float(df['carat'].quantile(0.99)), float(df['carat'].median()), step=0.01)
cut = st.selectbox('Cut', sorted(df['cut'].unique()))
color = st.selectbox('Color', sorted(df['color'].unique()))
clarity = st.selectbox('Clarity', sorted(df['clarity'].unique()))

def prepare_sample_for_app(sample_dict):
    df_s = pd.DataFrame([sample_dict])
    df_s['volume'] = df_s['x'] * df_s['y'] * df_s['z']
    cat_enc = encoder.transform(df_s[['cut','color','clarity']])
    cat_cols = encoder.get_feature_names_out(['cut','color','clarity'])
    df_cat = pd.DataFrame(cat_enc, columns=cat_cols)
    FEATURES = ['carat','depth','table','x','y','z','volume']
    X = pd.concat([df_s[FEATURES].reset_index(drop=True), df_cat.reset_index(drop=True)], axis=1)
    return X

if st.button('Estimate price'):
    # crude defaults: use medians for remaining numeric features
    sample = {
        'carat': carat,
        'depth': float(df['depth'].median()),
        'table': float(df['table'].median()),
        'x': float((carat ** (1/3)) * 6),
        'y': float((carat ** (1/3)) * 6),
        'z': float((carat ** (1/3)) * 4),
        'cut': cut,
        'color': color,
        'clarity': clarity
    }
    Xs = prepare_sample_for_app(sample)
    pred = model.predict(Xs)[0]
    st.metric('Predicted price (USD)', f"${pred:,.2f}")

    # show similar items
    similar = df[(df['cut']==cut)&(df['color']==color)&(df['clarity']==clarity)].sort_values('carat')
    st.write('Sample similar diamonds (top 10):')
    st.dataframe(similar.drop(columns=['Unnamed: 0'], errors='ignore').head(10))

