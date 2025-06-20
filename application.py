import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime

# --- Load Model and Application Data ---
try:
    with open('artifacts/regressor_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('artifacts/app_data.pkl', 'rb') as f:
        app_data = pickle.load(f)
        
    TOWN_OPTIONS = app_data['town_unique']
    FLAT_TYPE_OPTIONS = app_data['flat_type_unique']
    STOREY_RANGE_OPTIONS = app_data['storey_range_options']

except FileNotFoundError:
    st.error("Model or app data not found. Please run `model_training.py` first.")
    st.stop()


# --- Helper Function ---
def get_avg_storey(storey_range):
    """Calculates the average storey from a range string."""
    parts = storey_range.split(' TO ')
    return (int(parts[0]) + int(parts[1])) / 2


# --- Streamlit UI ---
st.set_page_config(page_title="Singapore HDB Price Predictor", layout="wide")

st.title("Singapore Resale Flat Price Predictor 🇸🇬")

st.markdown("""
This application predicts the resale price of HDB flats in Singapore using a machine learning model. 
Please provide the details of the flat below.
""")

# --- User Input Fields in Columns ---
col1, col2 = st.columns(2)

with col1:
    town = st.selectbox("Town", options=TOWN_OPTIONS, index=TOWN_OPTIONS.index('ANG MO KIO'))
    flat_type = st.selectbox("Flat Type", options=FLAT_TYPE_OPTIONS, index=FLAT_TYPE_OPTIONS.index('3 ROOM'))
    storey_range = st.selectbox("Storey Range", options=STOREY_RANGE_OPTIONS, index=5)
    
with col2:
    floor_area = st.number_input("Floor Area (sqm)", min_value=20.0, max_value=200.0, value=68.0, step=1.0)
    lease_commence_date = st.number_input("Lease Commence Date (Year)", min_value=1960, max_value=2023, value=1990)
    sale_year = st.slider("Sale Year", min_value=2015, max_value=datetime.date.today().year + 5, value=datetime.date.today().year)

# --- Prediction Logic ---
if st.button("Predict Resale Price", type="primary"):
    
    # Calculate derived features
    storey_avg = get_avg_storey(storey_range)
    remaining_lease_years = 99 - (sale_year - lease_commence_date)
    
    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'town': [town],
        'flat_type': [flat_type],
        'floor_area_sqm': [floor_area],
        'lease_commence_date': [lease_commence_date],
        'sale_year': [sale_year],
        'remaining_lease_years': [remaining_lease_years],
        'storey_avg': [storey_avg]
    })
    
    # Reorder columns to match training data order
    # This is crucial for the preprocessor
    # The saved model pipeline knows the correct order from training
    
    # Make prediction
    try:
        log_price_prediction = model.predict(input_data)
        
        # Inverse transform the prediction
        price_prediction = np.expm1(log_price_prediction)[0]
        
        st.success(f"**Predicted Resale Price: S$ {price_prediction:,.2f}**")
        
        st.info("Disclaimer: This is an estimated price. Actual market prices may vary.")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
