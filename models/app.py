import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os

# Load Models and Assets
@st.cache_resource
def load_lr_model(model_dir):
    weights_path = os.path.join(model_dir, 'weights.npy')
    scalers_path = os.path.join(model_dir, 'scalers.pkl')
    col_order_path = os.path.join(model_dir, 'col_order.pkl')
    
    weights = np.load(weights_path)
    with open(scalers_path, 'rb') as f:
        scaling_params = pickle.load(f)
    with open(col_order_path, 'rb') as f:
        col_order = pickle.load(f)
    
    return weights, scaling_params, col_order

@st.cache_resource
def load_fnn_model(model_dir):
    model_path = os.path.join(model_dir, 'fnn_model.keras')
    scaler_x_path = os.path.join(model_dir, 'scaler_X.pkl')
    scaler_y_path = os.path.join(model_dir, 'scaler_y.pkl')
    
    model = tf.keras.models.load_model(model_path)
    with open(scaler_x_path, 'rb') as f:
        scaler_x = pickle.load(f)
    with open(scaler_y_path, 'rb') as f:
        scaler_y = pickle.load(f)
    
    return model, scaler_x, scaler_y

lr_ne_weights, lr_ne_scalers, lr_ne_cols = load_lr_model('saved_model/normal_equation')
lr_gd_weights, lr_gd_scalers, lr_gd_cols = load_lr_model('saved_model/gradient_descent')
fnn_basic_model, fnn_basic_scaler_x, fnn_basic_scaler_y = load_fnn_model('saved_model/fnn_basic')
fnn_tuned_model, fnn_tuned_scaler_x, fnn_tuned_scaler_y = load_fnn_model('saved_model/fnn_tuned')

# Preprocess Inputs
def preprocess_lr_input(input_data, col_order, scalers):
    input_df = pd.DataFrame([input_data], columns=col_order[1:])  # Exclude bias
    input_df.insert(0, 'bias', 1.0)  # Add bias term
    input_array = input_df[col_order].values
    input_scaled = (input_array - scalers['mean']) / scalers['std']
    return input_scaled

def preprocess_fnn_input(input_data, scaler_x, col_order):
    input_df = pd.DataFrame([input_data], columns=col_order[1:])  # Exclude bias
    input_array = input_df.values
    input_scaled = scaler_x.transform(input_array)
    return input_scaled

# Make Predictions
def predict_lr(input_scaled, weights, scalers):
    pred_scaled = input_scaled @ weights
    pred = pred_scaled * scalers['y_std'] + scalers['y_mean']
    return pred[0]

def predict_fnn(input_scaled, model, scaler_y):
    pred_scaled = model.predict(input_scaled, verbose=0)
    pred = scaler_y.inverse_transform(pred_scaled)
    return pred[0]

# Compare Models
def compare_models(input_data):
    # Preprocess inputs for each model
    lr_ne_input = preprocess_lr_input(input_data, lr_ne_cols, lr_ne_scalers)
    lr_gd_input = preprocess_lr_input(input_data, lr_gd_cols, lr_gd_scalers)
    fnn_basic_input = preprocess_fnn_input(input_data, fnn_basic_scaler_x, lr_ne_cols)
    fnn_tuned_input = preprocess_fnn_input(input_data, fnn_tuned_scaler_x, lr_ne_cols)

    # Make predictions
    lr_ne_pred = predict_lr(lr_ne_input, lr_ne_weights, lr_ne_scalers)
    lr_gd_pred = predict_lr(lr_gd_input, lr_gd_weights, lr_gd_scalers)
    fnn_basic_pred = predict_fnn(fnn_basic_input, fnn_basic_model, fnn_basic_scaler_y)
    fnn_tuned_pred = predict_fnn(fnn_tuned_input, fnn_tuned_model, fnn_tuned_scaler_y)

    # Compile results
    results = {
        'Model': ['LR Normal Eq.', 'LR Gradient Descent', 'FNN Basic', 'FNN Tuned'],
        'Goals': [lr_ne_pred[0], lr_gd_pred[0], fnn_basic_pred[0], fnn_tuned_pred[0]],
        'Assists': [lr_ne_pred[1], lr_gd_pred[1], fnn_basic_pred[1], fnn_tuned_pred[1]]
    }
    return pd.DataFrame(results)

# Streamlit App
st.title("Player Goals and Assists Prediction Comparison")

# Sample input data for testing (replace with your test dataset or actual data)
sample_input = {
    'season': 2023,
    'age_at_season_start': 25,
    'height_in_cm': 180,
    'appearances_season': 30,
    'total_minutes_played_season': 2700,
    'position_Attack': 1,
    'position_Defender': 0,
    'position_Midfield': 0,
    'foot_left': 1,
    'foot_right': 0
}

# Compare model performance
try:
    results_df = compare_models(sample_input)
    
    # Visualize Results
    st.header("Model Performance Comparison")
    st.table(results_df.style.format({'Goals': '{:.2f}', 'Assists': '{:.2f}'}))
    
    # Bar chart for visualization
    st.subheader("Model Comparison")
    st.bar_chart(results_df.set_index('Model')[['Goals', 'Assists']])
except Exception as e:
    st.error(f"Error in prediction: {str(e)}")

# Deploy Instructions
# Save this file as app.py
# Create requirements.txt with:
# streamlit
# pandas
# numpy
# tensorflow
# scikit-learn
# Run locally: `streamlit run app.py`
# Deploy to Streamlit Cloud:
# 1. Push to GitHub with app.py, requirements.txt, and saved_model folder
# 2. Connect to Streamlit Cloud, select repo, and deploy