import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="Football G&A Predictor",
    page_icon="⚽",
    layout="wide"
)

# --- Caching Functions for Loading Models and Data ---

@st.cache_resource
def load_linear_model(model_name):
    """Loads a linear model's weights, scalers, and column order."""
    base_path = os.path.join('models', 'saved_model', model_name)
    with open(os.path.join(base_path, 'weights.npy'), 'rb') as f:
        weights = np.load(f)
    with open(os.path.join(base_path, 'scalers.pkl'), 'rb') as f:
        scalers = pickle.load(f)
    with open(os.path.join(base_path, 'col_order.pkl'), 'rb') as f:
        col_order = pickle.load(f)
    return weights, scalers, col_order

@st.cache_resource
def load_fnn_model(model_name):
    """Loads a Keras FNN model, scalers, and column order."""
    base_path = os.path.join('models', 'saved_model', model_name)
    model = tf.keras.models.load_model(os.path.join(base_path, 'fnn_model.keras'))
    with open(os.path.join(base_path, 'scaler_X.pkl'), 'rb') as f:
        scaler_X = pickle.load(f)
    with open(os.path.join(base_path, 'scaler_y.pkl'), 'rb') as f:
        scaler_y = pickle.load(f)
    with open(os.path.join(base_path, 'col_order.pkl'), 'rb') as f:
        col_order = pickle.load(f)
    return model, scaler_X, scaler_y, col_order

@st.cache_data
def load_full_data():
    """Loads the full raw dataset."""
    df = pd.read_csv('models/goals_predict.csv')
    return df

# --- Prediction and Preprocessing Logic ---

def preprocess_input(df, col_order, scaler_X=None, manual_scalers=None):
    """Preprocesses raw input DataFrame for a specific model."""
    # Define the fixed list of numerical columns that require scaling
    numerical_cols_to_scale = ['season', 'age_at_season_start', 'height_in_cm', 'appearances_season', 'total_minutes_played_season']

    # One-hot encode categorical features
    df_processed = pd.get_dummies(df, columns=['position', 'foot'], dtype=int)
    
    # Add bias term if it's in the column order (for linear models)
    if 'bias' in col_order and 'bias' not in df_processed.columns:
        df_processed.insert(0, 'bias', 1)

    # Align columns to the model's expected order, filling missing columns with 0
    df_aligned = df_processed.reindex(columns=col_order, fill_value=0)
    
    # Scale the features using the appropriate scaler
    if scaler_X:  # For FNN models (scikit-learn scaler)
        df_aligned[numerical_cols_to_scale] = scaler_X.transform(df_aligned[numerical_cols_to_scale])
    elif manual_scalers:  # For Linear models (manual scalers)
        df_aligned[numerical_cols_to_scale] = (df_aligned[numerical_cols_to_scale] - manual_scalers['mean']) / manual_scalers['std']
        
    return df_aligned.values.astype(float)

def predict(model_name, input_df):
    """Generic prediction function for any model."""
    if 'fnn' in model_name:
        model, scaler_X, scaler_y, col_order = load_fnn_model(model_name)
        X_processed = preprocess_input(input_df.copy(), col_order, scaler_X=scaler_X)
        y_pred_scaled = model.predict(X_processed)
        y_pred = np.expm1(scaler_y.inverse_transform(y_pred_scaled))
    else:
        weights, scalers, col_order = load_linear_model(model_name)
        X_processed = preprocess_input(input_df.copy(), col_order, manual_scalers=scalers)
        y_pred_scaled = X_processed.dot(weights)
        y_pred_log = y_pred_scaled * scalers['y_std'] + scalers['y_mean']
        y_pred = np.expm1(y_pred_log)
        
    y_pred[y_pred < 0] = 0
    return y_pred

# --- Page Implementations ---

def page_summary():
    st.title("⚽ Project Summary")
    st.markdown("""
    This project aims to predict the number of goals and assists a football player will have in the next season based on their historical performance and attributes.

    ### Key Objectives:
    - **Data Processing:** Clean and prepare player data for modeling.
    - **Model Training:** Develop and train several regression models to predict performance.
    - **Model Comparison:** Objectively evaluate the models to find the most accurate and reliable one.
    - **Interactive Prediction:** Build a tool to make predictions for individual players.

    ### Models Compared:
    We are comparing four different models to see which performs best:
    1.  **Linear Regression (Normal Equation):** A direct, analytical solution for linear regression. Very fast but assumes a linear relationship.
    2.  **Linear Regression (Gradient Descent):** An iterative optimization approach. Slower but more scalable than the Normal Equation.
    3.  **Feedforward Neural Network (FNN - Basic):** A simple neural network to capture non-linear patterns in the data.
    4.  **FNN (Tuned):** A more complex FNN whose architecture was optimized using a hyperparameter search (Keras Tuner).

    Use the navigation on the left to explore the model comparison and make your own predictions!
    """)

def page_model_comparison():
    st.title("📊 Model Comparison")
    st.markdown("Evaluating model performance on the unseen test dataset (`20%` of original data).")

    with st.spinner("Loading test data and evaluating models..."):
        full_df = load_full_data()
        _, test_df = train_test_split(full_df, test_size=0.2, random_state=42)
        X_test = test_df.drop(columns=['total_goals_season', 'total_assists_season', 'avg_market_value_season'])
        y_test = test_df[['total_goals_season', 'total_assists_season']].values

        models = {
            "Linear Regression (Normal Eq.)": "normal_equation",
            "Linear Regression (Gradient Descent)": "gradient_descent",
            "FNN (Basic)": "fnn_basic",
            "FNN (Tuned)": "fnn_tuned"
        }

        results = []
        predictions_dict = {}
        for display_name, model_name in models.items():
            predictions = predict(model_name, X_test)
            predictions_dict[display_name] = predictions
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            results.append({"Model": display_name, "MSE": mse, "MAE": mae})
        
        results_df = pd.DataFrame(results).set_index('Model')

    st.header("1. Performance Metrics")
    st.dataframe(results_df.style.highlight_min(axis=0, color='lightgreen'))
    
    st.header("2. Visual Comparison of Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Mean Squared Error (MSE)")
        st.bar_chart(results_df['MSE'])
    with col2:
        st.markdown("##### Mean Absolute Error (MAE)")
        st.bar_chart(results_df['MAE'])

    st.header("3. Prediction Analysis (FNN Tuned Model)")
    st.markdown("Let's take a closer look at the best performing model.")
    
    best_model_preds = predictions_dict["FNN (Tuned)"]
    residuals = y_test - best_model_preds
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Predicted vs. Actual Goals")
        fig = px.scatter(x=y_test[:, 0], y=best_model_preds[:, 0], labels={'x': 'Actual Goals', 'y': 'Predicted Goals'}, opacity=0.5)
        fig.add_shape(type='line', x0=0, y0=0, x1=y_test[:, 0].max(), y1=y_test[:, 0].max(), line=dict(color='red', dash='dash'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("##### Residual Distribution (Goals)")
        fig = px.histogram(residuals[:, 0], nbins=50, labels={'value': 'Error (Actual - Predicted)'})
        st.plotly_chart(fig, use_container_width=True)

def page_player_prediction():
    st.title("🔮 Player Performance Prediction")
    df = load_full_data()
    
    from difflib import get_close_matches

    # ✅ Single box: searchable dropdown
    player_names = sorted(df['name'].unique())
    selected_player = st.selectbox(
        "Search for a player by name:",
        options=player_names,
        index=0,
        placeholder="Type to search..."
    )

    
    if selected_player:
        player_data = df[df['name'] == selected_player].sort_values('season')
        
        st.header(f"Historical Stats for {selected_player}")
        st.dataframe(player_data[['season', 'age_at_season_start', 'appearances_season', 'total_minutes_played_season', 'total_goals_season', 'total_assists_season']])

        st.subheader("Performance Over Time")
        fig = px.line(player_data, x='season', y=['total_goals_season', 'total_assists_season'], markers=True)
        st.plotly_chart(fig, use_container_width=True)

        st.header(f"Predict Next Season's Performance")
        if st.button(f"Predict for {player_data['season'].max() + 1} Season", type="primary"):
            last_season = player_data.iloc[-1]
            next_season_input = last_season.copy()
            next_season_input['season'] += 1
            next_season_input['age_at_season_start'] += 1
            
            # Drop columns not needed for prediction
            input_df = next_season_input.drop(['total_goals_season', 'total_assists_season', 'avg_market_value_season']).to_frame().T

            models = {
                "Linear Regression (Normal Eq.)": "normal_equation",
                "Linear Regression (Gradient Descent)": "gradient_descent",
                "FNN (Basic)": "fnn_basic",
                "FNN (Tuned)": "fnn_tuned"
            }
            
            cols = st.columns(len(models))
            for idx, (display_name, model_name) in enumerate(models.items()):
                with cols[idx]:
                    st.markdown(f"##### {display_name}")
                    prediction = predict(model_name, input_df)
                    st.metric(label="Predicted Goals", value=f"{prediction[0, 0]:.2f}")
                    st.metric(label="Predicted Assists", value=f"{prediction[0, 1]:.2f}")

# --- Main App Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Project Summary", "Model Comparison", "Player Prediction"])

if page == "Project Summary":
    page_summary()
elif page == "Model Comparison":
    page_model_comparison()
elif page == "Player Prediction":
    page_player_prediction()

st.sidebar.markdown("---")
st.sidebar.info("App created to compare regression models for football analytics.")
