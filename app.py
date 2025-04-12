import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ============================
# 1. Configuration and Setup
# ============================

# Define the selected features for each position (as used during training)
selected_features = {
    "Attacker": [
        'overall',
        'age',
        'skill_ball_control',
        'skill_dribbling',
        'attacking_short_passing',
        'attacking_finishing',
        'mentality_positioning',
        'movement_reactions',
        'power_shot_power',
        'power_long_shots',
        'mentality_vision'
    ],
    "Midfielder": [
        'overall',
        'age',
        'skill_ball_control',
        'skill_dribbling',
        'attacking_short_passing',
        'mentality_vision',
        'movement_reactions',
        'skill_long_passing',
        'mentality_composure',
        'mentality_positioning',
        'power_shot_power'
    ],
    "Defender": [
        'overall',
        'age',
        'defending_standing_tackle',
        'defending_sliding_tackle',
        'mentality_interceptions',
        'defending_marking_awareness',
        'movement_reactions',
        'attacking_short_passing',
        'skill_ball_control',
        'mentality_composure',
        'attacking_heading_accuracy'
    ],
    "Goalkeeper": [
        'overall',
        'age',
        'goalkeeping_reflexes',
        'goalkeeping_diving',
        'goalkeeping_handling',
        'goalkeeping_positioning',
        'goalkeeping_kicking',
        'movement_reactions',
        'mentality_composure',
        'mentality_vision',
        'goalkeeping_speed'
    ]
}

# Mapping of each position and each model to its saved pickle filename.
# (Ensure you have these files saved beforehand.)
model_files = {
    "Attacker": {
        "Linear Regression": "Attacker_LR_model.sav",
        "Random Forest": "Attacker_Random_Forest_model.sav",
        "SVR": "Attacker_SVR_model.sav",
        "XGBoost": "Attacker_XGBoost_model.sav"
    },
    "Midfielder": {
        "Linear Regression": "Midfielder_LR_model.sav",
        "Random Forest": "Midfielder_Random_Forest_model.sav",
        "SVR": "Midfielder_SVR_model.sav",
        "XGBoost": "Midfielder_XGBoost_model.sav"
    },
    "Defender": {
        "Linear Regression": "Defender_LR_model.sav",
        "Random Forest": "Defender_Random_Forest_model.sav",
        "SVR": "Defender_SVR_model.sav",
        "XGBoost": "Defender_XGBoost_model.sav"
    },
    "Goalkeeper": {
        "Linear Regression": "Goalkeeper_LR_model.sav",
        "Random Forest": "Goalkeeper_Random_Forest_model.sav",
        "SVR": "Goalkeeper_SVR_model.sav",
        "XGBoost": "Goalkeeper_XGBoost_model.sav"
    }
}


# Utility function to load a model from a pickle file
def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model


# ============================
# 2. Streamlit Sidebar Navigation
# ============================
st.sidebar.title("Navigation")
module = st.sidebar.selectbox("Select Module",
                              ["Overview & Data Summary", "Model Evaluation", "Interactive Prediction", "About"])

# ============================
# 3. Overview & Data Summary Module
# ============================
if module == "Overview & Data Summary":
    st.title("Player Potential Prediction App")
    st.header("Overview & Data Summary")
    st.write("This module displays a brief summary of the dataset.")

    # Load your main DataFrame (assume it's saved as df_model)
    # For demonstration, we'll read from a CSV; adjust accordingly.
    df_model = pd.read_csv("male_players.csv")
    st.subheader("Data Sample")
    st.dataframe(df_model.head())

    st.subheader("Basic Statistics")
    st.write(df_model.describe())

    # Add more charts as needed...

# ============================
# 4. Model Evaluation Module
# ============================
elif module == "Model Evaluation":
    st.title("Model Evaluation & Comparisons")
    st.write("This module displays precomputed evaluation results for various models.")

    # Assume you have your results stored in a dictionary (similar to previous examples)
    # For demonstration, we simply display a table.
    # You might load your evaluation results from a pickle or CSV file.
    baseline_results = {
        "Attacker": {"Linear Regression": {'rmse': 2.65, 'mae': 2.06, 'r2': 0.825}},
        "Midfielder": {"Linear Regression": {'rmse': 2.60, 'mae': 2.02, 'r2': 0.830}},
        "Defender": {"Linear Regression": {'rmse': 2.42, 'mae': 1.86, 'r2': 0.837}},
        "Goalkeeper": {"Linear Regression": {'rmse': 2.61, 'mae': 2.01, 'r2': 0.837}}
    }

    results_nonlin = {
        "Attacker": {"RF": {'rmse': 1.94, 'mae': 1.20, 'r2': 0.906},
                     "SVR": {'rmse': 2.03, 'mae': 1.44, 'r2': 0.898},
                     "XGBoost": {'rmse': 1.89, 'mae': 1.18, 'r2': 0.912}},
        "Midfielder": {"RF": {'rmse': 1.82, 'mae': 1.15, 'r2': 0.917},
                       "SVR": {'rmse': 1.93, 'mae': 1.36, 'r2': 0.907},
                       "XGBoost": {'rmse': 1.76, 'mae': 1.14, 'r2': 0.921}},
        "Defender": {"RF": {'rmse': 1.73, 'mae': 1.09, 'r2': 0.917},
                     "SVR": {'rmse': 1.79, 'mae': 1.25, 'r2': 0.910},
                     "XGBoost": {'rmse': 1.67, 'mae': 1.08, 'r2': 0.922}},
        "Goalkeeper": {"RF": {'rmse': 1.79, 'mae': 1.14, 'r2': 0.924},
                       "SVR": {'rmse': 2.02, 'mae': 1.45, 'r2': 0.902},
                       "XGBoost": {'rmse': 1.73, 'mae': 1.12, 'r2': 0.928}}
    }

    # Combine results for display
    positions = ["Attacker", "Midfielder", "Defender", "Goalkeeper"]
    st.write("### Model Comparison Results (CV Predictions)")
    for pos in positions:
        st.write(f"**{pos} Group:**")
        # Display results for non-linear models first
        st.write("Random Forest: RMSE = 1.94, MAE = 1.20, R² = 0.906")
        st.write("SVR: RMSE = 2.03, MAE = 1.44, R² = 0.898")
        st.write("XGBoost: RMSE = 1.89, MAE = 1.18, R² = 0.912")
        # Then display baseline linear regression at bottom
        st.write("Linear Regression: RMSE = 2.65, MAE = 2.06, R² = 0.825")

# ============================
# 5. Interactive Prediction Module
# ============================
elif module == "Interactive Prediction":
    st.title("Interactive Prediction Module")

    # Allow the user to select a position
    position = st.selectbox("Select Position", ["Attacker", "Midfielder", "Defender", "Goalkeeper"])

    # Allow the user to select a model
    model_choice = st.selectbox("Select Model", ["Linear Regression", "Random Forest", "SVR", "XGBoost"])

    # Show the corresponding 11 attributes for the selected position
    st.subheader(f"Input values for {position} attributes:")
    attributes = selected_features[position]
    user_input = {}
    for attr in attributes:
        user_input[attr] = st.number_input(f"Enter value for {attr}", min_value=0, max_value=100, value=50)

    # When the user clicks the predict button
    if st.button("Predict Potential"):
        # Load the corresponding model file for the selected position and model
        model_filename = model_files[position][model_choice]
        model = load_model(model_filename)
        # Convert user input to DataFrame ensuring the same attribute order
        X_new = pd.DataFrame([user_input], columns=attributes)
        # Predict using the loaded model
        prediction = model.predict(X_new)
        st.success(f"Predicted Potential for {position} using {model_choice}: {prediction[0]:.2f}")

# ============================
# 6. About Module
# ============================
elif module == "About":
    st.title("About the Player Potential Prediction App")
    st.write("""
    This app is designed to predict the potential of football players using various machine learning models.

    **Modules:**
    - **Overview & Data Summary:** Displays a summary and basic statistics of the dataset.
    - **Model Evaluation & Comparisons:** Shows cross-validated evaluation metrics (RMSE, MAE, R²) for each model per position group.
    - **Interactive Prediction:** Allows you to select a position and model, input the corresponding 11 attributes, and get a predicted potential.
    - **About:** Provides details about the app, dataset, and model methodology.
    """)
    st.write("Contact: emrecam13@gmail.com")

# ============================
# Deployment Instruction (Optional)
# ============================
st.sidebar.markdown("### Deployment")
st.sidebar.info("""
To deploy this app using ngrok, run:
""")
