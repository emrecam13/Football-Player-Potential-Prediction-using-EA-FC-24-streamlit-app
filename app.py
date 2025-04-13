import streamlit as st
import pandas as pd
import numpy as np
import os
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
# Random Forest is replaced by LightGBM.
saved_models = {
    "Attacker": {
        "Linear Regression": "Attacker_Linear_Regression_model.sav",
        "LightGBM": "Attacker_LightGBM_model.sav",
        "SVR": "Attacker_SVR_model.sav",
        "XGBoost": "Attacker_XGBoost_model.sav"
    },
    "Midfielder": {
        "Linear Regression": "Midfielder_Linear_Regression_model.sav",
        "LightGBM": "Midfielder_LightGBM_model.sav",
        "SVR": "Midfielder_SVR_model.sav",
        "XGBoost": "Midfielder_XGBoost_model.sav"
    },
    "Defender": {
        "Linear Regression": "Defender_Linear_Regression_model.sav",
        "LightGBM": "Defender_LightGBM_model.sav",
        "SVR": "Defender_SVR_model.sav",
        "XGBoost": "Defender_XGBoost_model.sav"
    },
    "Goalkeeper": {
        "Linear Regression": "Goalkeeper_Linear_Regression_model.sav",
        "LightGBM": "Goalkeeper_LightGBM_model.sav",
        "SVR": "Goalkeeper_SVR_model.sav",
        "XGBoost": "Goalkeeper_XGBoost_model.sav"
    }
}


# Get the working directory (the directory of the current file, e.g., main.py)
working_dir = os.path.dirname(os.path.abspath(__file__))

# ----------------------------
# Load Attacker Models
# ----------------------------
attacker_lr_model = pickle.load(open(f'{working_dir}/saved_models/Attacker_Linear_Regression_model.sav', 'rb'))
attacker_lgb_model = pickle.load(open(f'{working_dir}/saved_models/Attacker_LightGBM_model.sav', 'rb'))
attacker_svr_model = pickle.load(open(f'{working_dir}/saved_models/Attacker_SVR_model.sav', 'rb'))
attacker_xgb_model = pickle.load(open(f'{working_dir}/saved_models/Attacker_XGBoost_model.sav', 'rb'))

attacker_models = {
    "Linear Regression": attacker_lr_model,
    "LightGBM": attacker_lgb_model,
    "SVR": attacker_svr_model,
    "XGBoost": attacker_xgb_model
}

# ----------------------------
# Load Midfielder Models
# ----------------------------
midfielder_lr_model = pickle.load(open(f'{working_dir}/saved_models/Midfielder_Linear_Regression_model.sav', 'rb'))
midfielder_lgb_model = pickle.load(open(f'{working_dir}/saved_models/Midfielder_LightGBM_model.sav', 'rb'))
midfielder_svr_model = pickle.load(open(f'{working_dir}/saved_models/Midfielder_SVR_model.sav', 'rb'))
midfielder_xgb_model = pickle.load(open(f'{working_dir}/saved_models/Midfielder_XGBoost_model.sav', 'rb'))

midfielder_models = {
    "Linear Regression": midfielder_lr_model,
    "LightGBM": midfielder_lgb_model,
    "SVR": midfielder_svr_model,
    "XGBoost": midfielder_xgb_model
}
# ----------------------------
# Load Defender Models
# ----------------------------
defender_lr_model = pickle.load(open(f'{working_dir}/saved_models/Defender_Linear_Regression_model.sav', 'rb'))
defender_lgb_model = pickle.load(open(f'{working_dir}/saved_models/Defender_LightGBM_model.sav', 'rb'))
defender_svr_model = pickle.load(open(f'{working_dir}/saved_models/Defender_SVR_model.sav', 'rb'))
defender_xgb_model = pickle.load(open(f'{working_dir}/saved_models/Defender_XGBoost_model.sav', 'rb'))

defender_models = {
    "Linear Regression": defender_lr_model,
    "LightGBM": defender_lgb_model,
    "SVR": defender_svr_model,
    "XGBoost": defender_xgb_model
}

# ----------------------------
# Load Goalkeeper Models
# ----------------------------
goalkeeper_lr_model = pickle.load(open(f'{working_dir}/saved_models/Goalkeeper_Linear_Regression_model.sav', 'rb'))
goalkeeper_lgb_model = pickle.load(open(f'{working_dir}/saved_models/Goalkeeper_LightGBM_model.sav', 'rb'))
goalkeeper_svr_model = pickle.load(open(f'{working_dir}/saved_models/Goalkeeper_SVR_model.sav', 'rb'))
goalkeeper_xgb_model = pickle.load(open(f'{working_dir}/saved_models/Goalkeeper_XGBoost_model.sav', 'rb'))

goalkeeper_models = {
    "Linear Regression": goalkeeper_lr_model,
    "LightGBM": goalkeeper_lgb_model,
    "SVR": goalkeeper_svr_model,
    "XGBoost": goalkeeper_xgb_model
}


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
    import pandas as pd

    # Load the compressed Pickle file instead of reading from a zip
    df_model = pd.read_pickle("dataset/male_players_compressed.pkl.bz2", compression="bz2")
    df_result = pd.read_pickle("dataset/df_model_result_compressed.pkl.gz", compression="gzip")
    st.subheader("Data Sample")
    st.dataframe(df_model.head())

    st.subheader("Basic Statistics")
    st.write(df_model.describe())

    st.subheader("Result Sample")
    st.dataframe(df_result.head())

    # Add more charts as needed...

# ============================
# 4. Model Evaluation Module
# ============================
elif module == "Model Evaluation":
    st.title("Model Evaluation & Comparisons")
    st.write("This module displays precomputed evaluation results for various models.")

    # Predefined CV Results
    baseline_results = {
        "Attacker": {"Linear Regression": {'rmse': 2.65, 'mae': 2.06, 'r2': 0.826}},
        "Midfielder": {"Linear Regression": {'rmse': 2.60, 'mae': 2.02, 'r2': 0.831}},
        "Defender": {"Linear Regression": {'rmse': 2.42, 'mae': 1.86, 'r2': 0.837}},
        "Goalkeeper": {"Linear Regression": {'rmse': 2.61, 'mae': 2.01, 'r2': 0.837}}
    }

    results_nonlin = {
        "Attacker": {"LightGBM": {'rmse': 1.87, 'mae': 1.17, 'r2': 0.913},
                     "SVR": {'rmse': 2.03, 'mae': 1.44, 'r2': 0.898},
                     "XGBoost": {'rmse': 1.89, 'mae': 1.18, 'r2': 0.912}},
        "Midfielder": {"LightGBM": {'rmse': 1.75, 'mae': 1.12, 'r2': 0.923},
                       "SVR": {'rmse': 1.93, 'mae': 1.36, 'r2': 0.907},
                       "XGBoost": {'rmse': 1.76, 'mae': 1.14, 'r2': 0.922}},
        "Defender": {"LightGBM": {'rmse': 1.66, 'mae': 1.06, 'r2': 0.923},
                     "SVR": {'rmse': 1.79, 'mae': 1.25, 'r2': 0.910},
                     "XGBoost": {'rmse': 1.67, 'mae': 1.08, 'r2': 0.922}},
        "Goalkeeper": {"LightGBM": {'rmse': 1.71, 'mae': 1.12, 'r2': 0.930},
                       "SVR": {'rmse': 2.02, 'mae': 1.45, 'r2': 0.902},
                       "XGBoost": {'rmse': 1.73, 'mae': 1.12, 'r2': 0.928}}
    }

    # Combine results
    positions = ["Attacker", "Midfielder", "Defender", "Goalkeeper"]
    st.write("### Model Comparison Results (CV Predictions)")
    for pos in positions:
        st.write(f"**{pos} Group:**")
        st.write("LightGBM: RMSE = {:.2f}, MAE = {:.2f}, R² = {:.3f}".format(
            results_nonlin[pos]["LightGBM"]['rmse'],
            results_nonlin[pos]["LightGBM"]['mae'],
            results_nonlin[pos]["LightGBM"]['r2']
        ))
        st.write("SVR: RMSE = {:.2f}, MAE = {:.2f}, R² = {:.3f}".format(
            results_nonlin[pos]["SVR"]['rmse'],
            results_nonlin[pos]["SVR"]['mae'],
            results_nonlin[pos]["SVR"]['r2']
        ))
        st.write("XGBoost: RMSE = {:.2f}, MAE = {:.2f}, R² = {:.3f}".format(
            results_nonlin[pos]["XGBoost"]['rmse'],
            results_nonlin[pos]["XGBoost"]['mae'],
            results_nonlin[pos]["XGBoost"]['r2']
        ))
        st.write("Linear Regression: RMSE = {:.2f}, MAE = {:.2f}, R² = {:.3f}".format(
            baseline_results[pos]["Linear Regression"]['rmse'],
            baseline_results[pos]["Linear Regression"]['mae'],
            baseline_results[pos]["Linear Regression"]['r2']
        ))

# ============================
# 5. Interactive Prediction Module
# ============================
elif module == "Interactive Prediction":
    st.title("Interactive Prediction Module")

    # Allow the user to select a position
    position = st.selectbox("Select Position", ["Attacker", "Midfielder", "Defender", "Goalkeeper"])

    # Allow the user to select a model (Random Forest replaced with LightGBM)
    model_choice = st.selectbox("Select Model", ["Linear Regression", "LightGBM", "SVR", "XGBoost"])

    st.subheader(f"Input values for the {position} attributes:")
    attributes = selected_features[position]
    user_input = {}
    for attr in attributes:
        user_input[attr] = st.number_input(f"Enter value for {attr}", min_value=0, max_value=100, value=50)

    if st.button("Predict Potential"):
        def load_model(filename):
            path = os.path.join(os.getcwd(), 'saved_models', filename)
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except FileNotFoundError:
                st.error(f"❌ Model file not found: {path}")
            except Exception as e:
                st.error(f"⚠️ Error loading model: {e}")
        # Load the corresponding model file for the selected position and model
        model_filename = saved_models[position][model_choice]
        print("Model filename:", model_filename)
        model = load_model(model_filename)
        # Convert user input to DataFrame ensuring the feature order is maintained
        X_new = pd.DataFrame([user_input], columns=attributes)
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
    - **Model Evaluation & Comparisons:** Shows precomputed evaluation metrics (RMSE, MAE, R²) for each model per position group.
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
