import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

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

# =============================
# 3. Overview & Data Summary Module
# =============================
if module == "Overview & Data Summary":
    st.title("Player Potential Prediction App")
    st.header("Overview & Data Summary")
    st.write("Below you’ll see first the _raw_ dataset that you prepared, then—separately—the model results you generated.")

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 1) Load raw features
    df_model = pd.read_pickle("dataset/male_players_compressed.pkl.bz2", compression="bz2")

    # 2) Show raw data
    st.subheader("🔹 Raw Data Sample")
    st.dataframe(df_model.head())

    st.subheader("🔹 Raw Data Basic Statistics")
    st.write(df_model.describe())

    # 3) Visualize raw features
    # 3a) Distributions
    st.subheader("🔹 Distributions of Raw Numeric Features")
    raw_num_cols = ["age","height_cm","weight_kg","overall","potential"]
    fig, axes = plt.subplots(len(raw_num_cols),1,figsize=(8,4*len(raw_num_cols)))
    for ax,col in zip(axes, raw_num_cols):
        sns.histplot(df_model[col], bins=30, kde=True, ax=ax)
        ax.set_title(f"{col} Distribution")
    st.pyplot(fig)

    # 4) Now load your results (predictions + residuals)
    df_result = pd.read_pickle("dataset/df_model_result_compressed.pkl.gz", compression="gzip")
    st.subheader("🔸 Result Sample (Model Outputs)")
    st.dataframe(df_result.head())
    
    
    # 3b) Position counts in the result data
    st.subheader("🔹 Raw Data: Players by Position Group")
    st.bar_chart(df_result["position_group"].value_counts())

    # 5) Predicted vs. Actual scatter (XGBoost)
    st.subheader("🔸 Predicted vs. Actual Potential (XGBoost)")
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(df_result["potential"], df_result["predicted_potential_XGBoost"], alpha=0.3)
    mn, mx = df_result["potential"].min(), df_result["potential"].max()
    ax.plot([mn,mx],[mn,mx],"r--", linewidth=2)
    ax.set_xlabel("Actual Potential")
    ax.set_ylabel("Predicted Potential")
    st.pyplot(fig)

    # 6) Residual distribution (XGBoost)
    st.subheader("🔸 Residuals Distribution (XGBoost)")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.histplot(df_result["residuals_XGB"], kde=True, ax=ax)
    ax.set_xlabel("Residual (Actual − Predicted)")
    ax.set_title("XGBoost Residuals")
    st.pyplot(fig)

    # 7) (Optional) compare mean & max errors by position
    agg = df_result.groupby("position_group").agg(
        mean_resid=("residuals_XGB","mean"),
        max_resid=("residuals_XGB", lambda x: x.abs().max())
    )
    st.subheader("🔸 XGBoost Residuals by Position Group")
    st.dataframe(agg)

    # 8) Custom, User-Driven Visualization (with grouping)
    # ——————————————————————————————
    st.markdown("### 📊 Build Your Own Plot")

    # 1) pick numeric columns
    numeric_cols = df_result.select_dtypes(include="number").columns.tolist()
    cols_to_plot = st.multiselect(
        "Select one or more numeric columns to visualize", 
        options=numeric_cols,
        default=["predicted_potential_XGBoost"]
    )
    
    # 2) pick chart type
    chart_type = st.selectbox(
        "Choose chart type", 
        ["Histogram", "Boxplot", "Scatter"]
    )
    
    # 3) pick a grouping column (hue)
    #    We allow "None" plus any object/category dtype in df_result
    cat_cols = df_result.select_dtypes(include=["object", "category"]).columns.tolist()
    group_by = st.selectbox(
        "Group by (add color categories)?", 
        options=["None"] + cat_cols
    )
    
    # 4) render
    if chart_type in ["Histogram", "Boxplot"]:
        for col in cols_to_plot:
            fig, ax = plt.subplots()
            if chart_type == "Histogram":
                if group_by != "None":
                    # grouped histogram
                    sns.histplot(
                        data=df_result,
                        x=col,
                        hue=group_by,
                        multiple="dodge",   # or "stack"/"layer"
                        bins=30,
                        kde=True,
                        ax=ax
                    )
                    ax.set_title(f"Histogram of {col} (grouped by {group_by})")
                else:
                    # single histogram
                    sns.histplot(
                        data=df_result,
                        x=col,
                        bins=30,
                        kde=True,
                        ax=ax
                    )
                    ax.set_title(f"Histogram of {col}")
            else:  # Boxplot
                if group_by != "None":
                    sns.boxplot(x=group_by, y=col, data=df_result, ax=ax)
                    ax.set_title(f"Boxplot of {col} by {group_by}")
                else:
                    sns.boxplot(y=df_result[col], ax=ax)
                    ax.set_title(f"Boxplot of {col}")
    
            st.pyplot(fig)
        
    elif chart_type == "Scatter":
        if len(cols_to_plot) >= 2:
            x_col = st.selectbox("X-axis", cols_to_plot, index=0)
            y_col = st.selectbox("Y-axis", cols_to_plot, index=1)
            fig, ax = plt.subplots()
    
            if group_by != "None":
                sns.scatterplot(
                    data=df_result,
                    x=x_col, y=y_col,
                    hue=group_by,
                    palette="tab10",
                    alpha=0.7,
                    ax=ax
                )
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            else:
                sns.scatterplot(
                    data=df_result,
                    x=x_col, y=y_col,
                    color="steelblue",
                    alpha=0.6,
                    ax=ax
                )
    
            ax.set_title(
                f"Scatter: {y_col} vs {x_col}"
                + (f" grouped by {group_by}" if group_by!="None" else "")
            )
            st.pyplot(fig)
        else:
            st.warning("Please select at least two columns for a scatter plot.")
    
# =============================
# 4. Model Evaluation Module
# =============================
elif module == "Model Evaluation":
    st.title("Model Evaluation & Comparisons")
    st.write("Here you can compare cross-validated performance across models and positions.")

    # --- 1. Pre-defined CV Results ---
    baseline = {
        "Attacker": {"Linear Regression": {'rmse': 2.65, 'mae': 2.06, 'r2': 0.826}},
        "Midfielder": {"Linear Regression": {'rmse': 2.60, 'mae': 2.02, 'r2': 0.831}},
        "Defender": {"Linear Regression": {'rmse': 2.42, 'mae': 1.86, 'r2': 0.837}},
        "Goalkeeper": {"Linear Regression": {'rmse': 2.61, 'mae': 2.01, 'r2': 0.837}},
    }
    nonlin = {
        "Attacker": {"LightGBM": {'rmse': 1.87,'mae': 1.17,'r2':0.913},
                     "SVR":      {'rmse': 2.03,'mae': 1.44,'r2':0.898},
                     "XGBoost":  {'rmse': 1.89,'mae': 1.18,'r2':0.912}},
        "Midfielder": {"LightGBM": {'rmse':1.75,'mae':1.12,'r2':0.923},
                       "SVR":      {'rmse':1.93,'mae':1.36,'r2':0.907},
                       "XGBoost":  {'rmse':1.76,'mae':1.14,'r2':0.922}},
        "Defender": {"LightGBM": {'rmse':1.66,'mae':1.06,'r2':0.923},
                     "SVR":      {'rmse':1.79,'mae':1.25,'r2':0.910},
                     "XGBoost":  {'rmse':1.67,'mae':1.08,'r2':0.922}},
        "Goalkeeper": {"LightGBM": {'rmse':1.71,'mae':1.12,'r2':0.930},
                       "SVR":      {'rmse':2.02,'mae':1.45,'r2':0.902},
                       "XGBoost":  {'rmse':1.73,'mae':1.12,'r2':0.928}},
    }

    # --- 2. Melt into a single DataFrame for easy plotting ---
    positions = ["Attacker", "Midfielder", "Defender", "Goalkeeper"]
    models    = ["LightGBM", "SVR", "XGBoost", "Linear Regression"]
    records = []
    for pos in positions:
        # non-linear first
        for m in ["LightGBM","SVR","XGBoost"]:
            rec = nonlin[pos][m].copy()
            rec.update({"Position":pos, "Model":m})
            records.append(rec)
        # baseline last
        rec = baseline[pos]["Linear Regression"].copy()
        rec.update({"Position":pos, "Model":"Linear Regression"})
        records.append(rec)

    df_eval = pd.DataFrame(records)

    st.subheader("Summary Table")
    st.dataframe(df_eval.set_index(["Position","Model"]))

    # --- 3. Bar-Charts for each metric ---
    fig, axes = plt.subplots(1,3, figsize=(18,5))
    sns.barplot(data=df_eval, x="Position", y="rmse", hue="Model", ax=axes[0])
    axes[0].set_title("RMSE by Position & Model")
    sns.barplot(data=df_eval, x="Position", y="mae", hue="Model", ax=axes[1])
    axes[1].set_title("MAE by Position & Model")
    sns.barplot(data=df_eval, x="Position", y="r2",  hue="Model", ax=axes[2])
    axes[2].set_title("R² by Position & Model")
    for ax in axes:
        ax.legend(loc="upper right")
    st.pyplot(fig)

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
        if attr == "age":
            # Age must be between 16 and 40
            user_input[attr] = st.number_input(
                f"Enter value for {attr}", 
                min_value=16, 
                max_value=40, 
                value=25  # sensible default in the middle
            )
        else:
            # All other attributes between 0 and 99
            user_input[attr] = st.number_input(
                f"Enter value for {attr}", 
                min_value=0, 
                max_value=99, 
                value=70
            )

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
st.sidebar.markdown("### Additional info")
st.sidebar.info("""
info
""")
