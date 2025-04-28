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

# Working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Define the selected features for each position (as used during training)
selected_features = {
    "Attacker": [
        'overall','age','skill_ball_control','skill_dribbling',
        'attacking_short_passing','attacking_finishing','mentality_positioning',
        'movement_reactions','power_shot_power','power_long_shots','mentality_vision'
    ],
    "Midfielder": [
        'overall','age','skill_ball_control','skill_dribbling',
        'attacking_short_passing','mentality_vision','movement_reactions',
        'skill_long_passing','mentality_composure','mentality_positioning','power_shot_power'
    ],
    "Defender": [
        'overall','age','defending_standing_tackle','defending_sliding_tackle',
        'mentality_interceptions','defending_marking_awareness','movement_reactions',
        'attacking_short_passing','skill_ball_control','mentality_composure','attacking_heading_accuracy'
    ],
    "Goalkeeper": [
        'overall','age','goalkeeping_reflexes','goalkeeping_diving',
        'goalkeeping_handling','goalkeeping_positioning','goalkeeping_kicking',
        'movement_reactions','mentality_composure','mentality_vision','goalkeeping_speed'
    ]
}

# Mapping of each position and each model to its saved pickle filename.
saved_models = {
    "Attacker":    {"Linear Regression": "Attacker_Linear_Regression_model.sav",
                      "LightGBM": "Attacker_LightGBM_model.sav",
                      "SVR": "Attacker_SVR_model.sav",
                      "XGBoost": "Attacker_XGBoost_model.sav"},
    "Midfielder":  {"Linear Regression": "Midfielder_Linear_Regression_model.sav",
                      "LightGBM": "Midfielder_LightGBM_model.sav",
                      "SVR": "Midfielder_SVR_model.sav",
                      "XGBoost": "Midfielder_XGBoost_model.sav"},
    "Defender":    {"Linear Regression": "Defender_Linear_Regression_model.sav",
                      "LightGBM": "Defender_LightGBM_model.sav",
                      "SVR": "Defender_SVR_model.sav",
                      "XGBoost": "Defender_XGBoost_model.sav"},
    "Goalkeeper":  {"Linear Regression": "Goalkeeper_Linear_Regression_model.sav",
                      "LightGBM": "Goalkeeper_LightGBM_model.sav",
                      "SVR": "Goalkeeper_SVR_model.sav",
                      "XGBoost": "Goalkeeper_XGBoost_model.sav"}
}

# Static lists for reuse
raw_num_cols = ["age", "overall", "potential"]
position_groups = ["Attacker", "Midfielder", "Defender", "Goalkeeper"]
plot_models = ["LightGBM", "SVR", "XGBoost", "Linear Regression"]

# ============================
# 2. Cached I/O functions
# ============================

@st.cache_data
def load_data(path: str, comp: str):
    return pd.read_pickle(path, compression=comp)

@st.cache_resource
def load_all_models():
    models = {}
    for pos, mfiles in saved_models.items():
        models[pos] = {}
        for mname, fname in mfiles.items():
            fpath = os.path.join(working_dir, 'saved_models', fname)
            with open(fpath, 'rb') as f:
                models[pos][mname] = pickle.load(f)
    return models

# Load data and models once
df_model  = load_data("dataset/male_players_compressed.pkl.bz2", "bz2")
df_result = load_data("dataset/df_model_result_compressed.pkl.gz", "gzip")
all_models = load_all_models()

# ============================
# 3. Streamlit Sidebar & Nav
# ============================
st.sidebar.title("Navigation")
module = st.sidebar.selectbox("Select Module", [
    "Overview & Data Summary",
    "Model Evaluation",
    "Interactive Prediction",
    "Build Your Own Plot",
    "About"
])
# ============================
# 4. Modules
# ============================
if module == "Overview & Data Summary":
    st.title("Player Potential Prediction App")
    st.header("Overview & Data Summary")

    st.subheader("🔹 Raw Data Sample")
    st.write("This dataframe below shows the first 5 rows of the raw dataset used to analysing, exploring, feature engineering, and building the prediction models.")
    st.dataframe(df_model.head())

    st.subheader("🔹 Raw Data Basic Statistics")
    st.write("The table below shows the basic statistics of the raw dataset.")
    st.write(df_model.describe())

    st.subheader("🔹 Distributions of Raw Numeric Features")
    st.write("The graphs below shows the distributions of age, overall, and potential features of the raw dataset.")
    fig, axes = plt.subplots(len(raw_num_cols), 1, figsize=(8, 4*len(raw_num_cols)))
    fig.subplots_adjust(hspace=0.6)
    for ax, col in zip(axes, raw_num_cols):
        sns.histplot(df_model[col], bins=30, kde=True, ax=ax)
        ax.set_title(f"{col} Distribution")
    st.pyplot(fig)

    st.subheader("🔸 Result Sample (Model Outputs)")
    st.write("This dataframe is the first 5 rows of the result dataset which is the final version.")
    st.dataframe(df_result.head())

    st.subheader("🔹 Result Data Basic Statistics")
    st.write("The table below shows the basic statistics of the result dataset.")
    st.write(df_result.describe())
    
    st.subheader("🔹 Result Data: Players by Position Group")
    st.write("This barchart shows the amount of players for each position group.")
    st.bar_chart(df_result["position_group"].value_counts())

    st.subheader("🔸 Predicted (LightGBM) vs. Actual Potential")
    st.write("This scatter plot describes the distribution of players' actual potential and predicted potential by LightGBM (The dotted red line is the perfect line).")
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(df_result["potential"], df_result["predicted_potential_LightGBM"], alpha=0.3)
    mn, mx = df_result["potential"].min(), df_result["potential"].max()
    ax.plot([mn,mx],[mn,mx],"r--", linewidth=2)
    ax.set_xlabel("Actual Potential")
    ax.set_ylabel("Predicted Potential")
    st.pyplot(fig)

    st.subheader("🔸 Residuals Distribution (LightGBM)")
    st.write("The difference distribution is shown down below between the actual potential and predicted potential (LightGBM).")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.histplot(df_result["residuals_LightGBM"], kde=True, ax=ax)
    ax.set_xlabel("Residual (Actual − Predicted)")
    ax.set_title("LightGBM Residuals")
    st.pyplot(fig)

    agg = df_result.groupby("position_group").agg(
        mean_resid=("residuals_LightGBM","mean"),
        max_resid=("residuals_LightGBM", lambda x: x.abs().max())
    )
    st.subheader("🔸 LightGBM Residuals by Position Group")
    st.write("For each position group, both the average and max residuals is shown in this table.")
    st.dataframe(agg)

# =============================
# 5. Model Evaluation Module
# =============================
elif module == "Model Evaluation":
    st.title("Model Evaluation & Comparisons")

    # --- 1. Pre-defined CV Results ---

    # Baseline (Linear Regression) CV results (precomputed)
    baseline_results = {
        "Attacker":    {"Linear Regression": {'rmse': 2.64, 'mae': 2.05, 'r2': 0.826}},
        "Midfielder":  {"Linear Regression": {'rmse': 2.59, 'mae': 2.02, 'r2': 0.831}},
        "Defender":    {"Linear Regression": {'rmse': 2.42, 'mae': 1.86, 'r2': 0.837}},
        "Goalkeeper":  {"Linear Regression": {'rmse': 2.57, 'mae': 1.99, 'r2': 0.841}}
    }
    
    # Non-linear models CV results with LightGBM, SVR, and XGBoost
    results_nonlin = {
        "Attacker": {
            "LightGBM": {'rmse': 1.86, 'mae': 1.17, 'r2': 0.914},
            "SVR":      {'rmse': 2.03, 'mae': 1.44, 'r2': 0.898},
            "XGBoost":  {'rmse': 1.89, 'mae': 1.19, 'r2': 0.911}
        },
        "Midfielder": {
            "LightGBM": {'rmse': 1.75, 'mae': 1.13, 'r2': 0.923},
            "SVR":      {'rmse': 1.92, 'mae': 1.36, 'r2': 0.907},
            "XGBoost":  {'rmse': 1.76, 'mae': 1.13, 'r2': 0.922}
        },
        "Defender": {
            "LightGBM": {'rmse': 1.66, 'mae': 1.07, 'r2': 0.923},
            "SVR":      {'rmse': 1.79, 'mae': 1.25, 'r2': 0.910},
            "XGBoost":  {'rmse': 1.67, 'mae': 1.08, 'r2': 0.922}
        },
        "Goalkeeper": {
            "LightGBM": {'rmse': 1.72, 'mae': 1.12, 'r2': 0.928},
            "SVR":      {'rmse': 2.02, 'mae': 1.46, 'r2': 0.902},
            "XGBoost":  {'rmse': 1.74, 'mae': 1.13, 'r2': 0.927}
        }
    }

    # --- 2. Melt into a single DataFrame for easy plotting ---
    positions = ["Attacker", "Midfielder", "Defender", "Goalkeeper"]
    models    = ["LightGBM", "SVR", "XGBoost", "Linear Regression"]
    records = []
    for pos in positions:
        # non-linear first
        for m in ["LightGBM","SVR","XGBoost"]:
            rec = results_nonlin[pos][m].copy()
            rec.update({"Position":pos, "Model":m})
            records.append(rec)
        # baseline last
        rec = baseline_results[pos]["Linear Regression"].copy()
        rec.update({"Position":pos, "Model":"Linear Regression"})
        records.append(rec)

    df_eval = pd.DataFrame(records)

    st.subheader("Summary Table")
    st.dataframe(df_eval.set_index(["Position","Model"]))

    st.subheader("Attacker Group", divider=True)
    st.write("""
    **Analysis**: **LightGBM** achieves the lowest RMSE (1.86) and MAE (1.17), and the highest R² (0.914), outperforming XGBoost (RMSE 1.89, MAE 1.19, R² 0.911), SVR, and Linear Regression.""")

    st.subheader("Midfielder Group", divider=True)
    st.write("""
    **Analysis**: **LightGBM** again leads with the lowest RMSE (1.75) and ties for the lowest MAE (1.13), while also posting the highest R² (0.923), just edging out XGBoost’s RMSE of 1.76 and R² of 0.922.""")

    st.subheader("Defender Group", divider=True)
    st.write("""
    **Analysis**: **LightGBM** posts the top metrics (RMSE 1.66, MAE 1.07, R² 0.923), narrowly beating XGBoost (RMSE 1.67, MAE 1.08, R² 0.922) and clearly outperforming SVR and Linear Regression.""")

    st.subheader("Goalkeeper Group", divider=True)
    st.write("""
    **Analysis**: With the lowest RMSE (1.72), lowest MAE (1.12), and highest R² (0.928), **LightGBM** slightly outperforms XGBoost (RMSE 1.74, MAE 1.13, R² 0.927).""")

    
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


    st.subheader("Overall Conclusion", divider=True)
    st.write("""
    Based on the comparison:

    **LightGBM** is the best performing model for all position groups, as it consistently delivers the best balance of accuracy (lowest RMSE/MAE) and explanatory power (highest R²).
    
    **XGBoost** is very close, but in every case, **LightGBM** edges out slightly.
    
    **SVR** performs better than the baseline **Linear Regression** but is outperformed by the tree-based ensemble methods.
    
    Thus, for this dataset and the given evaluation, **LightGBM** appears to be the best model across *Attacker*, *Midfielder*, *Defender*, and *Goalkeeper* groups.""")


# =======================
# 6. Build your own Plot
# =======================

elif module == "Build Your Own Plot":
    st.title("📊 Build Your Own Plot")
    st.write("Use the form below to select your variables, then hit **Draw Plot**.")

    # 1) Build the form
    with st.form("custom_plot_form"):
        numeric_cols = df_result.select_dtypes(include="number").columns.tolist()
        cols_to_plot = st.multiselect(
            "Select numeric columns to visualize",
            options=numeric_cols,
            default=[numeric_cols[0]] if numeric_cols else []
        )
        chart_type = st.selectbox(
            "Select chart type",
            ["Histogram", "Boxplot", "Scatter"]
        )
        cat_cols = df_result[["position_group", "preferred_foot"]].columns.tolist()
        group_by = st.selectbox(
            "Group by (optional)",
            options=["None"] + cat_cols
        )

        draw = st.form_submit_button("Draw Plot")

    # 2) Only render when user clicks
    if draw:
        if chart_type in ["Histogram", "Boxplot"]:
            for col in cols_to_plot:
                fig, ax = plt.subplots()
                if chart_type == "Histogram":
                    if group_by != "None":
                        sns.histplot(data=df_result, x=col, hue=group_by,
                                     multiple="dodge", bins=30, kde=True, ax=ax)
                        ax.set_title(f"Histogram of {col} (grouped by {group_by})")
                    else:
                        sns.histplot(data=df_result, x=col, bins=30, kde=True, ax=ax)
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
            if len(cols_to_plot) < 2:
                st.warning("Please select at least two columns for a scatter plot.")
            else:
                x_col, y_col = cols_to_plot[:2]
                fig, ax = plt.subplots()
                if group_by != "None":
                    sns.scatterplot(data=df_result, x=x_col, y=y_col,
                                    hue=group_by, alpha=0.7, ax=ax)
                    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                else:
                    sns.scatterplot(data=df_result, x=x_col, y=y_col,
                                    alpha=0.6, ax=ax)
                ax.set_title(f"Scatter: {y_col} vs {x_col}"
                             + (f" grouped by {group_by}" if group_by!="None" else ""))
                st.pyplot(fig)


# ============================
# 7. Interactive Prediction Module
# ============================
elif module == "Interactive Prediction":
    st.title("Interactive Prediction Module")
    position = st.selectbox("Select Position", position_groups)
    model_choice = st.selectbox("Select Model", plot_models)

    st.subheader("To understand the value of the input, please see the **"Result Data Basic Statistics"** table in the **"Overview & Data Summary"** module.")
    
    st.subheader(f"Input values for the {position} attributes:")
    inputs = {}
    for attr in selected_features[position]:
        default = 20 if attr == 'age' else 75
        minv, maxv = (16,40) if attr=='age' else (0,99)
        inputs[attr] = st.number_input(f"{attr}", minv, maxv, value=default)

    if st.button("Predict Potential"):
        model = all_models[position][model_choice]
        X_new = pd.DataFrame([inputs], columns=selected_features[position])
        pred = model.predict(X_new)[0]
        st.success(f"Predicted Potential: {pred:.2f}")
        
# ============================
# 8. About Module
# ============================
elif module == "About":
    st.title("About the Player Potential Prediction App")
    st.write("""
    This app predicts football players’ future potential using four machine learning models:
    
    - **Linear Regression**  
    - **LightGBM**  
    - **SVR**  
    - **XGBoost**  
    
    It is organized into five modules:
    1. **Overview & Data Summary**  
       Explore the raw dataset and basic summary statistics.  
    2. **Model Evaluation**  
       Compare CV metrics (RMSE, MAE, R²) across models and positions.  
    3. **Interactive Prediction**  
       Enter a player’s 11 attributes and get a potential score.  
    4. **Build Your Own Plot**  
       Customize charts (Histogram, Boxplot, Scatter) on any numeric column.  
    5. **About**  
       You’re here!
    """)

    st.subheader("📋 Guidelines")
    st.markdown("""
    - **Navigation:** Use the sidebar to switch modules.  
    - **Input Ranges (Interactive Prediction):**  
      - **Age:** 16–40  
      - **Other attributes:** 0–99  
    - **Model Selection:**  
      - Tree-based models (LightGBM, XGBoost) generally outperform SVR and Linear Regression.  
      - Use **Overview** or **Model Evaluation** to see which model suits your needs.  
    - **Build Your Own Plot:**  
      1. Select one or more numeric columns.  
      2. Choose a chart type.  
      3. (Optional) Group by position groups or preffered foot.  
      4. Click **Draw Plot** to render—this prevents reruns on every widget change.  
    - **Performance Tips:**  
      - Data loading is cached—switching modules won’t re-read the dataset.  
      - Use **Build Your Own Plot** for quick, custom visualizations without re-running the whole summary.  
    """)
    st.subheader("🔍 Feature Guide")
    st.markdown("""
    | Feature                               | Description                                                         |
    |---------------------------------------|---------------------------------------------------------------------|
    | **short_name**                        | Player’s commonly used name                                        |
    | **long_name**                         | Player’s full official name                                        |
    | **player_positions**                  | List of positions the player can play                              |
    | **overall**                           | Current overall rating (0–99)                                      |
    | **potential**                         | Projected future potential rating (0–99)                           |
    | **age**                               | Player’s age in years                                              |
    | **height_cm**                         | Player’s height in centimeters                                     |
    | **weight_kg**                         | Player’s weight in kilograms                                       |
    | **nationality_name**                  | Country of the player’s nationality                                |
    | **preferred_foot**                    | Preferred kicking foot (Left or Right)                             |
    | **weak_foot**                         | Ability with the non-preferred foot (1–5)                          |
    | **skill_moves**                       | Skill moves proficiency (1–5)                                      |
    | **international_reputation**          | International reputation level (1–5)                               |
    | **work_rate**                         | Attacking/defensive work rate (e.g., High/Medium)                  |
    | **attacking_crossing**                | Ability to deliver accurate crosses                                |
    | **attacking_finishing**               | Skill in finishing scoring opportunities                           |
    | **attacking_heading_accuracy**        | Accuracy when heading the ball                                     |
    | **attacking_short_passing**           | Accuracy of short passes                                           |
    | **attacking_volleys**                 | Ability to strike volleys accurately                               |
    | **skill_dribbling**                   | Ability to dribble past opponents                                  |
    | **skill_curve**                       | Ability to curl the ball on shots or passes                        |
    | **skill_fk_accuracy**                 | Accuracy of free kicks                                             |
    | **skill_long_passing**                | Accuracy of long-range passes                                      |
    | **skill_ball_control**                | Control of the ball under pressure                                 |
    | **movement_acceleration**             | Speed at which player reaches top speed                            |
    | **movement_sprint_speed**             | Top sprinting speed                                                |
    | **movement_agility**                  | Ability to change direction quickly                                |
    | **movement_reactions**                | Speed of reacting to loose balls                                   |
    | **movement_balance**                  | Stability under physical challenges                                |
    | **power_shot_power**                  | Power behind shots                                                 |
    | **power_jumping**                     | Jumping ability and aerial reach                                   |
    | **power_stamina**                     | Endurance over the duration of a match                             |
    | **power_strength**                    | Physical strength in challenges                                    |
    | **power_long_shots**                  | Accuracy and power of long-distance shots                          |
    | **mentality_aggression**              | Aggressiveness in duels and challenges                             |
    | **mentality_interceptions**           | Ability to anticipate and intercept passes                         |
    | **mentality_positioning**             | Positional sense off the ball                                      |
    | **mentality_vision**                  | Ability to see and execute key passes                              |
    | **mentality_penalties**               | Penalty-taking skill                                              |
    | **mentality_composure**               | Calmness under pressure                                            |
    | **defending_marking_awareness**       | Awareness when marking opponents                                   |
    | **defending_standing_tackle**         | Ability to perform standing tackles                                |
    | **defending_sliding_tackle**          | Ability to perform sliding tackles                                 |
    | **goalkeeping_diving**                | Agility and technique in diving saves                              |
    | **goalkeeping_handling**              | Ability to catch and hold the ball securely                         |
    | **goalkeeping_kicking**               | Distance and accuracy of goal kicks                                 |
    | **goalkeeping_positioning**           | Positioning and anticipation in goal                               |
    | **goalkeeping_reflexes**              | Reaction speed to shots near goal                                   |
    | **goalkeeping_speed**                 | Speed when coming off the line or covering ground                  |
    | **position_group**                    | Position category used for modeling                                |
    | **predicted_potential_Linear_Regression** | Potential predicted by Linear Regression                       |
    | **predicted_potential_SVR**           | Potential predicted by SVR                                         |
    | **predicted_potential_XGBoost**       | Potential predicted by XGBoost                                     |
    | **predicted_potential_LightGBM**      | Potential predicted by LightGBM                                    |
    | **residuals_LightGBM**                | Difference between actual and LightGBM prediction                  |
    """)

    st.write("**Contact:** emrecam13@gmail.com")

# ============================
# Additional info
# ============================
st.sidebar.info("Developed by **Emre Çam** for the MSc Data Science & AI dissertation at Bournemouth University.")
