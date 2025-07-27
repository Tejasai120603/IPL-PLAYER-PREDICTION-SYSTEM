import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import base64
from pathlib import Path

# --- Configuration and Helper Functions ---

# Get the directory of the current script to build robust file paths
SCRIPT_DIR = Path(__file__).parent
STATIC_PATH = SCRIPT_DIR / "static"
DATA_PATH = SCRIPT_DIR

def render_image(filepath: Path):
    """Renders an image from a file path."""
    try:
        with open(filepath, "rb") as f:
            content_bytes = f.read()
        content_b64encoded = base64.b64encode(content_bytes).decode()
        mime_type = filepath.suffix.replace('.', '')
        image_string = f'data:image/{mime_type};base64,{content_b64encoded}'
        st.image(image_string)
    except FileNotFoundError:
        st.error(f"Error: Image file not found at {filepath}")

def assign_phase(over):
    """Assigns a match phase based on the over number."""
    if 0 <= over <= 5:
        return "Phase 1 (Overs 1-6)"
    elif 6 <= over <= 14:
        return "Phase 2 (Overs 7-15)"
    else: # 15 to 19
        return "Phase 3 (Overs 16-20)"

@st.cache_data
def load_data():
    """Loads and caches the IPL datasets."""
    try:
        delivery_path = DATA_PATH / "deliveries.csv"
        matches_path = DATA_PATH / "matches.csv"
        delivery = pd.read_csv(delivery_path)
        matches = pd.read_csv(matches_path)
        return delivery, matches
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}. Make sure 'deliveries.csv' and 'matches.csv' are in the same directory as the app.")
        return None, None

def prepare_data(player_name, player_team):
    """Prepares the final dataset for a specific player and team."""
    delivery, matches = load_data()
    if delivery is None or matches is None:
        return None

    # --- Data Cleaning and Merging ---
    matches["season"] = matches["season"].astype(str).replace({'2007/08': '2008', '2009/10': '2010', '2020/21': '2020'})
    ipl = pd.merge(delivery, matches, on='id', how='inner')

    # Filter for the selected player
    ipl_player = ipl[ipl["batter"] == player_name].copy()

    if ipl_player.empty:
        st.error(f"No data found for player: '{player_name}'. Please check the spelling.")
        return None

    # --- Feature Engineering ---
    ipl_player['phase'] = ipl_player['over'].apply(assign_phase)

    # Correctly calculate runs per phase for each delivery
    for phase in ["Phase 1 (Overs 1-6)", "Phase 2 (Overs 7-15)", "Phase 3 (Overs 16-20)"]:
        col_name = f"runs_in_{phase.split(' ')[1]}"
        ipl_player[col_name] = np.where(ipl_player['phase'] == phase, ipl_player['batsman_runs'], 0)

    # Aggregate data by match
    phase_runs = ipl_player.groupby('id').agg({
        'runs_in_p1': 'sum',
        'runs_in_p2': 'sum',
        'runs_in_p3': 'sum'
    }).reset_index()

    match_info = ipl_player.groupby('id').agg({
        'inning': 'first', 'bowling_team': 'first', 'toss_winner': 'first',
        'season': 'first', 'city': 'first', 'venue': 'first', 'winner': 'first'
    }).reset_index()

    final = pd.merge(match_info, phase_runs, on='id', how='left')

    # Create binary features
    final['toss_winner_is_player_team'] = (final['toss_winner'] == player_team).astype(int)
    final['winner_is_player_team'] = (final['winner'] == player_team).astype(int)

    # One-hot encode categorical features
    final = pd.get_dummies(final, columns=['bowling_team', 'city', 'season', 'venue'], drop_first=True)
    
    # Drop original columns that have been encoded or are not needed
    final.drop(columns=['toss_winner', 'winner'], inplace=True)

    return final

# --- Streamlit App UI ---

st.set_page_config(layout="wide")
st.title("🏏 IPL Player Performance Predictor")

render_image(STATIC_PATH / "ipl.jpg")

st.sidebar.header("Player Details")
player_name = st.sidebar.text_input("Enter Player Name (e.g., V Kohli)", "V Kohli")
player_team = st.sidebar.text_input("Enter Player Team (e.g., Royal Challengers Bangalore)", "Royal Challengers Bangalore")

if st.sidebar.button("Predict Performance"):
    final_df = prepare_data(player_name, player_team)

    if final_df is not None:
        st.header(f"Prediction for {player_name}")

        # Define features (X) and target (y)
        # Ensure 'id' and target columns are not in features
        feature_cols = [col for col in final_df.columns if col not in ['id', 'runs_in_p1', 'runs_in_p2', 'runs_in_p3']]
        X = final_df[feature_cols]
        y = final_df[['runs_in_p1', 'runs_in_p2', 'runs_in_p3']]

        # --- Train/Test Split ---
        # Hardcoded split on season 2021 for demonstration
        test_season_col = 'season_2021'
        if test_season_col not in X.columns:
            st.error(f"No data available for the test season (2021) for {player_name}.")
        else:
            train_idx = X[test_season_col] == 0
            test_idx = X[test_season_col] == 1

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Drop the season column used for splitting from features
            X_train = X_train.drop(columns=[col for col in X_train.columns if 'season_' in col])
            X_test = X_test.drop(columns=[col for col in X_test.columns if 'season_' in col])


            if X_test.empty:
                st.warning(f"{player_name} did not play in the test season (2021). No prediction can be made.")
            else:
                # --- Model Training and Prediction ---
                st.subheader("Model Performance")
                
                # Align columns after one-hot encoding
                train_cols = X_train.columns
                test_cols = X_test.columns
                missing_in_test = set(train_cols) - set(test_cols)
                for c in missing_in_test:
                    X_test[c] = 0
                missing_in_train = set(test_cols) - set(train_cols)
                for c in missing_in_train:
                    X_train[c] = 0
                X_test = X_test[train_cols]


                dtrain = xgb.DMatrix(X_train, label=y_train)
                dtest = xgb.DMatrix(X_test, label=y_test)
                
                params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse'}
                model = xgb.train(params, dtrain, num_boost_round=100)
                y_pred = model.predict(dtest)

                rmse = mean_squared_error(y_test, y_pred, squared=False)
                st.metric(label="Model RMSE (Root Mean Squared Error)", value=f"{rmse:.2f} runs")

                # --- Display Results ---
                st.subheader("Actual vs. Predicted Runs (2021 Season)")
                results = pd.DataFrame({
                    'Actual P1': y_test['runs_in_p1'].values,
                    'Predicted P1': y_pred[:, 0].round(1),
                    'Actual P2': y_test['runs_in_p2'].values,
                    'Predicted P2': y_pred[:, 1].round(1),
                    'Actual P3': y_test['runs_in_p3'].values,
                    'Predicted P3': y_pred[:, 2].round(1)
                })
                st.dataframe(results)

                # --- Visualizations ---
                st.subheader("Performance Visualizations")
                fig, axes = plt.subplots(1, 3, figsize=(20, 6))
                plt.style.use('seaborn-v0_8-whitegrid')

                # Plot for Phase 1
                axes[0].scatter(y_test['runs_in_p1'], y_pred[:, 0], alpha=0.7, edgecolors='k')
                axes[0].plot([0, y_test['runs_in_p1'].max()], [0, y_test['runs_in_p1'].max()], 'r--', lw=2)
                axes[0].set_title('Phase 1 (Overs 1-6)')
                axes[0].set_xlabel('Actual Runs')
                axes[0].set_ylabel('Predicted Runs')

                # Plot for Phase 2
                axes[1].scatter(y_test['runs_in_p2'], y_pred[:, 1], alpha=0.7, edgecolors='k')
                axes[1].plot([0, y_test['runs_in_p2'].max()], [0, y_test['runs_in_p2'].max()], 'r--', lw=2)
                axes[1].set_title('Phase 2 (Overs 7-15)')
                axes[1].set_xlabel('Actual Runs')

                # Plot for Phase 3
                axes[2].scatter(y_test['runs_in_p3'], y_pred[:, 2], alpha=0.7, edgecolors='k')
                axes[2].plot([0, y_test['runs_in_p3'].max()], [0, y_test['runs_in_p3'].max()], 'r--', lw=2)
                axes[2].set_title('Phase 3 (Overs 16-20)')
                axes[2].set_xlabel('Actual Runs')
                
                fig.suptitle(f'Actual vs. Predicted Runs for {player_name}', fontsize=16)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                st.pyplot(fig)