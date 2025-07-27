import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import base64

def render_image(filepath: str):
    """
    Renders an image in the Streamlit app.
    """
    mime_type = filepath.split('.')[-1:][0].lower()
    with open(filepath, "rb") as f:
        content_bytes = f.read()
        content_b64encoded = base64.b64encode(content_bytes).decode()
        image_string = f'data:image/{mime_type};base64,{content_b64encoded}'
        st.image(image_string)

# To this
render_image("static/ipl.jpg") # Use raw string for Windows paths

# Function to assign Power Play Phase
def assign_phase(over):
    if 0 <= over <= 5:
        return "1"
    elif 6 <= over <= 14:
        return "2"
    elif 15 <= over <= 19:
        return "3"
    else:
        return "Unknown Phase"

# Load Data
@st.cache_data
def load_data():
    delivery = pd.read_csv("deliveries.csv")
    matches = pd.read_csv("matches.csv")
    return delivery, matches

# Function to prepare final dataset for the model
def prepare_data(player_name, player_team):
    delivery, matches = load_data()

    # Merge delivery and matches data
    matches["season"] = matches["season"].replace({'2007/08': '2008', '2009/10': '2010', '2020/21': '2020'})
    ipl = pd.merge(delivery, matches, on='id')

    # Assign phases (Power Play)
    ipl['Power Play Number'] = ipl['over'].apply(assign_phase)

    # Filter data for the player
    ipl_player = ipl[ipl["batter"] == player_name].copy()

    # Initialize columns for runs
    runs_summary = ipl_player.groupby('id').agg({
        'batsman_runs': 'sum',
        'Power Play Number': 'first'
    }).reset_index()

    # Create columns for runs in Power Play phases
    runs_summary['runs_in_p1'] = runs_summary.apply(lambda x: x['batsman_runs'] if x['Power Play Number'] == '1' else 0, axis=1)
    runs_summary['runs_in_p2'] = runs_summary.apply(lambda x: x['batsman_runs'] if x['Power Play Number'] == '2' else 0, axis=1)
    runs_summary['runs_in_p3'] = runs_summary.apply(lambda x: x['batsman_runs'] if x['Power Play Number'] == '3' else 0, axis=1)

    # Sum runs in each phase
    phase_sums = runs_summary.groupby('id').agg({
        'runs_in_p1': 'sum',
        'runs_in_p2': 'sum',
        'runs_in_p3': 'sum'
    }).reset_index()

    # Grouping by match ID to get match-related info
    grouped = ipl_player.groupby('id').agg({
        'inning': 'first', 'target_runs': 'first', 'bowling_team': 'first',
        'toss_winner': 'first', 'season': 'first', 'city': 'first',
        'player_of_match': 'first', 'venue': 'first', 'winner': 'first',
        'result': 'first', 'result_margin': 'first'
    }).reset_index()

    # Merge the runs data with match info
    final = pd.merge(grouped, phase_sums, on='id', how='left')

    # Handle missing or invalid columns and convert categorical to numeric
    final['player_of_match_is_player'] = (final['player_of_match'] == player_name).astype(int)
    final['toss_winner_is_player_team'] = (final['toss_winner'] == player_team).astype(int)
    final['winner_is_player_team'] = (final['winner'] == player_team).astype(int)

    # Drop unnecessary or non-encodable columns like 'result'
    final.drop(columns=['result', 'player_of_match', 'toss_winner', 'winner'], inplace=True)

    # One-hot encoding for categorical columns
    final = pd.get_dummies(final, columns=['bowling_team', 'city', 'season', 'venue'], drop_first=True)

    return final

# Streamlit App
st.title("IPL Player Performance Predictor")

# Get user inputs
player_name = st.text_input("Enter Player Name (e.g., RG Sharma)", "RG Sharma")
player_team = st.text_input("Enter Player Team (e.g., Mumbai Indians)", "Mumbai Indians")

# Display data preparation
st.write(f"Player Name: {player_name}")
st.write(f"Player Team: {player_team}")

# Load data based on input
final = prepare_data(player_name, player_team)

if final is not None:  # Proceed only if final data is valid
    # Model training
    st.write("### Training the XGBoost model...")

    # Check if runs_in_p1, runs_in_p2, runs_in_p3 exist in final DataFrame
    if {'runs_in_p1', 'runs_in_p2', 'runs_in_p3'}.issubset(final.columns):
        # Define features and target
        X = final.drop(columns=['id', 'result_margin', 'runs_in_p1', 'runs_in_p2', 'runs_in_p3'], errors='ignore')
        y = final[['runs_in_p1', 'runs_in_p2', 'runs_in_p3']]

        # Splitting train and test sets based on season
        if 'season_2021' in final.columns:
            train = final[final['season_2021'] == 0]  # Train on all seasons except 2021
            test = final[final['season_2021'] == 1]   # Test on season 2021

            X_train = train.drop(columns=['runs_in_p1', 'runs_in_p2', 'runs_in_p3'], errors='ignore')
            y_train = train[['runs_in_p1', 'runs_in_p2', 'runs_in_p3']]
            X_test = test.drop(columns=['runs_in_p1', 'runs_in_p2', 'runs_in_p3'], errors='ignore')
            y_test = test[['runs_in_p1', 'runs_in_p2', 'runs_in_p3']]

            # Train model
            dtrain = xgb.DMatrix(X_train, label=y_train.values)
            dtest = xgb.DMatrix(X_test, label=y_test.values)
            params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse'}
            model = xgb.train(params, dtrain, num_boost_round=100)

            # Predict
            y_pred = model.predict(dtest)

            # Display RMSE
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            st.write(f"RMSE: {rmse:.4f}")

            # Show table with actual and predicted values
            st.write("### Actual vs Predicted Runs Table")
            results = pd.DataFrame({
                'Actual Runs in P1': y_test['runs_in_p1'].values,
                'Predicted Runs in P1': y_pred[:, 0],
                'Actual Runs in P2': y_test['runs_in_p2'].values,
                'Predicted Runs in P2': y_pred[:, 1],
                'Actual Runs in P3': y_test['runs_in_p3'].values,
                'Predicted Runs in P3': y_pred[:, 2]
            })
            st.write(results)

            # Plot actual vs predicted for runs in Power Play 1
            st.write("### Predicted vs Actual Runs (Power Play 1)")
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test['runs_in_p1'], y_pred[:, 0], alpha=0.7)
            plt.plot([y_test['runs_in_p1'].min(), y_test['runs_in_p1'].max()], [y_test['runs_in_p1'].min(), y_test['runs_in_p1'].max()], 'k--', lw=2)
            plt.title('Predicted vs Actual Runs (Power Play 1)')
            plt.xlabel('Actual Runs')
            plt.ylabel('Predicted Runs')
            plt.grid(True)
            st.pyplot(plt)

            # Plot actual vs predicted for runs in Power Play 2
            st.write("### Predicted vs Actual Runs (Power Play 2)")
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test['runs_in_p2'], y_pred[:, 1], alpha=0.7)
            plt.plot([y_test['runs_in_p2'].min(), y_test['runs_in_p2'].max()], [y_test['runs_in_p2'].min(), y_test['runs_in_p2'].max()], 'k--', lw=2)
            plt.title('Predicted vs Actual Runs (Power Play 2)')
            plt.xlabel('Actual Runs')
            plt.ylabel('Predicted Runs')
            plt.grid(True)
            st.pyplot(plt)

            # Plot actual vs predicted for runs in Power Play 3
            st.write("### Predicted vs Actual Runs (Power Play 3)")
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test['runs_in_p3'], y_pred[:, 2], alpha=0.7)
            plt.plot([y_test['runs_in_p3'].min(), y_test['runs_in_p3'].max()], [y_test['runs_in_p3'].min(), y_test['runs_in_p3'].max()], 'k--', lw=2)
            plt.title('Predicted vs Actual Runs (Power Play 3)')
            plt.xlabel('Actual Runs')
            plt.ylabel('Predicted Runs')
            plt.grid(True)
            st.pyplot(plt)

            # Creating the full table as a DataFrame
            full_table = pd.DataFrame({
                'Actual Runs in P1': y_test['runs_in_p1'].values,
                'Predicted Runs in P1': y_pred[:, 0],
                'Actual Runs in P2': y_test['runs_in_p2'].values,
                'Predicted Runs in P2': y_pred[:, 1],
                'Actual Runs in P3': y_test['runs_in_p3'].values,
                'Predicted Runs in P3': y_pred[:, 2]
            })

            # Display the full table
            st.write("### Full Table of Actual vs Predicted Runs for Powerplay Phases")
            st.dataframe(full_table)

            # Plot the full table as a bar graph
            def plot_full_table(data, title):
                ax = data.plot(kind='bar', figsize=(14, 8), alpha=0.75)
                ax.set_title(title)
                ax.set_ylabel("Runs")
                ax.set_xticks(np.arange(len(data)))
                ax.set_xticklabels(np.arange(1, len(data) + 1), rotation=0)
                plt.grid(True)
                st.pyplot(plt)

            # Display the full table bar graph
            plot_full_table(full_table, "Actual vs Predicted Runs (Powerplay 1, 2, and 3 Combined)")
        else:
            st.error("The 'season_2021' column is missing or misnamed from the final DataFrame.")
    else:
        st.error("One or more run columns are missing from the final DataFrame.")
