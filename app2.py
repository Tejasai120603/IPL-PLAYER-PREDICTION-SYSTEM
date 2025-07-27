import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import base64
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#Function to analyze player performance
def analyze_player_performance(player_name):
    #Filter data for the specific player as a batter
    player_batter_data = data[data['batter'] == player_name]
    
    #Check if there is enough data for the player as a batter
    if player_batter_data.empty:
        st.write(f"No data available for player: {player_name}")
        return

    #Calculate runs scored by the player against different teams
    runs_against_teams = player_batter_data.groupby('bowling_team')['batsman_runs'].sum()
    
    #Calculate runs scored by the player in different years
    player_batter_data['year'] = pd.to_datetime(player_batter_data['date']).dt.year
    runs_by_year = player_batter_data.groupby('year')['batsman_runs'].sum()
    
    #Plot bar graphs for runs against different teams and in different years
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    #Bar graph for runs against different teams
    ax[0].bar(runs_against_teams.index, runs_against_teams, color='skyblue')
    ax[0].set_title(f'Runs scored by {player_name} against different teams')
    ax[0].set_ylabel('Runs')
    ax[0].set_xticklabels(runs_against_teams.index, rotation=45, ha='right')
    
    #Bar graph for runs in different years
    ax[1].bar(runs_by_year.index, runs_by_year, color='lightgreen')
    ax[1].set_title(f'Runs scored by {player_name} in different years')
    ax[1].set_ylabel('Runs')
    ax[1].set_xticks(runs_by_year.index)
    ax[1].set_xticklabels(runs_by_year.index, rotation=45, ha='right')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Encode categorical columns (teams and players) based on the entire dataset
    label_encoder_batter = LabelEncoder()
    label_encoder_team = LabelEncoder()

    data['batter_encoded'] = label_encoder_batter.fit_transform(data['batter'])
    data['bowling_team_encoded'] = label_encoder_team.fit_transform(data['bowling_team'])

    # Select features and target variable
    features = ['batter_encoded', 'bowling_team_encoded', 'over', 'ball']
    target = 'batsman_runs'

    # Split data into training and testing sets
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the XGBoost model
    xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, seed=42)
    xg_reg.fit(X_train, y_train)

    # Check if the player's name was in the LabelEncoder's known classes
    if player_name not in label_encoder_batter.classes_:
        st.write(f"Player {player_name} not seen in training data. Unable to make predictions.")
        return
    
    # Prepare data for predictions
    player_encoded = label_encoder_batter.transform([player_name])[0]
    player_team_data = data[data['batter_encoded'] == player_encoded][features]
    
    if player_team_data.empty:
        st.write(f"No data available for player: {player_name} to make predictions.")
        return
    
    # Make predictions
    player_predictions = xg_reg.predict(player_team_data)
    
    # Calculate and display mean squared error for the model
    mse = mean_squared_error(y_test, xg_reg.predict(X_test))
    st.write(f'Mean Squared Error: {mse}')
    
    # Display the first 5 predictions
    st.write(f'Predicted runs for {player_name} in first 5 instances: {player_predictions[:5]}')

    # Summary statistics for batter
    total_runs = player_batter_data['batsman_runs'].sum()
    matches_played = player_batter_data['match_id'].nunique()
    
    # Unique matches where the player was awarded "Player of the Match"
    player_of_match_awards = player_batter_data[player_batter_data['player_of_match'] == player_name]
    unique_matches_with_award = player_of_match_awards.groupby('match_id').size().count()
    
    st.write(f"\nSummary for {player_name}:")
    st.write(f"Total Runs: {total_runs}")
    st.write(f"Matches Played: {matches_played}")
    st.write(f"Player of the Match Awards: {unique_matches_with_award}")

    # Check if the player is also a bowler
    player_bowler_data = data[data['bowler'] == player_name]
    
    if not player_bowler_data.empty:
        st.write(f"{player_name} has bowling data.")
    else:
        st.write(f"{player_name} has no bowling data.")

# Load the data
file_path = 'C:\MSXL\mlfinalproject\merged_deliveries_matches_NEW.csv'
data = pd.read_csv(file_path, low_memory=False)



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

# Load your image here
render_image(r"C:\Users\suman\Downloads\personal\projects\MSXL\mlfinalproject\static\ipl.jpg")  # Use raw string for Windows paths

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

            # # Show table with actual and predicted values
            # st.write("### Actual vs Predicted Runs Table")
            # results = pd.DataFrame({
            #     'Actual Runs in P1': y_test['runs_in_p1'].values,
            #     'Predicted Runs in P1': y_pred[:, 0],
            #     'Actual Runs in P2': y_test['runs_in_p2'].values,
            #     'Predicted Runs in P2': y_pred[:, 1],
            #     'Actual Runs in P3': y_test['runs_in_p3'].values,
            #     'Predicted Runs in P3': y_pred[:, 2]
            # })
            # st.write(results)

            # # Plot actual vs predicted for runs in Power Play 1
            # st.write("### Predicted vs Actual Runs (Power Play 1)")
            # plt.figure(figsize=(10, 6))
            # plt.scatter(y_test['runs_in_p1'], y_pred[:, 0], alpha=0.7)
            # plt.plot([y_test['runs_in_p1'].min(), y_test['runs_in_p1'].max()], [y_test['runs_in_p1'].min(), y_test['runs_in_p1'].max()], 'k--', lw=2)
            # plt.title('Predicted vs Actual Runs (Power Play 1)')
            # plt.xlabel('Actual Runs')
            # plt.ylabel('Predicted Runs')
            # plt.grid(True)
            # st.pyplot(plt)

            # # Plot actual vs predicted for runs in Power Play 2
            # st.write("### Predicted vs Actual Runs (Power Play 2)")
            # plt.figure(figsize=(10, 6))
            # plt.scatter(y_test['runs_in_p2'], y_pred[:, 1], alpha=0.7)
            # plt.plot([y_test['runs_in_p2'].min(), y_test['runs_in_p2'].max()], [y_test['runs_in_p2'].min(), y_test['runs_in_p2'].max()], 'k--', lw=2)
            # plt.title('Predicted vs Actual Runs (Power Play 2)')
            # plt.xlabel('Actual Runs')
            # plt.ylabel('Predicted Runs')
            # plt.grid(True)
            # st.pyplot(plt)

            # # Plot actual vs predicted for runs in Power Play 3
            # st.write("### Predicted vs Actual Runs (Power Play 3)")
            # plt.figure(figsize=(10, 6))
            # plt.scatter(y_test['runs_in_p3'], y_pred[:, 2], alpha=0.7)
            # plt.plot([y_test['runs_in_p3'].min(), y_test['runs_in_p3'].max()], [y_test['runs_in_p3'].min(), y_test['runs_in_p3'].max()], 'k--', lw=2)
            # plt.title('Predicted vs Actual Runs (Power Play 3)')
            # plt.xlabel('Actual Runs')
            # plt.ylabel('Predicted Runs')
            # plt.grid(True)
            # st.pyplot(plt)

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



# Streamlit app to analyze performance
st.title("Analyze Player Performance")

# Input for player name
player_name = st.text_input("Enter player name:", "RG Sharma")
# Button to analyze performance
if st.button("Analyze Performance"):
    analyze_player_performance(player_name)

