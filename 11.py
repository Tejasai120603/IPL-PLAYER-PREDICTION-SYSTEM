# %%
import pandas as pd
import numpy as np

# %%
delivery = pd.read_csv("deliveries.csv")
matches = pd.read_csv("matches.csv")

# %%
delivery.tail()

# %%
matches["season"] = matches["season"].replace({
    '2007/08': '2008',
    '2009/10': '2010',
    '2020/21': '2020'
})

# %%
ipl = pd.merge(delivery, matches, on='id')

# %%
ipl.info()

# %%
ipl.tail()

# %%
# Define a function to categorize the phases
def assign_phase(over):
    if 0 <= over <= 5:
        return "1"
    elif 6 <= over <= 14:
        return "2"
    elif 15 <= over <= 19:
        return "3"
    else:
        return "Unknown Phase"

# Apply the function to create the 'phase' column
ipl['Power Play Number'] = ipl['over'].apply(assign_phase)

# Display the first few rows to check the result
print(ipl[['over', 'Power Play Number']].head())

# %%
player_name = "RG Sharma"
player_team = "Mumbai Indians"

# %%
ipl_player = ipl[ipl["batter"]==player_name].copy()
ipl_player.tail()

# %%
# # Define a function to categorize the phases
# def assign_phase(over):
#     if 0 <= over <= 5:
#         return "1"
#     elif 6 <= over <= 14:
#         return "2"
#     elif 15 <= over <= 19:
#         return "3"
#     else:
#         return "Unknown Phase"

# # Apply the function to create the 'phase' column
# ipl['Power Play Number'] = ipl['over'].apply(assign_phase)

# # Display the first few rows to check the result
# print(ipl[['over', 'Power Play Number']].head())

# %%
# ipl.tail()

# %%


# %%
ipl_player.tail()

# %%
ipl_player.columns

# %%
player_runs = ipl_player[['id','batsman_runs','Power Play Number']]
player_runs[player_runs["id"]==335982]

# %%
# Initialize new columns for storing the sum of runs in each Power Play
player_runs['runs_in_p1'] = 0
player_runs['runs_in_p2'] = 0
player_runs['runs_in_p3'] = 0

# Get the unique match ids
unique_ids = player_runs['id'].unique()



# Iterate over each match id
for match_id in unique_ids:
    # Calculate the sum of runs in each Power Play for the current match id
    runs_in_p1 = player_runs[(player_runs["id"] == match_id) & (player_runs["Power Play Number"] == '1')]["batsman_runs"].sum()
    runs_in_p2 = player_runs[(player_runs["id"] == match_id) & (player_runs["Power Play Number"] == '2')]["batsman_runs"].sum()
    runs_in_p3 = player_runs[(player_runs["id"] == match_id) & (player_runs["Power Play Number"] == '3')]["batsman_runs"].sum()

    # Assign these values to the respective rows in the DataFrame
    player_runs.loc[player_runs["id"] == match_id, 'runs_in_p1'] = runs_in_p1
    player_runs.loc[player_runs["id"] == match_id, 'runs_in_p2'] = runs_in_p2
    player_runs.loc[player_runs["id"] == match_id, 'runs_in_p3'] = runs_in_p3

# Display the DataFrame to verify
print(player_runs.head())


# %%
grouped = ipl_player.groupby('id').agg({
    'inning': 'first',                # First inning value (assuming it doesn't change for a match ID)
    'target_runs': 'first',           # Target runs (assuming one value per match)
    'bowling_team': 'first',          # First bowling team value
    'toss_winner': 'first',           # First toss winner value
    'season': 'first',                # First season value
    'city': 'first',                  # First city value
    'player_of_match': 'first',       # First player of match value
    'venue': 'first',                 # First venue value
    'winner': 'first',                # First winner value
    'result': 'first',                # First result value
    'result_margin': 'first'          # First result margin value
})

# Now create or update the final DataFrame with the grouped data
final = grouped.reset_index()

# Display the final DataFrame
print(final.tail())

# %%
# final = kohli_runs[["id"]].copy()

# %%
# final[["inning","target_runs","bowling_team","toss_winner",'season', 'city','player_of_match', 'venue','winner', 'result', 'result_margin']] = ipl_kohli[["inning","target_runs","bowling_team","toss_winner",'season', 'city','player_of_match', 'venue','winner', 'result', 'result_margin']]

# %%
# final.head()

# %%
player_runs[player_runs["id"]==1426310]

# %%
unique_ids = player_runs['id'].unique()

# Initialize the 'runs_in_p1', 'runs_in_p2', 'runs_in_p3' columns in 'final' DataFrame
final['runs_in_p1'] = 0
final['runs_in_p2'] = 0
final['runs_in_p3'] = 0

# Iterate over each unique match ID
for match_id in unique_ids:
    # Extract the corresponding runs for each Power Play from kohli_runs
    runs_in_p1 = player_runs.loc[player_runs["id"] == match_id, "runs_in_p1"].iloc[0]
    runs_in_p2 = player_runs.loc[player_runs["id"] == match_id, "runs_in_p2"].iloc[0]
    runs_in_p3 = player_runs.loc[player_runs["id"] == match_id, "runs_in_p3"].iloc[0]
    
    # Assign these values to the corresponding rows in the final DataFrame
    final.loc[final["id"] == match_id, 'runs_in_p1'] = runs_in_p1
    final.loc[final["id"] == match_id, 'runs_in_p2'] = runs_in_p2
    final.loc[final["id"] == match_id, 'runs_in_p3'] = runs_in_p3

# Display the updated 'final' DataFrame
print(final.head())

# %%
print(final.tail())


# %%
final.columns

# %%
final.head()

# %%
from sklearn.preprocessing import OneHotEncoder

# Assuming you have a DataFrame called final

# 1. One-hot encoding for 'bowling_team' using sklearn's OneHotEncoder
encoder_team = OneHotEncoder(drop='first', sparse_output=False)
encoded_teams = encoder_team.fit_transform(final[['bowling_team']])

# Create DataFrame for encoded teams
team_df = pd.DataFrame(encoded_teams, columns=encoder_team.get_feature_names_out(['bowling_team']))

# 2. One-hot encoding for 'city' using sklearn's OneHotEncoder
encoder_city = OneHotEncoder(drop='first', sparse_output=False)
encoded_city = encoder_city.fit_transform(final[['city']])

# Create DataFrame for encoded city
city_df = pd.DataFrame(encoded_city, columns=encoder_city.get_feature_names_out(['city']))

# 3. Concatenate the encoded columns back to the original DataFrame
final = pd.concat([final, team_df, city_df], axis=1)

# 4. Drop the original 'bowling_team' and 'city' columns
final.drop(columns=['bowling_team', 'city'], inplace=True)

# 5. Binary column for 'toss_winner' if it matches 'player_team'
final['toss_winner_is_player_team'] = (final['toss_winner'] == player_team).astype(int)

# Create a mapping dictionary for seasons starting from 2008 to 2024
season_mapping = {str(year): idx + 1 for idx, year in enumerate(range(2008, 2025))}

# Apply the mapping to the 'season' column
final['season_mapped'] = final['season'].map(season_mapping)


# 7. Binary column for 'player_of_match' if it matches 'player_name'
final['player_of_match_is_player'] = (final['player_of_match'] == player_name).astype(int)

# 8. Binary column for 'winner' if it matches 'player_team'
final['winner_is_player_team'] = (final['winner'] == player_team).astype(int)

# 9. Drop the 'result' and 'result_margin' columns as they are not needed
final.drop(columns=['result', 'result_margin'], inplace=True)

# Display the final DataFrame
print(final.head())


# %%
final.tail()

# %%
final.drop(columns=["id","toss_winner","season","venue","player_of_match","winner"], inplace=True)

# %%
final.columns

# %%
final.info()

# %%
from sklearn.preprocessing import StandardScaler
# Initialize StandardScaler for 'target_runs' column only
scaler = StandardScaler()
# Fit and transform the 'target_runs' column
final['target_runs_scaled'] = scaler.fit_transform(final[['target_runs']])

# Drop the original 'target_runs' column (optional, if you only want the scaled version)
final.drop(columns=['target_runs'], inplace=True)


# %%
# Separate the training and testing datasets
train = final[final['season_mapped'] != 17]
test = final[final['season_mapped'] == 17]

# Display the shapes of the training and testing datasets
print(f"Shape of training set: {train.shape}")
print(f"Shape of testing set: {test.shape}")


# %%


# Separate X (features) and y (target variables)
X_train = train.drop(columns=['runs_in_p1', 'runs_in_p2', 'runs_in_p3'])
y_train = train[['runs_in_p1', 'runs_in_p2', 'runs_in_p3']]
X_test = test.drop(columns=['runs_in_p1', 'runs_in_p2', 'runs_in_p3'])
y_test = test[['runs_in_p1', 'runs_in_p2', 'runs_in_p3']]


# %%
X_train[X_train["season_mapped"]==16]

# %%
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Step 3: Prepare DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Step 4: Set parameters for the XGBoost model
params = {
    'objective': 'reg:squarederror',  # Regression task
    'eval_metric': 'rmse',             # Evaluation metric
    'learning_rate': 0.1,               # Step size shrinkage
    'max_depth': 6,                     # Maximum depth of a tree
    'alpha': 10,                        # L1 regularization term
    'n_estimators': 100                 # Number of boosting rounds
}

# Step 5: Train the XGBoost model
model = xgb.train(params, dtrain, num_boost_round=100)

# Step 6: Make predictions on the test set
y_pred = model.predict(dtest)

# Step 7: Calculate RMSE for the predictions
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE on test set: {rmse}")

# %%
import matplotlib.pyplot as plt

# Assuming y_test is in a DataFrame and has three columns: runs_in_p1, runs_in_p2, runs_in_p3
# For the sake of this example, let's assume we are focusing on the first target variable (runs_in_p1)

# Extract the first column of y_test for actual values
actual_values = y_test['runs_in_p1'].values
predicted_values = y_pred[:, 0]  # Get predictions for runs_in_p1

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(actual_values, predicted_values, alpha=0.7)
plt.plot([actual_values.min(), actual_values.max()], [actual_values.min(), actual_values.max()], 'k--', lw=2)  # Diagonal line
plt.title('Predicted vs Actual Runs (p1)')
plt.xlabel('Actual Runs')
plt.ylabel('Predicted Runs')
plt.grid()
plt.show()


# %%
# Create a DataFrame to hold actual and predicted values
results = pd.DataFrame({
    'Actual Runs in p1': y_test['runs_in_p1'].values,
    'Predicted Runs in p1': y_pred[:, 0],  # Predictions for runs_in_p1
    'Actual Runs in p2': y_test['runs_in_p2'].values,
    'Predicted Runs in p2': y_pred[:, 1],  # Predictions for runs_in_p2
    'Actual Runs in p3': y_test['runs_in_p3'].values,
    'Predicted Runs in p3': y_pred[:, 2]   # Predictions for runs_in_p3
})

# Display the results
print(results)


# %%



