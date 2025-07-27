# %%
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt

# %%
# Load the data
delivery = pd.read_csv("deliveries.csv")
matches = pd.read_csv("matches.csv")

# %%
# Fix the season format
matches["season"] = matches["season"].replace({
    '2007/08': '2008',
    '2009/10': '2010',
    '2020/21': '2020'
})

# %%
# Merge the deliveries and matches data
ipl = pd.merge(delivery, matches, on='id')

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

# %%
# Filter data for a specific player
player_name = "RG Sharma"
player_team = "Mumbai Indians"
ipl_player = ipl[ipl["batter"] == player_name].copy()

# %%
# Summarize runs in each Power Play
player_runs = ipl_player[['id', 'batsman_runs', 'Power Play Number']].copy()
player_runs['runs_in_p1'] = 0
player_runs['runs_in_p2'] = 0
player_runs['runs_in_p3'] = 0

# Get the unique match ids
unique_ids = player_runs['id'].unique()

# Iterate over each match id to calculate runs in each Power Play
for match_id in unique_ids:
    runs_in_p1 = player_runs[(player_runs["id"] == match_id) & (player_runs["Power Play Number"] == '1')]["batsman_runs"].sum()
    runs_in_p2 = player_runs[(player_runs["id"] == match_id) & (player_runs["Power Play Number"] == '2')]["batsman_runs"].sum()
    runs_in_p3 = player_runs[(player_runs["id"] == match_id) & (player_runs["Power Play Number"] == '3')]["batsman_runs"].sum()

    player_runs.loc[player_runs["id"] == match_id, 'runs_in_p1'] = runs_in_p1
    player_runs.loc[player_runs["id"] == match_id, 'runs_in_p2'] = runs_in_p2
    player_runs.loc[player_runs["id"] == match_id, 'runs_in_p3'] = runs_in_p3

# %%
# Group data by match ID for further processing
grouped = ipl_player.groupby('id').agg({
    'inning': 'first',
    'target_runs': 'first',
    'bowling_team': 'first',
    'toss_winner': 'first',
    'season': 'first',
    'city': 'first',
    'player_of_match': 'first',
    'venue': 'first',
    'winner': 'first',
    'result': 'first',
    'result_margin': 'first'
})

final = grouped.reset_index()

# %%
# Merge Power Play runs into the final DataFrame
for match_id in unique_ids:
    runs_in_p1 = player_runs.loc[player_runs["id"] == match_id, "runs_in_p1"].iloc[0]
    runs_in_p2 = player_runs.loc[player_runs["id"] == match_id, "runs_in_p2"].iloc[0]
    runs_in_p3 = player_runs.loc[player_runs["id"] == match_id, "runs_in_p3"].iloc[0]
    final.loc[final["id"] == match_id, 'runs_in_p1'] = runs_in_p1
    final.loc[final["id"] == match_id, 'runs_in_p2'] = runs_in_p2
    final.loc[final["id"] == match_id, 'runs_in_p3'] = runs_in_p3

# %%
# One-hot encoding for 'bowling_team' and 'city'
encoder_team = OneHotEncoder(drop='first', sparse_output=False)
encoded_teams = encoder_team.fit_transform(final[['bowling_team']])
team_df = pd.DataFrame(encoded_teams, columns=encoder_team.get_feature_names_out(['bowling_team']))

encoder_city = OneHotEncoder(drop='first', sparse_output=False)
encoded_city = encoder_city.fit_transform(final[['city']])
city_df = pd.DataFrame(encoded_city, columns=encoder_city.get_feature_names_out(['city']))

final = pd.concat([final, team_df, city_df], axis=1)
final.drop(columns=['bowling_team', 'city'], inplace=True)

# Binary column for 'toss_winner' if it matches 'player_team'
final['toss_winner_is_player_team'] = (final['toss_winner'] == player_team).astype(int)

# Mapping for 'season'
season_mapping = {str(year): idx + 1 for idx, year in enumerate(range(2008, 2025))}
final['season_mapped'] = final['season'].map(season_mapping)

# Binary columns for 'player_of_match' and 'winner'
final['player_of_match_is_player'] = (final['player_of_match'] == player_name).astype(int)
final['winner_is_player_team'] = (final['winner'] == player_team).astype(int)

# Drop unnecessary columns
final.drop(columns=['result', 'result_margin', 'id', 'toss_winner', 'season', 'venue', 'player_of_match', 'winner'], inplace=True)

# %%
# Standardize 'target_runs'
scaler = StandardScaler()
final['target_runs_scaled'] = scaler.fit_transform(final[['target_runs']])
final.drop(columns=['target_runs'], inplace=True)

# %%
# Split data into training and test sets
train = final[final['season_mapped'] != 17]
test = final[final['season_mapped'] == 17]

X_train = train.drop(columns=['runs_in_p1', 'runs_in_p2', 'runs_in_p3'])
y_train = train[['runs_in_p1', 'runs_in_p2', 'runs_in_p3']]
X_test = test.drop(columns=['runs_in_p1', 'runs_in_p2', 'runs_in_p3'])
y_test = test[['runs_in_p1', 'runs_in_p2', 'runs_in_p3']]

# %%
# Train the XGBoost model
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.1,
    'max_depth': 6,
    'alpha': 10,
    'n_estimators': 100
}

model = xgb.train(params, dtrain, num_boost_round=100)

# Make predictions
y_pred = model.predict(dtest)

# %%
# Calculate and print RMSE for each phase
rmse_p1 = mean_squared_error(y_test['runs_in_p1'], y_pred[:, 0], squared=False)
rmse_p2 = mean_squared_error(y_test['runs_in_p2'], y_pred[:, 1], squared=False)
rmse_p3 = mean_squared_error(y_test['runs_in_p3'], y_pred[:, 2], squared=False)

print(f"RMSE for Power Play 1 (p1): {rmse_p1}")
print(f"RMSE for Power Play 2 (p2): {rmse_p2}")
print(f"RMSE for Power Play 3 (p3): {rmse_p3}")

average_rmse = (rmse_p1 + rmse_p2 + rmse_p3) / 3
print(f"Average RMSE across all phases: {average_rmse}")

# %%
# Scatter plot of predicted vs actual runs for p1
actual_values = y_test['runs_in_p1'].values
predicted_values = y_pred[:, 0]

plt.figure(figsize=(10, 6))
plt.scatter(actual_values, predicted_values, alpha=0.7)
plt.plot([actual_values.min(), actual_values.max()], [actual_values.min(), actual_values.max()], 'k--', lw=2)
plt.title('Predicted vs Actual Runs (p1)')
plt.xlabel('Actual Runs')
plt.ylabel('Predicted Runs')
plt.grid()
plt.show()

# %%
# Create a DataFrame for actual and predicted values
results = pd.DataFrame({
    'Actual Runs in p1': y_test['runs_in_p1'].values,
    'Predicted Runs in p1': y_pred[:, 0],
    'Actual Runs in p2': y_test['runs_in_p2'].values,
    'Predicted Runs in p2': y_pred[:, 1],
    'Actual Runs in p3': y_test['runs_in_p3'].values,
    'Predicted Runs in p3': y_pred[:, 2]
})

# Display the results
print(results)
