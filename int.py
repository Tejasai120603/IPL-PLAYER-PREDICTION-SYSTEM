# %%
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# %%
delivery = pd.read_csv("deliveries.csv")
matches = pd.read_csv("matches.csv")

# %%
matches["season"] = matches["season"].replace({
    '2007/08': '2008',
    '2009/10': '2010',
    '2020/21': '2020'
})

# %%
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
player_name = "RG Sharma"
player_team = "Mumbai Indians"

# %%
ipl_player = ipl[ipl["batter"] == player_name].copy()

# %%
player_runs = ipl_player[['id', 'batsman_runs', 'Power Play Number']]

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

# %%
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
unique_ids = player_runs['id'].unique()

# Initialize the 'runs_in_p1', 'runs_in_p2', 'runs_in_p3' columns in 'final' DataFrame
final['runs_in_p1'] = 0
final['runs_in_p2'] = 0
final['runs_in_p3'] = 0

# Iterate over each unique match ID
for match_id in unique_ids:
    # Extract the corresponding runs for each Power Play from player_runs
    runs_in_p1 = player_runs.loc[player_runs["id"] == match_id, "runs_in_p1"].iloc[0]
    runs_in_p2 = player_runs.loc[player_runs["id"] == match_id, "runs_in_p2"].iloc[0]
    runs_in_p3 = player_runs.loc[player_runs["id"] == match_id, "runs_in_p3"].iloc[0]

    # Assign these values to the corresponding rows in the final DataFrame
    final.loc[final["id"] == match_id, 'runs_in_p1'] = runs_in_p1
    final.loc[final["id"] == match_id, 'runs_in_p2'] = runs_in_p2
    final.loc[final["id"] == match_id, 'runs_in_p3'] = runs_in_p3

# %%
# One-hot encoding and feature engineering
from sklearn.preprocessing import OneHotEncoder

encoder_team = OneHotEncoder(drop='first', sparse_output=False)
encoded_teams = encoder_team.fit_transform(final[['bowling_team']])
team_df = pd.DataFrame(encoded_teams, columns=encoder_team.get_feature_names_out(['bowling_team']))

encoder_city = OneHotEncoder(drop='first', sparse_output=False)
encoded_city = encoder_city.fit_transform(final[['city']])
city_df = pd.DataFrame(encoded_city, columns=encoder_city.get_feature_names_out(['city']))

final = pd.concat([final, team_df, city_df], axis=1)
final.drop(columns=['bowling_team', 'city'], inplace=True)

final['toss_winner_is_player_team'] = (final['toss_winner'] == player_team).astype(int)

season_mapping = {str(year): idx + 1 for idx, year in enumerate(range(2008, 2025))}
final['season_mapped'] = final['season'].map(season_mapping)

final['player_of_match_is_player'] = (final['player_of_match'] == player_name).astype(int)
final['winner_is_player_team'] = (final['winner'] == player_team).astype(int)

final.drop(columns=['result', 'result_margin'], inplace=True)

# %%
# Scale the 'target_runs' column
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
final['target_runs_scaled'] = scaler.fit_transform(final[['target_runs']])
final.drop(columns=['target_runs'], inplace=True)

# %%
# Split the data into training and test sets
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

# %%
# Predictions and performance metrics
y_pred = model.predict(dtest)

# Calculate RMSE for the predictions
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE on test set: {rmse}")

# Calculate R² (accuracy) for each phase (p1, p2, p3)
r2_p1 = r2_score(y_test['runs_in_p1'], y_pred[:, 0])
r2_p2 = r2_score(y_test['runs_in_p2'], y_pred[:, 1])
r2_p3 = r2_score(y_test['runs_in_p3'], y_pred[:, 2])

# Print R² values
print(f"R² score (accuracy) for runs_in_p1: {r2_p1}")
print(f"R² score (accuracy) for runs_in_p2: {r2_p2}")
print(f"R² score (accuracy) for runs_in_p3: {r2_p3}")

# Calculate the average R² score
average_r2 = np.mean([r2_p1, r2_p2, r2_p3])
print(f"Average R² score (overall accuracy): {average_r2}")

# %%
# Scatter plot for actual vs predicted runs in Power Play 1
actual_values = y_test['runs_in_p1'].values
predicted_values = y_pred[:, 0]  # Predictions for runs_in_p1

plt.figure(figsize=(10, 6))
plt.scatter(actual_values, predicted_values, alpha=0.7)
plt.plot([actual_values.min(), actual_values.max()], [actual_values.min(), actual_values.max()], 'k--', lw=2)  # Diagonal line
plt.title('Predicted vs Actual Runs (p1)')
plt.xlabel('Actual Runs')
plt.ylabel('Predicted Runs')
plt.grid()
plt.show()

# %%
# Display actual and predicted values for all phases (p1, p2, p3)
results = pd.DataFrame({
    'Actual Runs in p1': y_test['runs_in_p1'].values,
    'Predicted Runs in p1': y_pred[:, 0],  # Predictions for runs_in_p1
    'Actual Runs in p2': y_test['runs_in_p2'].values,
    'Predicted Runs in p2': y_pred[:, 1],  # Predictions for runs_in_p2
    'Actual Runs in p3': y_test['runs_in_p3'].values,
    'Predicted Runs in p3': y_pred[:, 2]   # Predictions for runs_in_p3
})

print(results.head())
