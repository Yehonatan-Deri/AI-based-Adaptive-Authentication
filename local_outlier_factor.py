from collections import Counter
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns

# part 1
# region Load the CSV file
file_path = 'csv_dir/test_year_status.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
df.head()
# endregion

# part 2
# region Preprocess the data
# Remove the "@" character from the timestamp
df['@timestamp'] = df['@timestamp'].str.replace('@', '').str.strip()

# Convert the cleaned timestamp to datetime
df['@timestamp'] = pd.to_datetime(df['@timestamp'], format='%b %d, %Y %H:%M:%S.%f')

# Extract hour from timestamp for login time analysis
df['login_hour'] = df['@timestamp'].dt.hour

# Extract auth_id from the message
df['auth_id'] = df['message'].str.extract(r'auth\s+([A-Za-z0-9]+)')

# Mark if the message contains 'denied' or 'approved'
df['is_denied'] = df['message'].str.contains('denied', case=False, na=False)
df['is_approved'] = df['message'].str.contains('approved', case=False, na=False)

# Fill missing user_id in finish actions by matching start actions based on auth_id
start_actions = df[df['message'].str.contains('start', case=False, na=False)][['auth_id', 'user_id']]
df = df.merge(start_actions, on='auth_id', suffixes=('', '_start'), how='left')
df['user_id'] = df['user_id'].fillna(df['user_id_start'])
df.drop(columns=['user_id_start'], inplace=True)

# Identify incomplete sessions (sessions without any 'denied' or 'approved')
incomplete_sessions = df.groupby('auth_id').filter(
    lambda x: (x['is_denied'].sum() == 0) & (x['is_approved'].sum() == 0))

# Get the list of incomplete auth_id
incomplete_auth_ids = incomplete_sessions['auth_id'].unique()

# Mark these incomplete sessions as denied
df.loc[df['auth_id'].isin(incomplete_auth_ids), 'is_denied'] = True

# Extract hour from timestamp for login time analysis
df['login_hour'] = df['@timestamp'].dt.hour

# Calculate the number of logins per user per hour
login_counts = df.groupby(['user_id', 'login_hour']).size().reset_index(name='login_count')


# endregion

# part 3
# region Create a user profile dataframe and feature selection
# Create a function to calculate session duration
def calculate_session_duration(timestamps):
    timestamps = pd.to_datetime(timestamps).sort_values()
    durations = timestamps.diff().dropna().dt.total_seconds()
    return durations.mean() if len(durations) > 0 else 0


# Create detailed user profiles
user_profiles = df.groupby('user_id').agg({
    'login_hour': ['mean', 'std'],
    'user_agent.os.name': lambda x: x.nunique(),
    'iOS sum': 'sum',
    'Android sum': 'sum',
    'is_denied': 'mean',
    '@timestamp': calculate_session_duration
}).reset_index()

# Flatten MultiIndex columns
user_profiles.columns = ['_'.join(col).strip() for col in user_profiles.columns.values]

# Rename columns for clarity
user_profiles.rename(columns={
    'login_hour_mean': 'average_login_hour',
    'login_hour_std': 'login_hour_std_dev',
    'user_agent.os.name_<lambda>': 'device_changes',
    'iOS sum_sum': 'total_ios_actions',
    'Android sum_sum': 'total_android_actions',
    'is_denied_mean': 'denial_rate',
    '@timestamp_calculate_session_duration': 'average_session_duration'
}, inplace=True)

# Replace NaN in login_hour_std_dev with 0 to indicate no variability
user_profiles['login_hour_std_dev'] = user_profiles['login_hour_std_dev'].fillna(0)

# Summarize login counts
login_counts_summary = login_counts.groupby('user_id')['login_count'].sum().reset_index()

# Rename user_id columns to user_id_ for merging matching columns
login_counts_summary.rename(columns={'user_id': 'user_id_'}, inplace=True)

# Merge login counts into user profiles
user_profiles = user_profiles.merge(login_counts_summary, on='user_id_', how='left')
user_profiles.rename(columns={'login_count': 'total_logins'}, inplace=True)

features = user_profiles[['average_login_hour', 'login_hour_std_dev', 'device_changes', 'total_ios_actions',
                          'total_android_actions', 'denial_rate', 'average_session_duration', 'total_logins']]

user_profiles.head()
# endregion

# part 4
# region Validate Denial Rate Calculation

# Count the total number of sessions and denied sessions per user
user_denied_sessions = df[df['is_denied']].groupby('user_id').size()
user_total_sessions = df.groupby('user_id').size()

# Calculate the denial rate as the ratio of denied sessions to total sessions
user_profiles['denial_rate'] = user_profiles['user_id_'].map(user_denied_sessions) / user_profiles['user_id_'].map(
    user_total_sessions)

# Replace NaN denial rates with 0 (users with no denied sessions)
user_profiles['denial_rate'] = user_profiles['denial_rate'].fillna(0)
# endregion

# part 5
# region Scale the features
# Scale the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Initialize the Local Outlier Factor model
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)

# Fit the model and predict the outliers
y_pred = lof.fit_predict(features_scaled)

# The negative outlier factor (NOF) for each sample
lof_scores = -lof.negative_outlier_factor_

# Identify outliers
user_profiles['outlier'] = y_pred

print("Number of outliers detected:", len(user_profiles[user_profiles['outlier'] == -1]))

user_profiles.head()

# Define feature mapping dictionary
feature_mapping = {
    'login_hour': 'login_hour',
    'device_changes': 'user_agent.os.name',
    'total_ios_actions': 'iOS sum',
    'total_android_actions': 'Android sum',
    'denial_rate': 'is_denied',
    'average_session_duration': '@timestamp'
}


# Function to train individual models
def train_user_model(user_id, df):
    user_data = df[df['user_id'] == user_id]
    if len(user_data) < 5:  # Ensure enough data points
        return None, None

    # Select relevant features
    features = user_data[[feature_mapping['login_hour'],
                          feature_mapping['device_changes'],
                          feature_mapping['total_ios_actions'],
                          feature_mapping['total_android_actions'],
                          feature_mapping['denial_rate']]]

    # Convert all features to numeric
    features = features.apply(pd.to_numeric, errors='coerce').fillna(0)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    lof = LocalOutlierFactor(n_neighbors=5, contamination=0.1)
    lof.fit(features_scaled)

    return lof, scaler


# Dictionary to hold user models and scalers
user_models = {}
user_scalers = {}

for user_id in df['user_id'].unique():
    model, scaler = train_user_model(user_id, df)
    if model:
        user_models[user_id] = model
        user_scalers[user_id] = scaler


# Function to predict if a new login is an anomaly
def predict_user_login(user_id, login_features, threshold_inbetween=-1.5, threshold_invalid=-3.0):
    if user_id not in user_models:
        return "valid"  # If no model for the user, assume valid

    model = user_models[user_id]
    scaler = user_scalers[user_id]

    # Get the existing user data
    user_data = df[df['user_id'] == user_id]
    if len(user_data) < 5:
        return "valid"  # If not enough data points, assume valid

    # Select relevant features
    existing_features = user_data[[feature_mapping['login_hour'],
                                   feature_mapping['device_changes'],
                                   feature_mapping['total_ios_actions'],
                                   feature_mapping['total_android_actions'],
                                   feature_mapping['denial_rate']]]

    # Convert all features to numeric
    existing_features = existing_features.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Combine existing features with the new login features
    combined_features = np.vstack((existing_features, login_features))

    # Scale the combined features
    combined_features_scaled = scaler.fit_transform(combined_features)

    # Fit the model on the combined data
    model.fit(combined_features_scaled)

    # Get the LOF score for the new login
    lof_score = model.negative_outlier_factor_[-1]

    # Determine the category based on thresholds
    if lof_score > threshold_inbetween:
        return "valid"
    elif threshold_invalid < lof_score <= threshold_inbetween:
        return "in-between"
    else:
        return "not valid"


# Example: Predicting a new login
new_login = {
    'user_id': 'example_user',
    'login_hour': 14,
    'user_agent.os.name': 1,
    'iOS sum': 0,
    'Android sum': 1,
    'is_denied': 0
}

login_features = [
    new_login[feature_mapping['login_hour']],
    new_login[feature_mapping['device_changes']],
    new_login[feature_mapping['total_ios_actions']],
    new_login[feature_mapping['total_android_actions']],
    new_login[feature_mapping['denial_rate']]
]

prediction = predict_user_login(new_login['user_id'], login_features)
print(f"New login prediction: {prediction}")

# Continue from where we left off
# Make predictions for all users
predictions = []
for user_id in df['user_id'].unique():
    user_data = df[df['user_id'] == user_id]
    if len(user_data) < 5:  # Ensure enough data points
        continue

    # Select relevant features
    features = user_data[[feature_mapping['login_hour'],
                          feature_mapping['device_changes'],
                          feature_mapping['total_ios_actions'],
                          feature_mapping['total_android_actions'],
                          feature_mapping['denial_rate']]].tail(1)

    # Convert all features to numeric
    features = features.apply(pd.to_numeric, errors='coerce').fillna(0)

    login_features = features.values[0]

    prediction = predict_user_login(user_id, login_features)
    predictions.append((user_id, prediction))

# Count occurrences of each category
count = Counter([prediction for _, prediction in predictions])

# Prepare data for the pie chart
labels = count.keys()
sizes = count.values()
colors = ['lightgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0, 0)  # explode 1st slice (valid)

# Plotting the pie chart
plt.figure(figsize=(8, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Distribution of Login Predictions')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# Part 6: Plot the Results
# Plot average login time and identify outliers
plt.figure(figsize=(10, 6))
plt.scatter(user_profiles['average_login_hour'], user_profiles['login_hour_std_dev'], c=user_profiles['outlier'], cmap='coolwarm', edgecolors='k')
plt.xlabel('Average Login Hour')
plt.ylabel('Login Hour Std Deviation')
plt.title('User Login Behavior - Outliers Detection')
plt.colorbar(label='Outlier')
plt.show()

# Plot device type changes
plt.figure(figsize=(10, 6))
plt.scatter(user_profiles['device_changes'], user_profiles['outlier'], c=user_profiles['outlier'], cmap='coolwarm', edgecolors='k')
plt.xlabel('Number of Different Devices Used')
plt.ylabel('Outlier')
plt.title('Device Type Changes - Outliers Detection')
plt.colorbar(label='Outlier')
plt.show()

# Plot login outcomes
plt.figure(figsize=(10, 6))
plt.scatter(user_profiles['denial_rate'], user_profiles['outlier'], c=user_profiles['outlier'], cmap='coolwarm', edgecolors='k')
plt.xlabel('Denial Rate')
plt.ylabel('Outlier')
plt.title('Login Outcomes - Outliers Detection')
plt.colorbar(label='Outlier')
plt.show()

# Plot session duration
plt.figure(figsize=(10, 6))
plt.scatter(user_profiles['average_session_duration'], user_profiles['outlier'], c=user_profiles['outlier'], cmap='coolwarm', edgecolors='k')
plt.xlabel('Average Session Duration (seconds)')
plt.ylabel('Outlier')
plt.title('Session Duration - Outliers Detection')
plt.colorbar(label='Outlier')
plt.show()

# Plot the number of logins per user per hour
plt.figure(figsize=(12, 6))
sns.histplot(login_counts['login_count'], bins=20, kde=True)
plt.xlabel('Number of Logins per Hour')
plt.ylabel('Frequency')
plt.title('Distribution of Number of Logins per User per Hour')
plt.show()

