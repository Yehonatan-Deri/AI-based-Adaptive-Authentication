import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score

from sandbox import data_collector

datetime_format = "%b %d, %Y @ %H:%M:%S.%f"

# Load your user profiles data
user_profiles_df = data_collector.get_user_profiles(df_path='csv_dir/data_year_status.csv')

# logs_data = data_collector.load_df(df_path='csv_dir/data_after_dropped.csv')
logs_data = data_collector.load_df(df_path='csv_dir/test_year_status.csv')
logs_data['@timestamp'] = pd.to_datetime(logs_data['@timestamp'], format=datetime_format)
logs_data['event.created'] = pd.to_datetime(logs_data['event.created'], format=datetime_format)
logs_data['@timestamp'] = pd.to_numeric(logs_data['@timestamp'])
logs_data['event.created'] = pd.to_numeric(logs_data['event.created'])

selected_features = ['@timestamp', 'Android sum', 'iOS sum', 'log_type', 'user_agent.device.name',
                     'user_id', 'auth_id']
logs_data_selected = logs_data[selected_features]

# Handle missing values (replace with your chosen strategy)
logs_data_selected['user_agent.device.name'].fillna('unknown', inplace=True)
logs_data_selected['user_id'].fillna('unknown', inplace=True)


# change 20% of the data to be anomalies by:
# - adding hours to the timestamp or subtracting hours from the timestamp by 6-10 hours
#   do this for same start and finish by auth_id
# - change the phone type to be different from the most common device
#   if it android change it to iOS and vice versa
# and edit logs_data_selected

# anomaly detection
# 0: normal
# 1: anomaly

# 1. Add 6-10 hours to the timestamp for 20% of the data
# Select 20% of the data to be modified start and finish of the same auth_id
# find pairs start and finish by auth_id
auth_ids = logs_data_selected['auth_id'].unique()
pairs = []
for auth_id in auth_ids:
    auth_id_data = logs_data_selected[logs_data_selected['auth_id'] == auth_id]
    if auth_id_data.shape[0] == 2:
        pairs.append(auth_id_data)

# select 20% of the pairs
anomaly_pairs = pd.concat(pairs).sample(frac=0.2)
logs_data_selected.loc[anomaly_pairs.index, '@timestamp'] = logs_data_selected.loc[anomaly_pairs.index, '@timestamp'] + 8 * 3600


# 2. Change the phone type for 10% of the data
# Select 20% of the data to be modified
anomaly_data = logs_data_selected.sample(frac=0.1)
# change the phone type to be different from the most common device
for index, row in anomaly_data.iterrows():
    if row['user_agent.device.name'] == 'Generic Smartphone':
        logs_data_selected.at[index, 'user_agent.device.name'] = 'iPhone'
    else:
        logs_data_selected.at[index, 'user_agent.device.name'] = 'XiaoMi Redmi Note 9S, Xiaomi, curtana, curtana_global'

logs_data_selected.drop(['auth_id'], axis=1, inplace=True)

# Convert categorical columns to numerical using LabelEncoder
# Create separate encoders for each column
log_type_encoder = LabelEncoder()
device_name_encoder = LabelEncoder()
user_id_encoder = LabelEncoder()

logs_data_selected['log_type'] = log_type_encoder.fit_transform(logs_data_selected['log_type'])
logs_data_selected['user_agent.device.name'] = device_name_encoder.fit_transform(logs_data_selected['user_agent.device.name'])
logs_data_selected['user_id'] = user_id_encoder.fit_transform(logs_data_selected['user_id'])

# Split data into training and testing sets (adjust test_size as needed)
X_train, X_test = train_test_split(logs_data_selected, test_size=0.3, random_state=42)


# Train the Isolation Forest model
model = IsolationForest(n_estimators=100)
model.fit(X_train)

# Predict anomaly scores for test data
y_pred = model.decision_function(X_test)


# Set a threshold for anomaly classification (adjust based on risk tolerance)
threshold = 0.01  # You can adjust this value

# Identify potential anomalies (logins with scores below the threshold)
potential_anomalies = X_test.loc[(abs(y_pred) > threshold)]

# invert the encoding for the user_agent.device.name, log_type, and user_id columns
potential_anomalies['user_agent.device.name'] = device_name_encoder.inverse_transform(potential_anomalies['user_agent.device.name'])
potential_anomalies['log_type'] = log_type_encoder.inverse_transform(potential_anomalies['log_type'])
potential_anomalies['user_id'] = user_id_encoder.inverse_transform(potential_anomalies['user_id'])
potential_anomalies['@timestamp'] = pd.to_datetime(potential_anomalies['@timestamp'], unit='ns')
# Investigate potential anomalies (further analysis needed to confirm)
print("Potential anomalies:")
print(potential_anomalies)

# accuracy
print("Accuracy: ", potential_anomalies.shape[0] / X_test.shape[0])



# # Select relevant features for the model
# # selected_features = ['avg_start_finish_login_time_seconds', 'avg_hour_of_login', 'most_common_device',
# #                      'num_logins', 'login_with_android', 'login_with_iOS']
# selected_features = ['avg_hour_of_login', 'most_common_device', 'login_with_android', 'login_with_iOS']
# selected_profile_features = user_profiles_df[selected_features]
#
# # Create label encoder for most_common_device
# encoder = LabelEncoder()
# encoder.fit(user_profiles_df['most_common_device'])
#
# # Encode most_common_device in training data
# selected_profile_features.loc[:, 'most_common_device'] = encoder.transform(selected_profile_features['most_common_device'])
#
# # Create Isolation Forest model
# model = IsolationForest(n_estimators=100)  # Adjust parameters as needed
#
# # Train the model
# model.fit(selected_profile_features)
#
# # Get anomaly scores for new data (replace 'new_data' with your actual data)
# # Create a DataFrame with the single test input
# # data prediction: Not OK
# test_input = {
#     'user_id': ['023c8254-61ab-44db-ad99-1539813d996a'],
#     'avg_start_finish_login_time_seconds': [150],
#     'avg_hour_of_login': [18],
#     'most_common_device': ['iPhone'],
#     'os_used': ['iOS'],
#     'login_with_android': [0],
#     'login_with_iOS': [1],
#     'num_logins': [10]
# }
#
# # # data prediction: OK
# # test_input = {
# #     'user_id': ['023c8254-61ab-44db-ad99-1539813d996a'],
# #     'avg_start_finish_login_time_seconds': [20],
# #     'avg_hour_of_login': [18],
# #     'most_common_device': ['Generic Smartphone'],
# #     'os_used': ['Android'],
# #     'login_with_android': [1],
# #     'login_with_iOS': [0],
# #     'num_logins': [10]
# # }
#
# test_input_df = pd.DataFrame(test_input)
# test_selected_features = test_input_df[selected_features]
# # Encode most_common_device in test data
# test_selected_features.loc[:, 'most_common_device'] = encoder.transform(test_selected_features['most_common_device'])
#
#
# new_score = model.decision_function(test_selected_features)
#
# # Define anomaly threshold (adjust based on model evaluation)
# threshold = -0.04  # Example threshold (adjust based on model evaluation)
#
# anomaly_label = 'OK' if new_score > threshold else 'Not OK'
#
# # Optionally, add 'Maybe' category for borderline cases
# maybe_threshold = threshold * 1.1  # Example, slightly higher than main threshold
# anomaly_label = 'Maybe' if threshold < new_score <= maybe_threshold else anomaly_label
#
# print(f"Anomaly Label: {anomaly_label}")



# # Convert categorical columns to numerical using LabelEncoder
# label_encoder = LabelEncoder()
# user_profiles_df['most_common_device'] = label_encoder.fit_transform(user_profiles_df['most_common_device'])
# user_profiles_df['os_used'] = label_encoder.fit_transform(user_profiles_df['os_used'])
#
# # Drop user_id column as it's not needed for training
# X_train = user_profiles_df.drop(['user_id'], axis=1)
#
# # Initialize and train the Isolation Forest model
# model = IsolationForest(random_state=42)
# model.fit(X_train)
#
# # Save the trained model
# joblib.dump(model, 'isolation_forest_model.pkl')
#
# # Save the label encoder
# joblib.dump(label_encoder, 'label_encoder.pkl')
#
# print("Model trained and saved successfully!")
