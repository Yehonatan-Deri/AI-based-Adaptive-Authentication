import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

datetime_format = "%b %d, %Y @ %H:%M:%S.%f"


# Load user profiles
user_profiles_df = pd.read_csv('user_profiles.csv', index_col='user_id')

# Load your data
data = pd.read_csv('data_after_dropped.csv')

# Merge user profiles with data
data = data.merge(user_profiles_df, on='user_id', how='left')

# Feature Engineering - You might have already done this
# Feature Engineering
data['@timestamp'] = pd.to_datetime(data['@timestamp'], format=datetime_format)
data['event.created'] = pd.to_datetime(data['event.created'], format=datetime_format)
data['time_difference'] = (data['event.created'] - data['@timestamp']).dt.total_seconds()

# Feature selection
features = ['time_difference', 'user_agent.device.name', 'user_agent.os.full', 'user_id']
X = data[features]
features = ['time_difference', 'user_agent.device.name', 'user_agent.os.full', 'num_logins']
X = data[features]

# One-hot encode categorical variables
cat_columns = ['user_agent.device.name', 'user_agent.os.full']
X_encoded = pd.get_dummies(X, columns=cat_columns)

# Labeling anomalies - You would need to define what constitutes an anomaly
data['anomaly_label'] = 0  # Start with all as normal

# Set anomalies based on deviations from user profile
for index, row in data.iterrows():
    if row['time_difference'] > (row['avg_login_duration'] * 2):
        data.at[index, 'anomaly_label'] = 1  # Set as anomaly if login duration is more than twice the average

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, data['anomaly_label'], test_size=0.2, random_state=42)

# Train the Isolation Forest model
model = IsolationForest(contamination=0.05)
model.fit(X_train)

# Predict anomalies on the test set
predictions = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# # Labeling anomalies - You would need to define what constitutes an anomaly
# data['anomaly_label'] = 0  # Start with all as normal
#
# # Set anomalies based on deviations from user profile
# for index, row in data.iterrows():
#     if row['time_difference'] > (row['avg_login_duration'] * 2):
#         data.at[index, 'anomaly_label'] = 1  # Set as anomaly if login duration is more than twice the average
#
#
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, data['anomaly_label'], test_size=0.2, random_state=42)
#
# # Train the Isolation Forest model
# model = IsolationForest(contamination=0.05)  # Contamination is the expected proportion of anomalies
# model.fit(X_train)
#
# # Predict anomalies on the test set
# predictions = model.predict(X_test)
#
#
#
# # Evaluate the model
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, predictions))
#
# print("\nClassification Report:")
# print(classification_report(y_test, predictions))

# print(data['anomaly_label'].value_counts())