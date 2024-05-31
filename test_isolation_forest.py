import pandas as pd
import joblib

# Load the trained model
model = joblib.load('isolation_forest_model.pkl')

# Load the label encoder
label_encoder = joblib.load('label_encoder.pkl')

# Create a DataFrame with the single test input
test_input = {
    'user_id': ['023c8254-61ab-44db-ad99-1539813d996a'],
    'avg_start_finish_login_time_seconds': [150],
    'avg_hour_of_login': [18],
    'most_common_device': ['iPhone'],
    'os_used': ['iOS'],
    'login_with_android': [0],
    'login_with_iOS': [1],
    'num_logins': [10]
}
test_input_df = pd.DataFrame(test_input)

# Convert categorical columns to numerical using LabelEncoder
test_input_df['most_common_device'] = label_encoder.transform(test_input_df['most_common_device'])
test_input_df['os_used'] = label_encoder.transform(test_input_df['os_used'])


# Define a function to map unknown categories to a default value
def handle_unknown_category(category):
    if category in label_encoder.classes_:
        return label_encoder.transform([category])[0]
    else:
        # Return a default value for unknown category
        return label_encoder.transform(['Unknown'])[0]


# Drop user_id column as it's not needed for prediction
X_test_input = test_input_df.drop(['user_id'], axis=1)

# Convert the 'most_common_device' column to numerical
X_test_input['most_common_device'] = X_test_input['most_common_device'].apply(handle_unknown_category)

# Predict on the test input data
anomaly_score_test = model.predict(X_test_input)


# Map anomaly score to category: OK, Not OK, Maybe
def map_to_category(score):
    if score == 1:
        return 'OK'
    elif score == -1:
        return 'Not OK'
    else:
        return 'Maybe'


# Apply the mapping function to the anomaly scores
predictions = [map_to_category(score) for score in anomaly_score_test]

# Add the predictions to the DataFrame
test_input_df['prediction'] = predictions

# Print the prediction
print("Prediction for the test input:")
print(test_input_df[['user_id', 'prediction']])
