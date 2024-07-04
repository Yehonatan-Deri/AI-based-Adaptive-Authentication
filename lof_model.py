# lof_model.py

import pandas as pd
import preprocess_data
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from user_profiling import UserProfiler


class LOFModel:
    def __init__(self, preprocessed_df):
        self.profiler = UserProfiler(preprocessed_df)
        self.user_models = {}
        self.user_scalers = {}

    def train_user_model(self, user_id):
        user_data = self.profiler.df[self.profiler.df['user_id'] == user_id]

        if user_data.empty or len(user_data) < 5:  # Ensure enough data points
            return None, None

        # Get user profile
        user_profile = self.profiler.create_user_profile(user_id)
        profile_features = [
            user_profile['average_login_hour'],
            user_profile['login_hour_std_dev'],
            user_profile['device_changes'],
            user_profile['total_ios_actions'],
            user_profile['total_android_actions'],
            user_profile['denial_rate'],
            user_profile['average_session_duration'],
            user_profile['session_duration_std_dev']
        ]

        # Extract relevant features for each action
        action_features = user_data[
            ['hour_of_timestamp', 'phone_versions', 'iOS sum', 'Android sum', 'is_denied', 'session_duration']].copy()

        # Convert categorical 'phone_versions' to numerical using factorize
        action_features['phone_versions'] = pd.factorize(action_features['phone_versions'])[0]

        # Repeat profile features to match the number of action rows
        profile_features_repeated = np.tile(profile_features, (len(action_features), 1))

        # Combine action features with profile features
        combined_features = np.hstack((action_features, profile_features_repeated))

        # Replace NaNs in numerical columns with 0
        combined_features = pd.DataFrame(combined_features).fillna(0).values

        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(combined_features)

        # Train LOF model
        lof = LocalOutlierFactor(n_neighbors=5, contamination=0.1)
        lof.fit(scaled_features)

        # Save the model and scaler for the user
        self.user_models[user_id] = lof
        self.user_scalers[user_id] = scaler

    def train_all_users(self):
        user_ids = self.profiler.df['user_id'].unique()
        for user_id in user_ids:
            self.train_user_model(user_id)

    def predict_user_action(self, user_id, action_features):
        if user_id not in self.user_models:
            raise ValueError(f"No model found for user_id: {user_id}")

        # Get user profile
        user_profile = self.profiler.create_user_profile(user_id)
        profile_features = [
            user_profile['average_login_hour'],
            user_profile['login_hour_std_dev'],
            user_profile['device_changes'],
            user_profile['total_ios_actions'],
            user_profile['total_android_actions'],
            user_profile['denial_rate'],
            user_profile['average_session_duration'],
            user_profile['session_duration_std_dev']
        ]

        # Combine action features with profile features
        combined_features = action_features + profile_features

        # Scale action features
        scaler = self.user_scalers[user_id]
        scaled_features = scaler.transform([combined_features])

        # Predict using LOF model
        lof = self.user_models[user_id]
        prediction = lof.predict(scaled_features)

        # Return 'normal' for 1 and 'anomaly' for -1
        return 'normal' if prediction[0] == 1 else 'anomaly'


if __name__ == "__main__":
    preprocessed_file_path = 'csv_dir/jerusalem_location_15.csv'
    preprocessed_df = preprocess_data.Preprocessor(preprocessed_file_path).preprocess()

    # Initialize and train LOF models
    lof_model = LOFModel(preprocessed_df)
    lof_model.train_all_users()

    # Example: Predicting a new action for a user
    example_user_id = 'aca17b2f-0840-4e47-a24a-66d47f9f16d7'
    example_action_features = [
        15,  # average_login_hour
        2,  # login_hour_std_dev
        3,  # device_changes
        5,  # total_ios_actions
        10,  # total_android_actions
        0.1,  # denial_rate
        300,  # average_session_duration
        50  # session_duration_std_dev
    ]
    prediction = lof_model.predict_user_action(example_user_id, example_action_features)
    print(f"Prediction for new action: {prediction}")
