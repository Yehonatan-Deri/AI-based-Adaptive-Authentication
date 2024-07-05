# lof_model.py

import pandas as pd
import preprocess_data
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from user_profiling import UserProfiler


class LOFModel:
    def __init__(self, preprocessed_df, min_samples=5):
        self.profiler = UserProfiler(preprocessed_df)
        self.user_models = {}
        self.user_scalers = {}
        self.min_samples = min_samples  # Minimum samples required for training

    def train_user_model(self, user_id):
        user_data = self.profiler.df[self.profiler.df['user_id'] == user_id]

        if user_data.empty or len(user_data) < self.min_samples:  # Ensure enough data points
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

        # Combine action features with profile features
        combined_features = np.hstack((action_features, np.tile(profile_features, (len(action_features), 1))))

        # Convert combined features to DataFrame and handle NaNs
        combined_df = pd.DataFrame(combined_features, columns=[
            'hour_of_timestamp', 'phone_versions', 'iOS sum', 'Android sum', 'is_denied', 'session_duration',
            'average_login_hour', 'login_hour_std_dev', 'device_changes',
            'total_ios_actions', 'total_android_actions', 'denial_rate',
            'average_session_duration', 'session_duration_std_dev'
        ])
        combined_df.fillna(0, inplace=True)
        combined_df = combined_df.infer_objects(copy=False)
        combined_features = combined_df.values

        # Debugging print statements
        print(f"Training user: {user_id}")
        print(f"Combined features shape (training): {combined_features.shape}")
        print(f"Feature names (training): {combined_df.columns.tolist()}")

        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(combined_features)

        # Train LOF model
        lof = LocalOutlierFactor(n_neighbors=min(5, len(user_data) - 1), contamination=0.1)
        lof.fit(scaled_features)

        # Save the model and scaler for the user
        self.user_models[user_id] = lof
        self.user_scalers[user_id] = scaler

    def train_all_users(self):
        user_ids = self.profiler.df['user_id'].unique()
        for user_id in user_ids:
            self.train_user_model(user_id)

    def predict_user_action(self, user_id, action_features, threshold_inbetween=-1.5, threshold_invalid=-3.0):
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
        combined_features = np.hstack(
            (np.array(action_features).reshape(1, -1), np.array(profile_features).reshape(1, -1)))

        # Convert combined features to DataFrame and handle NaNs
        combined_df = pd.DataFrame(combined_features, columns=[
            'hour_of_timestamp', 'phone_versions', 'iOS sum', 'Android sum', 'is_denied', 'session_duration',
            'average_login_hour', 'login_hour_std_dev', 'device_changes',
            'total_ios_actions', 'total_android_actions', 'denial_rate',
            'average_session_duration', 'session_duration_std_dev'
        ])
        combined_df.fillna(0, inplace=True)
        combined_df = combined_df.infer_objects(copy=False)
        combined_features = combined_df.values

        # Debugging print statements
        print(f"Predicting for user: {user_id}")
        print(f"Combined features shape (prediction): {combined_features.shape}")
        print(f"Feature names (prediction): {combined_df.columns.tolist()}")

        # Get existing user data
        user_data = self.profiler.df[self.profiler.df['user_id'] == user_id]
        existing_features = user_data[
            ['hour_of_timestamp', 'phone_versions', 'iOS sum', 'Android sum', 'is_denied', 'session_duration']].copy()
        existing_features['phone_versions'] = pd.factorize(existing_features['phone_versions'])[0]
        existing_features = existing_features.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Combine existing features with the new login features
        combined_existing_features = np.hstack(
            (existing_features, np.tile(profile_features, (len(existing_features), 1))))
        combined_existing_features = np.vstack((combined_existing_features, combined_features))

        # Scale the combined features
        scaler = self.user_scalers[user_id]
        combined_existing_features_scaled = scaler.fit_transform(combined_existing_features)

        # Fit the model on the combined data
        lof = LocalOutlierFactor(n_neighbors=min(5, len(combined_existing_features_scaled) - 1), contamination=0.1)
        lof.fit(combined_existing_features_scaled)

        # Get the LOF score for the new login
        lof_score = lof.negative_outlier_factor_[-1]

        # Determine the category based on thresholds
        if lof_score > threshold_inbetween:
            return "valid"
        elif threshold_invalid < lof_score <= threshold_inbetween:
            return "need_second_check"
        else:
            return "invalid"

if __name__ == "__main__":
    preprocessed_file_path = 'csv_dir/jerusalem_location_15.csv'
    preprocessed_df = preprocess_data.Preprocessor(preprocessed_file_path).preprocess()

    # Initialize and train LOF models
    lof_model = LOFModel(preprocessed_df)
    lof_model.train_user_model('aca17b2f-0840-4e47-a24a-66d47f9f16d7')
    # lof_model.train_all_users()

    # Example: Predicting a new action for a user
    example_user_id = 'aca17b2f-0840-4e47-a24a-66d47f9f16d7'
    example_action_features = [
        15,  # hour_of_timestamp
        1,  # phone_versions (factorized value)
        1,  # iOS sum
        0,  # Android sum
        0,  # is_denied
        300  # session_duration
    ]
    prediction = lof_model.predict_user_action(example_user_id, example_action_features)
    print(f"Prediction for new action: {prediction}")
