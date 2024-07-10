import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from user_profiling import UserProfiler
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures
import threading
from tqdm import tqdm
import random


class LOFModel:
    """
    Local Outlier Factor (LOF) model for detecting anomalies in user behavior.

    This class implements a multi-threaded LOF model that trains on individual features
    for each user and provides methods for prediction and evaluation.
    """

    def __init__(self, preprocessed_df, min_samples=5, max_workers=None):
        """
        Initialize the LOF model.

        Args:
            preprocessed_df (pd.DataFrame): Preprocessed dataframe containing user data.
            min_samples (int): Minimum number of samples required to train a user model.
            max_workers (int): Maximum number of threads to use for parallel processing.

        Attributes:
            profiler (UserProfiler): User profiling object.
            user_models (dict): Stores LOF models for each user and feature.
            user_scalers (dict): Stores scalers for each user and feature.
            user_test_data (dict): Stores test data for each user.
            categorical_columns (dict): Stores categorical column names for each user.
            features (list): List of features used in the model.
            feature_weights (dict): Weights assigned to each feature for anomaly scoring.
            feature_thresholds (dict): Thresholds for each feature to determine anomalies.
        """
        self.profiler = UserProfiler(preprocessed_df)
        self.user_models = {}
        self.user_scalers = {}
        self.user_test_data = {}
        self.categorical_columns = {}
        self.min_samples = min_samples
        self.max_workers = max_workers
        self.lock = threading.Lock()
        self.features = ['hour_of_timestamp', 'phone_versions', 'iOS sum', 'Android sum', 'is_denied',
                         'session_duration', 'location_or_ip']
        self.feature_weights = {
            'hour_of_timestamp': 0.2,
            'phone_versions': 0.2,
            'iOS sum': 0.1,
            'Android sum': 0.1,
            'is_denied': 0.2,
            'session_duration': 0.1,
            'location_or_ip': 0.3
        }
        self.feature_thresholds = {feature: -1.5 for feature in self.features}

    def train_user_model(self, user_id):
        user_data = self.profiler.df[self.profiler.df['user_id'] == user_id]

        if user_data.empty or len(user_data) < self.min_samples:  # Ensure enough data points
            return None, None

        # Initialize dictionaries to store models and scalers for each feature
        self.user_models[user_id] = {}
        self.user_scalers[user_id] = {}

        features_to_train = [
            'hour_of_timestamp', 'phone_versions', 'iOS sum', 'Android sum', 'is_denied', 'session_duration'
        ]

        for feature in features_to_train:
            feature_data = user_data[[feature]].copy()
            feature_data = feature_data.apply(pd.to_numeric, errors='coerce').fillna(0)

            scaler = StandardScaler()
            scaled_feature = scaler.fit_transform(feature_data)

            lof = LocalOutlierFactor(n_neighbors=min(5, len(user_data) - 1), contamination=0.1)
            lof.fit(scaled_feature)

            self.user_models[user_id][feature] = lof
            self.user_scalers[user_id][feature] = scaler

    def train_all_users(self):
        user_ids = self.profiler.df['user_id'].unique()
        for user_id in user_ids:
            self.train_user_model(user_id)

    def predict_user_action(self, user_id, action_features, threshold_inbetween=-2.5, threshold_invalid=-3.0):
        if user_id not in self.user_models:
            raise ValueError(f"No model found for user_id: {user_id}")

        lof_scores = []
        for feature, value in action_features.items():
            if feature not in self.user_models[user_id]:
                continue

            lof = self.user_models[user_id][feature]
            scaler = self.user_scalers[user_id][feature]

            # Get existing user data
            user_data = self.profiler.df[self.profiler.df['user_id'] == user_id]
            existing_features = user_data[[feature]].copy()
            existing_features = existing_features.apply(pd.to_numeric, errors='coerce').fillna(0)

            # Combine existing features with the new login feature
            combined_features = pd.concat([existing_features, pd.DataFrame([[value]], columns=[feature])])

            # Scale the combined features
            combined_features_scaled = scaler.transform(combined_features)

            # Get the LOF score for the new login
            lof.fit(combined_features_scaled)
            lof_score = lof.negative_outlier_factor_[-1]

            # Cap the LOF score at -10 if it's lower than -10
            lof_score = max(lof_score, -6)

            # Debugging print statements
            print(f"Feature: {feature}, Value: {value}, LOF Score: {lof_score}")

            # Apply profile weight adjustment (if applicable)
            if feature in self.profiler.create_user_profile(user_id):
                profile_value = self.profiler.create_user_profile(user_id)[feature]
                profile_weight = self.calculate_profile_weight(profile_value, value)
                weighted_score = lof_score * profile_weight
            else:
                weighted_score = lof_score

            lof_scores.append((feature, weighted_score))

        return self.aggregate_scores(lof_scores, threshold_inbetween, threshold_invalid)

    def calculate_profile_weight(self, profile_value, value):
        # Define logic to calculate weight based on profile value and current value
        # For simplicity, we can use an inverse relationship, where more deviation means a lower weight
        deviation = abs(profile_value - value)
        if deviation == 0:
            return 1.0
        return 1 / (1 + deviation)

    def aggregate_scores(self, lof_scores, threshold_inbetween, threshold_invalid):
        total_weighted_score = 0

        for feature, score in lof_scores:
            threshold = self.feature_thresholds[feature]
            weight = self.feature_weights[feature]

            total_weighted_score += score * weight

        print(f"Total Weighted Score: {total_weighted_score}")

        if total_weighted_score <= threshold_invalid:
            return "invalid"
        elif threshold_invalid < total_weighted_score <= threshold_inbetween:
            return "need_second_check"
        else:
            return "valid"


if __name__ == "__main__":
    preprocessed_file_path = 'csv_dir/jerusalem_location_15.csv'
    preprocessed_df = preprocess_data.Preprocessor(preprocessed_file_path).preprocess()

    # Initialize and train LOF models
    lof_model = LOFModel(preprocessed_df)
    # lof_model.train_all_users()
    lof_model.train_user_model('aca17b2f-0840-4e47-a24a-66d47f9f16d7')

    # Initialize the GraphGenerator
    graph_generator = GraphGenerator(lof_model.profiler)

    # Example: Predicting a new action for a user
    example_user_id = 'aca17b2f-0840-4e47-a24a-66d47f9f16d7'
    example_action_features = {
        'hour_of_timestamp': 15,
        'phone_versions': 1,  # factorized value
        'iOS sum': 1,
        'Android sum': 0,
        'is_denied': 0,
        'session_duration': 300
    }
    prediction = lof_model.predict_user_action(example_user_id, example_action_features)
    print(f"Prediction for new action: {prediction}")

    # Valid Action Example
    valid_action_features = {
        'hour_of_timestamp': 13,  # close to average
        'phone_versions': 0,  # iPhone14_5 factorized as 0
        'iOS sum': 1,
        'Android sum': 0,
        'is_denied': 0,
        'session_duration': 7  # close to average
    }
    prediction_valid = lof_model.predict_user_action(example_user_id, valid_action_features)
    print(f"Valid action prediction: {prediction_valid}")

    # Need Second Check Example
    second_check_action_features = {
        'hour_of_timestamp': 15,  # slightly outside average
        'phone_versions': 1,  # new device, factorized as 1
        'iOS sum': 1,
        'Android sum': 0,
        'is_denied': 0,
        'session_duration': 10  # slightly outside average
    }
    prediction_second_check = lof_model.predict_user_action(example_user_id, second_check_action_features)
    print(f"Second check action prediction: {prediction_second_check}")

    # Invalid Action Example
    invalid_action_features = {
        'hour_of_timestamp': 22,  # significantly outside average
        'phone_versions': 2,  # new device, factorized as 2
        'iOS sum': 1,
        'Android sum': 0,
        'is_denied': 0,
        'session_duration': 20  # significantly outside average
    }
    prediction_invalid = lof_model.predict_user_action(example_user_id, invalid_action_features)
    print(f"Invalid action prediction: {prediction_invalid}")

    # Generate all plots for the example user
    graph_generator.generate_all_plots(example_user_id, lof_model, prediction)