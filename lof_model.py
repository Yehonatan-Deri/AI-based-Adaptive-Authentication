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
        """
        Train LOF models for a single user.

        This method trains separate LOF models for each feature of a user's data.

        Args:
            user_id (str): The ID of the user to train models for.

        Note:
            This method skips users with insufficient data (less than min_samples).
        """
        user_data = self.profiler.df[self.profiler.df['user_id'] == user_id]

        if len(user_data) < self.min_samples:
            print(f"\nInsufficient data for user {user_id}. Skipping.")
            return

        user_models = {}
        user_scalers = {}
        categorical_columns = {}

        for feature in self.features:
            feature_data = user_data[[feature]].copy()

            if feature in ['phone_versions', 'location_or_ip']:
                feature_data = pd.get_dummies(feature_data, prefix=feature)
                categorical_columns[feature] = feature_data.columns.tolist()
            else:
                feature_data = feature_data.apply(pd.to_numeric, errors='coerce').fillna(0)

            scaler = StandardScaler()
            scaled_feature = scaler.fit_transform(feature_data)

            lof = LocalOutlierFactor(n_neighbors=min(5, len(user_data) - 1), contamination=0.1, novelty=True)
            lof.fit(scaled_feature)

            user_models[feature] = lof
            user_scalers[feature] = scaler

        # Split data for testing
        _, test_data = train_test_split(user_data, test_size=0.2, random_state=42)

        with self.lock:
            self.user_models[user_id] = user_models
            self.user_scalers[user_id] = user_scalers
            self.categorical_columns[user_id] = categorical_columns
            self.user_test_data[user_id] = test_data

    def train_all_users(self):
        """
        Train LOF models for all users in parallel.

        This method uses a ThreadPoolExecutor to train models for multiple users concurrently.
        """
        user_ids = list(self.profiler.df['user_id'].unique())
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            list(tqdm(executor.map(self.train_user_model, user_ids), total=len(user_ids), desc="Training users"))

    def predict_user_action(self, user_id, action_features, threshold_inbetween=-2.5, threshold_invalid=-3.0):
        """
        Predict whether a user action is anomalous.

        This method calculates LOF scores for each feature of the action and aggregates them
        to determine if the action is valid, needs a second check, or is invalid.

        Args:
            user_id (str): The ID of the user.
            action_features (dict): A dictionary of feature names and their values for the action.
            threshold_inbetween (float): The threshold for classifying an action as needing a second check.
            threshold_invalid (float): The threshold for classifying an action as invalid.

        Returns:
            str: 'valid', 'need_second_check', or 'invalid'
        """
        if user_id not in self.user_models:
            raise ValueError(f"No model found for user_id: {user_id}")

        lof_scores = []
        for feature in self.features:
            if feature not in action_features:
                continue

            value = action_features[feature]
            lof = self.user_models[user_id][feature]
            scaler = self.user_scalers[user_id][feature]

            if feature in ['phone_versions', 'location_or_ip']:
                feature_data = pd.get_dummies(pd.Series([value]), prefix=feature)
                # Ensure all columns from training are present
                for col in self.categorical_columns[user_id][feature]:
                    if col not in feature_data.columns:
                        feature_data[col] = 0
                feature_data = feature_data.reindex(columns=self.categorical_columns[user_id][feature], fill_value=0)
            else:
                feature_data = pd.DataFrame([[value]], columns=[feature])

            scaled_feature = scaler.transform(feature_data)
            lof_score = lof.score_samples(scaled_feature)[0]
            lof_score = max(lof_score, -6)  # Cap the LOF score

            profile_weight = self.calculate_profile_weight(user_id, feature, value)
            weighted_score = lof_score * profile_weight * self.feature_weights[feature]
            lof_scores.append((feature, weighted_score))

        return self.aggregate_scores(lof_scores, threshold_inbetween, threshold_invalid)

    def calculate_profile_weight(self, user_id, feature, value):
        """
        Calculate a weight based on how much a feature value deviates from the user's profile.

        Args:
            user_id (str): The ID of the user.
            feature (str): The name of the feature.
            value: The value of the feature for the current action.

        Returns:
            float: A weight between 0 and 1, where 1 indicates no deviation from the profile.
        """
        profile = self.profiler.create_user_profile(user_id)
        if feature in profile:
            profile_value = profile[feature]
            deviation = abs(profile_value - value)
            return 1 / (1 + deviation)
        return 1.0

    def aggregate_scores(self, lof_scores, threshold_inbetween, threshold_invalid):
        """
        Aggregate LOF scores for all features and classify the action.

        Args:
            lof_scores (list): List of tuples containing (feature, score) pairs.
            threshold_inbetween (float): Threshold for 'need_second_check' classification.
            threshold_invalid (float): Threshold for 'invalid' classification.

        Returns:
            str: 'valid', 'need_second_check', or 'invalid'
        """
        total_weighted_score = sum(score for _, score in lof_scores)

        if total_weighted_score <= threshold_invalid:
            return "invalid"
        elif threshold_invalid < total_weighted_score <= threshold_inbetween:
            return "need_second_check"
        else:
            return "valid"

    def evaluate_model(self):
        """
        Evaluate the model on test data for all users.

        This method calculates the number of valid, need_second_check, and invalid predictions
        across all users' test data. It also displays random anomalies for further inspection.
        """
        results = {"valid": 0, "need_second_check": 0, "invalid": 0}
        user_results = {}
        user_ids = list(self.user_test_data.keys())

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_user = {executor.submit(self.evaluate_user, user_id): user_id for user_id in user_ids}
            for future in tqdm(concurrent.futures.as_completed(future_to_user), total=len(user_ids),
                               desc="Evaluating users"):
                user_id = future_to_user[future]
                user_result = future.result()
                user_results[user_id] = user_result
                with self.lock:
                    for key in results:
                        results[key] += user_result[key]

        total = sum(results.values())
        print("Evaluation Results:")
        for category, count in results.items():
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"{category.capitalize()}: {count} ({percentage:.2f}%)")

        self.display_random_anomalies(user_results)

    def evaluate_user(self, user_id):
        """
        Evaluate the model on a single user's test data.

        Args:
            user_id (str): The ID of the user to evaluate.

        Returns:
            dict: A dictionary containing counts of valid, need_second_check, and invalid predictions.
        """
        user_results = {"valid": 0, "need_second_check": 0, "invalid": 0}
        test_data = self.user_test_data[user_id]
        for _, row in test_data.iterrows():
            action_features = {feature: row[feature] for feature in self.features}
            try:
                prediction = self.predict_user_action(user_id, action_features)
                user_results[prediction] += 1
            except Exception as e:
                print(f"Error in prediction for user {user_id}: {e}")
        return user_results

    def display_random_anomalies(self, user_results):
        """
        Display random anomalies from the 'need_second_check' and 'invalid' categories.

        This method selects up to 3 random users from each category and displays their
        profiles along with up to 3 random anomalous actions.

        Args:
            user_results (dict): Dictionary containing evaluation results for each user.
        """
        need_second_check_users = [user for user, results in user_results.items() if results['need_second_check'] > 0]
        invalid_users = [user for user, results in user_results.items() if results['invalid'] > 0]

        for category in ['need_second_check', 'invalid']:
            users = need_second_check_users if category == 'need_second_check' else invalid_users
            print(f"\nDisplaying 3 random users from {category} category:")

            for user in random.sample(users, min(3, len(users))):
                print(f"\nUser ID: {user}")
                profile = self.profiler.create_user_profile(user)
                print("User Profile:")
                for key, value in profile.items():
                    print(f"  {key}: {value}")

                print("\nAnomalous Actions:")
                test_data = self.user_test_data[user]
                anomalous_actions = []
                for _, row in test_data.iterrows():
                    action_features = {feature: row[feature] for feature in self.features}
                    prediction = self.predict_user_action(user, action_features)
                    if prediction == category:
                        anomalous_actions.append(action_features)

                for i, action in enumerate(random.sample(anomalous_actions, min(3, len(anomalous_actions)))):
                    print(f"  Anomalous Action {i + 1}:")
                    for key, value in action.items():
                        print(f"    {key}: {value}")
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