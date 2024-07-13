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
from anomaly_visualizer import AnomalyVisualizer
import pickle
import os


class LOFModel:
    """
    Local Outlier Factor (LOF) model for detecting anomalies in user behavior.

    This class implements a multi-threaded LOF model that trains on individual features
    for each user and provides methods for prediction and evaluation.
    """

    def __init__(self, preprocessed_df, min_samples=5, max_workers=None, save_models=False, overwrite_models=False):
        """
        Initialize the LOF model.

        Args:
            preprocessed_df (pd.DataFrame): Preprocessed dataframe containing user data.
            min_samples (int): Minimum number of samples required to train a user model.
            max_workers (int): Maximum number of threads to use for parallel processing.
            save_models (bool): Whether to save trained models to disk.

        Attributes:
            profiler (UserProfiler): User profiling object.
            user_models (dict): Stores LOF models for each user and feature.
            user_scalers (dict): Stores scalers for each user and feature.
            user_test_data (dict): Stores test data for each user.
            categorical_columns (dict): Stores categorical column names for each user.
            features (list): List of features used in the model.
            feature_weights (dict): Weights assigned to each feature for anomaly scoring.
            feature_thresholds (dict): Thresholds for each feature to determine anomalies.
            save_models (bool): Whether to save trained models to disk.
        """
        self.profiler = UserProfiler(preprocessed_df)
        self.user_models = {}
        self.user_scalers = {}
        self.user_test_data = {}
        self.categorical_columns = {}
        self.min_samples = min_samples
        self.max_workers = max_workers
        self.save_models = save_models
        self.overwrite_models = overwrite_models
        self.lock = threading.Lock()
        self.features = ['hour_of_timestamp', 'phone_versions', 'iOS sum', 'Android sum', 'is_denied',
                         'session_duration', 'location_or_ip']
        self.feature_weights = {
            'hour_of_timestamp': 0.2,
            'phone_versions': 0.2,
            'iOS sum': 0.1,
            'Android sum': 0.1,
            'is_denied': 0.2,
            'session_duration': 0.2,
            'location_or_ip': 0.3
        }
        self.feature_thresholds = {feature: -1.5 for feature in self.features}
        self.visualizer = AnomalyVisualizer()

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
            # print(f"\nInsufficient data for user {user_id}. Skipping.") # Skip users with insufficient data
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

        if self.save_models:
            self.save_user_model(user_id)

    def train_all_users(self):
        """
        Train LOF models for all users in parallel.

        This method uses a ThreadPoolExecutor to train models for multiple users concurrently.
        """
        user_ids = list(self.profiler.df['user_id'].unique())
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            list(tqdm(executor.map(self.train_user_model, user_ids), total=len(user_ids), desc="Training users"))

    def save_user_model(self, user_id):
        """
        Save a trained user model to disk.

        Args:
            user_id (str): The ID of the user whose model to save.
        """
        if not os.path.exists('lof_trained_models'):
            os.makedirs('lof_trained_models')

        model_path = f'lof_trained_models/{user_id}_model.pkl'

        if not self.overwrite_models and os.path.exists(model_path):
            print(f"Model for user {user_id} already exists. Skipping save.")
            return

        with open(model_path, 'wb') as f:
            pickle.dump({
                'models': self.user_models[user_id],
                'scalers': self.user_scalers[user_id],
                'categorical_columns': self.categorical_columns.get(user_id, {}),
                'test_data': self.user_test_data.get(user_id, None)
            }, f)

    def load_user_model(self, user_id):
        """
        Load a trained user model from disk.

        Args:
            user_id (str): The ID of the user whose model to load.

        Returns:
            bool: True if the model was successfully loaded, False otherwise.
        """
        model_path = f'lof_trained_models/{user_id}_model.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.user_models[user_id] = data['models']
                self.user_scalers[user_id] = data['scalers']
                self.categorical_columns[user_id] = data['categorical_columns']
                self.user_test_data[user_id] = data['test_data']
            return True
        return False

    def train_or_load_all_users(self):
        """
        Train or load models for all users, depending on availability of saved models.
        """
        user_ids = list(self.profiler.df['user_id'].unique())
        for user_id in tqdm(user_ids, desc="Training/Loading users"):
            if self.overwrite_models or not self.load_user_model(user_id):
                self.train_user_model(user_id)

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
        all_user_data = []
        user_ids = list(self.user_test_data.keys())

        for user_id in tqdm(user_ids, desc="Evaluating users"):
            user_data, normal_data, anomalous_data = self.analyze_user(user_id, print_results=False)

            user_result = {
                "valid": len(normal_data),
                "need_second_check": len(anomalous_data[anomalous_data['prediction'] == 'need_second_check']),
                "invalid": len(anomalous_data[anomalous_data['prediction'] == 'invalid'])
            }

            user_results[user_id] = user_result
            all_user_data.append(user_data)

            for key in results:
                results[key] += user_result[key]

        all_user_data = pd.concat(all_user_data, ignore_index=True)

        total = sum(results.values())
        evaluation_content = "Evaluation Results:\n"
        for category, count in results.items():
            percentage = (count / total) * 100 if total > 0 else 0
            evaluation_content += f"{category.capitalize()}: {count} ({percentage:.2f}%)\n"

        print(self.visualizer.create_boxed_output(evaluation_content.strip(), "Model Evaluation"))

        user_profiles = {user_id: self.profiler.create_user_profile(user_id) for user_id in user_ids}
        self.display_random_anomalies(user_results, user_profiles, all_user_data)

    def evaluate_user(self, user_id):
        """
        Evaluate the model on a single user's test data.

        Args:
            user_id (str): The ID of the user to evaluate.

        Returns:
            dict: A dictionary containing counts of valid, need_second_check, and invalid predictions.
        """
        user_results = {"valid": 0, "need_second_check": 0, "invalid": 0}
        test_data = self.user_test_data[user_id].copy()
        predictions = []

        for _, row in test_data.iterrows():
            action_features = {feature: row[feature] for feature in self.features}
            try:
                prediction = self.predict_user_action(user_id, action_features)
                user_results[prediction] += 1
                predictions.append(prediction)
            except Exception as e:
                print(f"Error in prediction for user {user_id}: {e}")
                predictions.append("error")

        test_data['prediction'] = predictions
        return user_results, test_data

    def display_random_anomalies(self, user_results, user_profiles, user_data):
        """
        Display random anomalies from the 'need_second_check' and 'invalid' categories.

        This method selects up to 3 random users from each category and displays their
        profiles along with up to 3 random anomalous actions.

        Args:
            user_results (dict): Dictionary containing evaluation results for each user.
        """
        need_second_check_users = [user for user, results in user_results.items() if results['need_second_check'] > 0]
        invalid_users = [user for user, results in user_results.items() if results['invalid'] > 0]

        for category, users in [('need_second_check', need_second_check_users), ('invalid', invalid_users)]:
            if users:
                print(self.visualizer.create_boxed_output(f"Displaying up to 3 random users from {category} category:",
                                                          "Random Users Sample"))
                for user in np.random.choice(users, min(3, len(users)), replace=False):
                    content = f"User ID: {user}\n\n"

                    # Add user profile
                    content += "User Profile:\n"
                    profile = user_profiles[user]
                    for key, value in profile.items():
                        content += f"  {key}: {value}\n"
                    content += "\n"

                    # Add anomalous actions
                    content += f"Anomalous Actions ({category}):\n"
                    user_anomalies = user_data[(user_data['user_id'] == user) & (user_data['prediction'] == category)]
                    for _, action in user_anomalies.iterrows():
                        content += "  Action:\n"
                        for feature in self.features:
                            content += f"    {feature}: {action[feature]}\n"
                        content += "\n"

                    print(self.visualizer.create_boxed_output(content.strip()))
            else:
                print(self.visualizer.create_boxed_output(f"No users found in {category} category",
                                                          "Random Users Sample"))

    def visualize_anomalies(self, user_id):
        """
        Visualize anomalies for a specific user.

        Args:
            user_id (str): The ID of the user to visualize anomalies for.
        """
        if user_id not in self.user_models:
            print(f"No model found for user {user_id}")
            return

        user_data = self.profiler.df[self.profiler.df['user_id'] == user_id]
        self.visualizer.visualize_user_anomalies(user_id, user_data, self.features)

    def analyze_user(self, user_id, print_results=False):
        """
        Perform a comprehensive analysis of a user's behavior and anomalies.

        This method visualizes the user's data, predicts anomalies, compares normal and anomalous data,
        visualizes categorical features, and prints statistics about the user's actions and anomalies.

        Args:
            user_id (str): The ID of the user to analyze.
        """
        if user_id not in self.user_models:
            print(f"No model found for user {user_id}")
            return

        user_data = self.profiler.df[self.profiler.df['user_id'] == user_id].copy()

        # Predict anomalies
        predictions = []
        for _, row in user_data.iterrows():
            action_features = {feature: row[feature] for feature in self.features}
            prediction = self.predict_user_action(user_id, action_features)
            predictions.append(prediction)

        user_data['prediction'] = predictions

        # Prepare the content for the boxed output
        normal_data = user_data[user_data['prediction'] == 'valid']
        anomalous_data = user_data[user_data['prediction'].isin(['need_second_check', 'invalid'])]

        breakdown = user_data['prediction'].value_counts()

        if print_results:
            content = f"""Analysis for User {user_id}:
        Total actions: {len(user_data)}
        Valid actions: {len(normal_data)}
        Anomalous actions: {len(anomalous_data)}
        Anomaly breakdown:
        {breakdown.to_string()}"""

            # Print the boxed output
            print(self.visualizer.create_boxed_output(content, "User Analysis Summary"))

            # Visualize feature distributions
            self.visualizer.visualize_feature_distributions(user_id, user_data)

        return user_data, normal_data, anomalous_data


if __name__ == "__main__":
    import preprocess_data

    preprocessed_file_path = 'csv_dir/jerusalem_location_15.csv'
    preprocessed_df = preprocess_data.Preprocessor(preprocessed_file_path).preprocess()

    lof_model = LOFModel(preprocessed_df, max_workers=10, save_models=True, overwrite_models=False)
    lof_model.train_or_load_all_users()

    # Example usage
    example_user_id = 'aca17b2f-0840-4e47-a24a-66d47f9f16d7'
    example_action_features = {
        'hour_of_timestamp': 15,
        'phone_versions': 'iPhone14_5',
        'iOS sum': 1,
        'Android sum': 0,
        'is_denied': 0,
        'session_duration': 300,
        'location_or_ip': 'Jerusalem, Israel (212.117.143.250)'
    }
    # prediction = lof_model.predict_user_action(example_user_id, example_action_features)
    # print(f"Prediction for new action: {prediction}")

    # Evaluate the model
    lof_model.evaluate_model()

    # # Visualize anomalies for the example user
    # lof_model.visualize_anomalies(example_user_id)

    # Perform comprehensive analysis for the example user
    lof_model.analyze_user(example_user_id, print_results=True)

    # You can add more users to analyze here
    # other_user_id = 'another-user-id'
    # lof_model.analyze_user(other_user_id)
