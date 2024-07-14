import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from user_profiling import UserProfiler
import concurrent.futures
import threading
from tqdm import tqdm
import pickle
import os
from anomaly_visualizer import AnomalyVisualizer


class IsolationForestModel:
    def __init__(self, preprocessed_df, n_estimators=100, contamination=0.1, min_samples=5, max_workers=None,
                 save_models=False, overwrite_models=False):
        self.profiler = UserProfiler(preprocessed_df)
        self.user_models = {}
        self.user_scalers = {}
        self.user_test_data = {}
        self.categorical_columns = {}
        self.n_estimators = n_estimators
        self.contamination = contamination
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
        user_data = self.profiler.df[self.profiler.df['user_id'] == user_id]

        if len(user_data) < self.min_samples:
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

            iso_forest = IsolationForest(n_estimators=self.n_estimators, contamination=self.contamination,
                                         random_state=42)
            iso_forest.fit(scaled_feature)

            user_models[feature] = iso_forest
            user_scalers[feature] = scaler

        _, test_data = train_test_split(user_data, test_size=0.2, random_state=42)

        with self.lock:
            self.user_models[user_id] = user_models
            self.user_scalers[user_id] = user_scalers
            self.categorical_columns[user_id] = categorical_columns
            self.user_test_data[user_id] = test_data

        if self.save_models:
            self.save_user_model(user_id)

    def train_all_users(self):
        user_ids = list(self.profiler.df['user_id'].unique())
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            list(tqdm(executor.map(self.train_user_model, user_ids), total=len(user_ids), desc="Training users"))

    def save_user_model(self, user_id):
        if not os.path.exists('isolation_forest_trained_models'):
            os.makedirs('isolation_forest_trained_models')

        model_path = f'isolation_forest_trained_models/{user_id}_model.pkl'

        if not self.overwrite_models and os.path.exists(model_path):
            return

        with open(model_path, 'wb') as f:
            pickle.dump({
                'models': self.user_models[user_id],
                'scalers': self.user_scalers[user_id],
                'categorical_columns': self.categorical_columns.get(user_id, {}),
                'test_data': self.user_test_data.get(user_id, None)
            }, f)

    def load_user_model(self, user_id):
        model_path = f'isolation_forest_trained_models/{user_id}_model.pkl'
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
        user_ids = list(self.profiler.df['user_id'].unique())
        for user_id in tqdm(user_ids, desc="Training/Loading users"):
            if self.overwrite_models or not self.load_user_model(user_id):
                self.train_user_model(user_id)

    def predict_user_action(self, user_id, action_features):
        if user_id not in self.user_models:
            raise ValueError(f"No model found for user_id: {user_id}")

        anomaly_scores = []
        for feature in self.features:
            if feature not in action_features:
                continue

            value = action_features[feature]
            iso_forest = self.user_models[user_id][feature]
            scaler = self.user_scalers[user_id][feature]

            if feature in ['phone_versions', 'location_or_ip']:
                feature_data = pd.get_dummies(pd.Series([value]), prefix=feature)
                for col in self.categorical_columns[user_id][feature]:
                    if col not in feature_data.columns:
                        feature_data[col] = 0
                feature_data = feature_data.reindex(columns=self.categorical_columns[user_id][feature], fill_value=0)
            else:
                feature_data = pd.DataFrame([[value]], columns=[feature])

            scaled_feature = scaler.transform(feature_data)
            anomaly_score = iso_forest.score_samples(scaled_feature)[0]
            anomaly_score = max(anomaly_score, -6)  # Cap the anomaly score

            profile_weight = self.calculate_profile_weight(user_id, feature, value)
            weighted_score = anomaly_score * profile_weight * self.feature_weights[feature]
            anomaly_scores.append((feature, weighted_score))

        return self.aggregate_scores(anomaly_scores)

    def calculate_profile_weight(self, user_id, feature, value):
        profile = self.profiler.create_user_profile(user_id)
        if feature in profile:
            profile_value = profile[feature]
            deviation = abs(profile_value - value)
            return 1 / (1 + deviation)
        return 1.0

    def aggregate_scores(self, anomaly_scores):
        total_weighted_score = sum(score for _, score in anomaly_scores)
        threshold_inbetween = -2.5
        threshold_invalid = -3.0

        if total_weighted_score <= threshold_invalid:
            return "invalid"
        elif threshold_invalid < total_weighted_score <= threshold_inbetween:
            return "need_second_check"
        else:
            return "valid"

    def evaluate_model(self):
        results = {"valid": 0, "need_second_check": 0, "invalid": 0}
        user_results = {}
        all_user_data = []
        user_ids = list(self.user_models.keys())

        for user_id in tqdm(user_ids, desc="Evaluating users"):
            user_data, normal_data, anomalous_data = self.analyze_user(user_id, print_results=False)
            if user_data is not None:
                user_result = {
                    "valid": len(normal_data),
                    "need_second_check": len(anomalous_data[anomalous_data['prediction'] == 'need_second_check']),
                    "invalid": len(anomalous_data[anomalous_data['prediction'] == 'invalid'])
                }

                user_results[user_id] = user_result
                all_user_data.append(user_data)

                for key in results:
                    results[key] += user_result[key]

        if all_user_data:
            all_user_data = pd.concat(all_user_data, ignore_index=True)

        self.visualizer.print_evaluation_results(results)

        user_profiles = {user_id: self.profiler.create_user_profile(user_id) for user_id in user_ids}
        self.visualizer.print_random_users(user_results)

    def analyze_user(self, user_id, print_results=False):
        if user_id not in self.user_models:
            return None, None, None

        user_data = self.profiler.df[self.profiler.df['user_id'] == user_id].copy()

        predictions = []
        for _, row in user_data.iterrows():
            action_features = {feature: row[feature] for feature in self.features}
            prediction = self.predict_user_action(user_id, action_features)
            predictions.append(prediction)

        user_data['prediction'] = predictions

        normal_data = user_data[user_data['prediction'] == 'valid']
        anomalous_data = user_data[user_data['prediction'].isin(['need_second_check', 'invalid'])]

        if print_results:
            self.visualizer.visualize_user_anomalies(user_id, user_data, self.features)
            self.visualizer.visualize_anomaly_comparison(user_id, normal_data, anomalous_data)

        return user_data, normal_data, anomalous_data


if __name__ == "__main__":
    import preprocess_data

    preprocessed_file_path = 'csv_dir/jerusalem_location_15.csv'
    preprocessed_df = preprocess_data.Preprocessor(preprocessed_file_path).preprocess()

    iso_forest_model = IsolationForestModel(preprocessed_df, max_workers=10, save_models=True, overwrite_models=False)
    iso_forest_model.train_or_load_all_users()

    # Evaluate the model
    iso_forest_model.evaluate_model()

    # Example usage for a single user
    example_user_id = 'aca17b2f-0840-4e47-a24a-66d47f9f16d7'
    iso_forest_model.analyze_user(example_user_id, print_results=True)

    # Debug: Print number of users and their data
    print(f"Total number of users: {len(iso_forest_model.user_models)}")
    for user_id, models in iso_forest_model.user_models.items():
        print(f"User {user_id}: {len(models)} models")
