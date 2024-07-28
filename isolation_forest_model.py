import warnings

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from user_profiling import UserProfiler
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures
import threading
from tqdm import tqdm
import pickle
import preprocess_data
import os
from anomaly_visualizer import AnomalyVisualizer
from sklearn.cluster import KMeans

class IsolationForestModel:
    def __init__(self, preprocessed_df, n_estimators=100, contamination=0.1, random_state=42, min_samples=2,
                 max_workers=None, save_models=True, overwrite_models=False, save_evaluations=True,
                 overwrite_evaluations=False):
        self.profiler = UserProfiler(preprocessed_df)
        self.user_models = {}
        self.user_scalers = {}
        self.user_test_data = {}
        self.categorical_columns = {}
        self.min_samples = min_samples
        self.max_workers = max_workers
        self.save_models = save_models
        self.overwrite_models = overwrite_models
        self.save_evaluations = save_evaluations
        self.overwrite_evaluations = overwrite_evaluations
        self.lock = threading.Lock()
        self.features = ['hour_of_timestamp', 'phone_versions', 'iOS sum', 'Android sum', 'is_denied',
                         'session_duration', 'location_or_ip']
        self.feature_weights = {
            'hour_of_timestamp': 0.3,
            'phone_versions': 0.2,
            'iOS sum': 0.1,
            'Android sum': 0.1,
            'is_denied': 0.4,
            'session_duration': 0.2,
            'location_or_ip': 0.3
        }
        self.feature_thresholds = {feature: -0.5 for feature in self.features}
        self.visualizer = AnomalyVisualizer()
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state

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

            iforest = IsolationForest(n_estimators=self.n_estimators, contamination=self.contamination,
                                      random_state=self.random_state)
            iforest.fit(scaled_feature)

            user_models[feature] = iforest
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
        model_path = f'isolation_forest_trained_models/{user_id}_model.pkl'
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.user_models[user_id] = data['models']
                    self.user_scalers[user_id] = data['scalers']
                    self.categorical_columns[user_id] = data['categorical_columns']
                    self.user_test_data[user_id] = data['test_data']
                # print(f"Loaded model for user {user_id}")
                return True
            except Exception as e:
                print(f"Error loading model for user {user_id}: {e}")
                return False
        return False

    def train_or_load_all_users(self):
        user_ids = list(self.profiler.df['user_id'].unique())
        for user_id in tqdm(user_ids, desc="Training/Loading users"):
            if not self.overwrite_models and self.load_user_model(user_id):
                continue  # If the model was successfully loaded, skip to the next user
            self.train_user_model(user_id)

    def predict_user_action(self, user_id, action_features, threshold_inbetween=-0.2, threshold_invalid=-0.5):
        if user_id not in self.user_models:
            raise ValueError(f"No model found for user_id: {user_id}")

        iforest_scores = []
        for feature in self.features:
            if feature not in action_features:
                continue

            value = action_features[feature]
            iforest = self.user_models[user_id][feature]
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
            iforest_score = iforest.decision_function(scaled_feature)[0]

            profile_weight = self.calculate_profile_weight(user_id, feature, value)
            weighted_score = iforest_score * profile_weight * self.feature_weights[feature]
            iforest_scores.append((feature, weighted_score))

        return self.aggregate_scores(iforest_scores, threshold_inbetween, threshold_invalid)

    def calculate_profile_weight(self, user_id, feature, value):
        profile = self.profiler.create_user_profile(user_id)
        if feature in profile:
            profile_value = profile[feature]
            deviation = abs(profile_value - value)
            return 1 / (1 + deviation)
        return 1.0

    def aggregate_scores(self, iforest_scores, threshold_inbetween, threshold_invalid):
        total_weighted_score = sum(score for _, score in iforest_scores)

        if total_weighted_score <= threshold_invalid:
            return "invalid"
        elif threshold_invalid < total_weighted_score <= threshold_inbetween:
            return "need_second_check"
        else:
            return "valid"

    def save_evaluation(self, user_id, evaluation_result):
        if not os.path.exists('isolation_forest_evaluations'):
            os.makedirs('isolation_forest_evaluations')

        eval_path = f'isolation_forest_evaluations/{user_id}_evaluation.pkl'

        if not self.overwrite_evaluations and os.path.exists(eval_path):
            print(f"Evaluation for user {user_id} already exists. Skipping save.")
            return

        with open(eval_path, 'wb') as f:
            pickle.dump(evaluation_result, f)

    def load_evaluation(self, user_id):
        eval_path = f'isolation_forest_evaluations/{user_id}_evaluation.pkl'
        if os.path.exists(eval_path):
            with open(eval_path, 'rb') as f:
                return pickle.load(f)
        return None

    def evaluate_model(self):
        results = {"valid": 0, "need_second_check": 0, "invalid": 0}
        user_results = {}
        all_user_data = []
        user_ids = list(self.user_test_data.keys())

        for user_id in tqdm(user_ids, desc="Evaluating users"):
            if self.save_evaluations and not self.overwrite_evaluations:
                loaded_evaluation = self.load_evaluation(user_id)
                if loaded_evaluation is not None:
                    user_data, normal_data, anomalous_data = loaded_evaluation
                else:
                    user_data, normal_data, anomalous_data = self.analyze_user(user_id, print_results=False)
                    if self.save_evaluations:
                        self.save_evaluation(user_id, (user_data, normal_data, anomalous_data))
            else:
                user_data, normal_data, anomalous_data = self.analyze_user(user_id, print_results=False)
                if self.save_evaluations:
                    self.save_evaluation(user_id, (user_data, normal_data, anomalous_data))

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

    def display_random_anomalies(self, user_results, user_profiles, user_data):
        need_second_check_users = [user for user, results in user_results.items() if results['need_second_check'] > 0]
        invalid_users = [user for user, results in user_results.items() if results['invalid'] > 0]

        for category, users in [('need_second_check', need_second_check_users), ('invalid', invalid_users)]:
            if users:
                print(self.visualizer.create_boxed_output(f"Displaying up to 3 random users from {category} category:",
                                                          "Random Users Sample"))
                for user in np.random.choice(users, min(3, len(users)), replace=False):
                    content = f"User ID: {user}\n\n"

                    content += "User Profile:\n"
                    profile = user_profiles[user]
                    for key, value in profile.items():
                        content += f"  {key}: {value}\n"
                    content += "\n"

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
        if user_id not in self.user_models:
            print(f"No model found for user {user_id}")
            return

        user_data = self.profiler.df[self.profiler.df['user_id'] == user_id]
        self.visualizer.visualize_user_anomalies(user_id, user_data, self.features)

    def analyze_user_patterns_with_kmeans(self, user_id, feature, n_clusters=2):
        """
        Analyze user patterns using k-means clustering and visualize the distribution.

        :param user_id: ID of the user to analyze
        :param feature: 'hour_of_timestamp' or 'session_duration'
        :param n_clusters: Number of clusters to use in k-means
        :return: Tuple containing cluster centers, potential anomalies, and silhouette score
        """
        # Suppress specific warnings
        warnings.filterwarnings("ignore", message="Blended transforms not yet supported.")

        user_data = self.profiler.df[self.profiler.df['user_id'] == user_id]

        # Extract the feature data
        X = user_data[feature].values.reshape(-1, 1)

        # Apply k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_.flatten()

        # Calculate silhouette score
        silhouette_avg = silhouette_score(X, labels)

        # Create the plot
        plt.figure(figsize=(12, 6))

        # Plot the distribution
        sns.histplot(X, kde=True, color="blue", alpha=0.5)

        # Plot the k-means centers
        for center in centers:
            plt.axvline(x=center, color='red', linestyle='--', label='Cluster Center')

        # Identify potential anomalies
        distances = np.min(np.abs(X - centers), axis=1)
        threshold = np.percentile(distances, 95)  # Using 95th percentile as threshold
        anomalies = X[distances > threshold]

        # Plot potential anomalies
        plt.scatter(anomalies, np.zeros_like(anomalies), color='green', s=100, label='Potential Anomalies', zorder=5)

        # Customize the plot
        plt.title(f"Distribution of {feature} for User {user_id} with K-means Analysis")
        plt.xlabel(feature)
        plt.ylabel("Density")

        # Add text annotations for cluster centers
        y_max = plt.gca().get_ylim()[1]
        for i, center in enumerate(centers):
            plt.text(center, y_max * 0.95, f'Center {i + 1}: {center:.2f}',
                     horizontalalignment='center', verticalalignment='top', rotation=90)

        # Remove duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.tight_layout()
        plt.show()

        # Print information about potential anomalies and silhouette score
        print(f"Potential anomalies for {feature}:")
        print(f"Silhouette Score: {silhouette_avg:.4f}")
        clustering_quality_percentage = silhouette_avg * 100
        print(f"Clustering quality: {clustering_quality_percentage:.2f}%")

        return centers, anomalies, silhouette_avg

    def analyze_user(self, user_id, print_results=False):
        if user_id not in self.user_models:
            print(f"No model found for user {user_id}")
            return

        user_data = self.profiler.df[self.profiler.df['user_id'] == user_id].copy()

        predictions = []
        for _, row in user_data.iterrows():
            action_features = {feature: row[feature] for feature in self.features}
            prediction = self.predict_user_action(user_id, action_features)
            predictions.append(prediction)

        user_data['prediction'] = predictions

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

            print(self.visualizer.create_boxed_output(content, "User Analysis Summary"))

            self.visualizer.visualize_feature_distributions(user_id, user_data)

            # Add k-means analysis
            self.analyze_user_patterns_with_kmeans(user_id, 'hour_of_timestamp')
            self.analyze_user_patterns_with_kmeans(user_id, 'session_duration')


        return user_data, normal_data, anomalous_data


if __name__ == "__main__":
    preprocessed_file_path = 'csv_dir/jerusalem_location_15.csv'
    preprocessed_df = preprocess_data.Preprocessor(preprocessed_file_path).preprocess()

    iforest_model = IsolationForestModel(preprocessed_df, max_workers=10, save_models=True, overwrite_models=False,
                                         save_evaluations=True, overwrite_evaluations=False)
    iforest_model.train_or_load_all_users()

    # Example usage
    example_user_id = '095ffcae-011c-4b6d-a5a3-5eea7368806f'
    example_action_features = {
        'hour_of_timestamp': 15,
        'phone_versions': 'iPhone14_5',
        'iOS sum': 1,
        'Android sum': 0,
        'is_denied': 0,
        'session_duration': 300,
        'location_or_ip': 'Jerusalem, Israel (212.117.143.250)'
    }
    # prediction = iforest_model.predict_user_action(example_user_id, example_action_features)
    # print(f"Prediction for new action: {prediction}")

    # Evaluate the model
    iforest_model.evaluate_model()

    # Visualize anomalies for the example user
    iforest_model.visualize_anomalies(example_user_id)

    # Perform comprehensive analysis for the example user
    iforest_model.analyze_user(example_user_id, print_results=True)

    # You can add more users to analyze here
    # other_user_id = 'another-user-id'
    # iforest_model.analyze_user(other_user_id)
