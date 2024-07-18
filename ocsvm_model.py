import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from user_profiling import UserProfiler
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures
import threading
from tqdm import tqdm
import pickle
import os
from anomaly_visualizer import AnomalyVisualizer

class OCSVMModel:
    def __init__(self, preprocessed_df, kernel='rbf', nu=0.1, gamma='scale', min_samples=5, max_workers=None,
                 save_models=False, overwrite_models=False, save_evaluations=False, overwrite_evaluations=False):
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
            'hour_of_timestamp': 0.2,
            'phone_versions': 0.2,
            'iOS sum': 0.1,
            'Android sum': 0.1,
            'is_denied': 0.2,
            'session_duration': 0.2,
            'location_or_ip': 0.3
        }
        self.feature_thresholds = {feature: -0.5 for feature in self.features}
        self.visualizer = AnomalyVisualizer()
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma

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

            ocsvm = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)
            ocsvm.fit(scaled_feature)

            user_models[feature] = ocsvm
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
        if not os.path.exists('ocsvm_trained_models'):
            os.makedirs('ocsvm_trained_models')

        model_path = f'ocsvm_trained_models/{user_id}_model.pkl'

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
        model_path = f'ocsvm_trained_models/{user_id}_model.pkl'
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.user_models[user_id] = data['models']
                    self.user_scalers[user_id] = data['scalers']
                    self.categorical_columns[user_id] = data['categorical_columns']
                    self.user_test_data[user_id] = data['test_data']
                # print(f"Loaded model for user {user_id}") todo delete print
                return True
            except Exception as e:
                print(f"Error loading model for user {user_id}: {e}")
                return False
        return False

    def train_or_load_all_users(self):
        user_ids = list(self.profiler.df['user_id'].unique())
        loaded_count = 0
        trained_count = 0
        for user_id in tqdm(user_ids, desc="Training/Loading users"):
            if not self.overwrite_models and self.load_user_model(user_id):
                loaded_count += 1
                continue
            self.train_user_model(user_id)
            trained_count += 1
        # print(f"Loaded {loaded_count} models, trained {trained_count} models")
    # todo delete print
    def predict_user_action(self, user_id, action_features, threshold_inbetween=-0.2, threshold_invalid=-0.5):
        if user_id not in self.user_models:
            raise ValueError(f"No model found for user_id: {user_id}")

        ocsvm_scores = []
        for feature in self.features:
            if feature not in action_features:
                continue

            value = action_features[feature]
            ocsvm = self.user_models[user_id][feature]
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
            ocsvm_score = ocsvm.decision_function(scaled_feature)[0]

            profile_weight = self.calculate_profile_weight(user_id, feature, value)
            weighted_score = ocsvm_score * profile_weight * self.feature_weights[feature]
            ocsvm_scores.append((feature, weighted_score))

        return self.aggregate_scores(ocsvm_scores, threshold_inbetween, threshold_invalid)

    def calculate_profile_weight(self, user_id, feature, value):
        profile = self.profiler.create_user_profile(user_id)
        if feature in profile:
            profile_value = profile[feature]
            deviation = abs(profile_value - value)
            return 1 / (1 + deviation)
        return 1.0

    def aggregate_scores(self, ocsvm_scores, threshold_inbetween, threshold_invalid):
        total_weighted_score = sum(score for _, score in ocsvm_scores)

        if total_weighted_score <= threshold_invalid:
            return "invalid"
        elif threshold_invalid < total_weighted_score <= threshold_inbetween:
            return "need_second_check"
        else:
            return "valid"

    def save_evaluation(self, user_id, evaluation_result):
        if not os.path.exists('ocsvm_evaluations'):
            os.makedirs('ocsvm_evaluations')

        eval_path = f'ocsvm_evaluations/{user_id}_evaluation.pkl'

        if not self.overwrite_evaluations and os.path.exists(eval_path):
            print(f"Evaluation for user {user_id} already exists. Skipping save.")
            return

        with open(eval_path, 'wb') as f:
            pickle.dump(evaluation_result, f)

    def load_evaluation(self, user_id):
        eval_path = f'ocsvm_evaluations/{user_id}_evaluation.pkl'
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

        return user_data, normal_data, anomalous_data

if __name__ == "__main__":
    import preprocess_data

    preprocessed_file_path = 'csv_dir/jerusalem_location_15.csv'
    preprocessed_df = preprocess_data.Preprocessor(preprocessed_file_path).preprocess()

    ocsvm_model = OCSVMModel(preprocessed_df, max_workers=10, save_models=True, overwrite_models=False,
                             save_evaluations=True, overwrite_evaluations=False)
    ocsvm_model.train_or_load_all_users()

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
    # prediction = ocsvm_model.predict_user_action(example_user_id, example_action_features)
    # print(f"Prediction for new action: {prediction}")

    # Evaluate the model
    ocsvm_model.evaluate_model()

    # Visualize anomalies for the example user
    ocsvm_model.visualize_anomalies(example_user_id)

    # Perform comprehensive analysis for the example user
    ocsvm_model.analyze_user(example_user_id, print_results=True)

    # You can add more users to analyze here
    # other_user_id = 'another-user-id'
    # ocsvm_model.analyze_user(other_user_id)