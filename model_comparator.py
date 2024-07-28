
import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, \
    average_precision_score

import preprocess_data
from isolation_forest_model import IsolationForestModel
from lof_model import LOFModel
from ocsvm_model import OCSVMModel


class ModelComparator:
    def __init__(self, lof_model, iforest_model, ocsvm_model):
        self.models = {
            'LOF': lof_model,
            'Isolation Forest': iforest_model,
            'OCSVM': ocsvm_model
        }
        self.evaluation_results = {}
        self.performance_metrics = {}

    def load_evaluations(self):
        for model_name, model in self.models.items():
            self.evaluation_results[model_name] = {}
            eval_dir = f'{model_name.lower().replace(" ", "_")}_evaluations'
            if os.path.exists(eval_dir):
                print(f"Loading evaluations for {model_name}...")
                pkl_files = glob.glob(os.path.join(eval_dir, '*_evaluation.pkl'))
                print(f"Found {len(pkl_files)} PKL files in {eval_dir}")

                for file_path in pkl_files:
                    user_id = os.path.basename(file_path).split('_evaluation.pkl')[0]
                    try:
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                        if isinstance(data, tuple) and len(data) == 3:
                            self.evaluation_results[model_name][user_id] = data
                        else:
                            print(f"Warning: Unexpected data structure in {file_path}")
                    except Exception as e:
                        print(f"Error loading {file_path}: {str(e)}")

                print(f"Successfully loaded {len(self.evaluation_results[model_name])} evaluations for {model_name}")
            else:
                print(f"Warning: Evaluation directory not found for {model_name}: {eval_dir}")

        # Print summary of loaded evaluations
        print("\nSummary of loaded evaluations:")
        for model_name, evaluations in self.evaluation_results.items():
            print(f"{model_name}: {len(evaluations)} users")

        # Check for missing users across models
        all_users = set()
        for evaluations in self.evaluation_results.values():
            all_users.update(evaluations.keys())

        print("\nChecking for missing users across models:")
        for user_id in all_users:
            missing_models = [model_name for model_name, evaluations in self.evaluation_results.items() if
                              user_id not in evaluations]
            if missing_models:
                print(f"User {user_id} is missing from: {', '.join(missing_models)}")

    def calculate_performance_metrics(self):
        for model_name, evaluations in self.evaluation_results.items():
            true_labels = []
            predicted_labels = []
            for user_id, (user_data, _, _) in evaluations.items():
                true_labels.extend(['anomaly' if label != 0 else 'normal' for label in user_data['is_denied']])
                predicted_labels.extend(
                    ['anomaly' if label != 'valid' else 'normal' for label in user_data['prediction']])

            # Check if both classes are present
            unique_true = set(true_labels)
            unique_pred = set(predicted_labels)

            if len(unique_true) < 2 or len(unique_pred) < 2:
                print(f"Warning: Not all classes are present in the {model_name} results.")
                print(f"Unique true labels: {unique_true}")
                print(f"Unique predicted labels: {unique_pred}")
                self.performance_metrics[model_name] = {
                    'confusion_matrix': None,
                    'classification_report': None
                }
                continue

            try:
                cm = confusion_matrix(true_labels, predicted_labels, labels=['normal', 'anomaly'])
                cr = classification_report(true_labels, predicted_labels,
                                           target_names=['normal', 'anomaly'],
                                           output_dict=True,
                                           zero_division=0)

                self.performance_metrics[model_name] = {
                    'confusion_matrix': cm,
                    'classification_report': cr
                }
            except ValueError as e:
                print(f"Error calculating metrics for {model_name}: {str(e)}")
                self.performance_metrics[model_name] = {
                    'confusion_matrix': None,
                    'classification_report': None
                }

    def compare_overall_performance(self):
        print("Overall Performance Comparison:")
        for model_name, metrics in self.performance_metrics.items():
            print(f"\n{model_name}:")
            if metrics['classification_report'] is None:
                print("Performance metrics could not be calculated.")
            else:
                cr = metrics['classification_report']
                print(f"Accuracy: {cr['accuracy']:.4f}")
                print(f"Precision (Anomaly): {cr['anomaly']['precision']:.4f}")
                print(f"Recall (Anomaly): {cr['anomaly']['recall']:.4f}")
                print(f"F1-score (Anomaly): {cr['anomaly']['f1-score']:.4f}")
    def compare_feature_importance(self):
        feature_importance = {model_name: {feature: 0 for feature in self.models[model_name].features}
                              for model_name in self.models.keys()}

        for model_name, model in self.models.items():
            for feature, weight in model.feature_weights.items():
                feature_importance[model_name][feature] = weight

        df = pd.DataFrame(feature_importance)
        df.plot(kind='bar')
        plt.title("Feature Importance Comparison")
        plt.xlabel("Features")
        plt.ylabel("Importance Weight")
        plt.legend(title="Models")
        plt.tight_layout()
        plt.show()

    def compare_user_consistency(self):
        user_consistency = {}
        all_user_ids = set()
        for model_results in self.evaluation_results.values():
            all_user_ids.update(model_results.keys())

        for user_id in all_user_ids:
            predictions = {}
            for model_name, model_results in self.evaluation_results.items():
                if user_id in model_results:
                    user_data, _, _ = model_results[user_id]
                    predictions[model_name] = user_data['prediction'].tolist()
                else:
                    print(f"Warning: User {user_id} is missing from {model_name} model's evaluations.")

            if len(predictions) == len(self.models):
                prediction_lengths = [len(p) for p in predictions.values()]
                if len(set(prediction_lengths)) > 1:
                    print(f"Warning: Inconsistent prediction lengths for user {user_id}")
                    user_consistency[user_id] = None
                else:
                    agreement = sum(len(set(p)) == 1 for p in zip(*predictions.values()))
                    consistency = agreement / prediction_lengths[0]
                    user_consistency[user_id] = consistency
            else:
                user_consistency[user_id] = None

        valid_consistencies = [c for c in user_consistency.values() if c is not None]

        if valid_consistencies:
            plt.figure(figsize=(10, 6))
            plt.hist(valid_consistencies, bins=20)
            plt.title("User Consistency Across Models")
            plt.xlabel("Consistency Score")
            plt.ylabel("Number of Users")
            plt.tight_layout()
            plt.show()
        else:
            print("No valid consistency scores to plot.")

        return user_consistency

    def identify_challenging_cases(self):
        challenging_cases = {}
        all_user_ids = set()
        for model_results in self.evaluation_results.values():
            all_user_ids.update(model_results.keys())

        for user_id in all_user_ids:
            user_predictions = {}
            for model_name, model_results in self.evaluation_results.items():
                if user_id in model_results:
                    user_data, _, _ = model_results[user_id]
                    user_predictions[model_name] = user_data['prediction'].tolist()
                else:
                    print(f"Warning: User {user_id} is missing from {model_name} model's evaluations.")

            if len(user_predictions) == len(self.models):
                prediction_lengths = [len(p) for p in user_predictions.values()]
                if len(set(prediction_lengths)) > 1:
                    print(f"Warning: Inconsistent prediction lengths for user {user_id}")
                else:
                    disagreements = [i for i, predictions in enumerate(zip(*user_predictions.values()))
                                     if len(set(predictions)) > 1]

                    if disagreements:
                        challenging_cases[user_id] = disagreements

        return challenging_cases

    def plot_precision_recall_curves_with_kmeans(self):
        plt.figure(figsize=(12, 10))

        # Use a list of colors instead of get_cmap
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf']

        for (model_name, evaluations), color in zip(self.evaluation_results.items(), colors):
            true_labels = []
            predicted_scores = []
            for user_id, (user_data, _, _) in evaluations.items():
                true_labels.extend([1 if label != 0 else 0 for label in user_data['is_denied']])
                score_map = {'valid': 0, 'need_second_check': 0.5, 'invalid': 1}
                scores = [score_map[pred] for pred in user_data['prediction']]
                predicted_scores.extend(scores)

            true_labels = np.array(true_labels)
            predicted_scores = np.array(predicted_scores)

            if len(np.unique(true_labels)) < 2:
                print(f"Warning: Only one class present in true labels for {model_name}")
                continue

            precision, recall, _ = precision_recall_curve(true_labels, predicted_scores)
            average_precision = average_precision_score(true_labels, predicted_scores)

            plt.plot(recall, precision, label=f'{model_name} (AP = {average_precision:.2f})')

            # Apply K-means clustering
            points = np.column_stack((recall, precision))
            kmeans = KMeans(n_clusters=2, random_state=42)
            cluster_labels = kmeans.fit_predict(points)

            # # Plot both clusters
            # for i in range(2):  # Changed from range(1) to range(2)
            #     cluster_points = points[cluster_labels == i]
            #     plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
            #                 label=f'{model_name} Cluster {i + 1}', alpha=0.7, s=150)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves with K-means Clustering')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def run_comparison(self):
        print("Loading evaluations...")
        self.load_evaluations()

        print("Calculating performance metrics...")
        self.calculate_performance_metrics()

        print("Comparing overall performance...")
        self.compare_overall_performance()

        print("Comparing feature importance...")
        self.compare_feature_importance()

        print("Analyzing user consistency...")
        user_consistency = self.compare_user_consistency()

        print("Identifying challenging cases...")
        challenging_cases = self.identify_challenging_cases()

        print("\nAnalysis complete. You can further investigate user_consistency and challenging_cases.")
        return user_consistency, challenging_cases

    def run_enhanced_comparison(self):
        comparison_methods = [
            ("Comparing precision-recall curves with K-means", self.plot_precision_recall_curves_with_kmeans),
        ]

        for description, method in comparison_methods:
            print(f"\n{description}...")
            try:
                method()
            except Exception as e:
                print(f"Error in {description}: {str(e)}")
                print("Skipping this comparison and continuing with the next one.")


if __name__ == "__main__":
    # Load preprocessed data
    preprocessed_file_path = 'csv_dir/jerusalem_location_15.csv'
    preprocessed_df = preprocess_data.Preprocessor(preprocessed_file_path).preprocess()

    # Initialize models
    lof_model = LOFModel(preprocessed_df)
    iforest_model = IsolationForestModel(preprocessed_df)
    ocsvm_model = OCSVMModel(preprocessed_df)

    # Create ModelComparator instance
    comparator = ModelComparator(lof_model, iforest_model, ocsvm_model)

    # Run comparison
    user_consistency, challenging_cases = comparator.run_comparison()
    comparator.run_enhanced_comparison()
    # You can now analyze user_consistency and challenging_cases further if needed
