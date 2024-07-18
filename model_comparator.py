
import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

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

    def plot_confusion_matrices(self):
        available_models = [model_name for model_name, metrics in self.performance_metrics.items() if
                            metrics['confusion_matrix'] is not None]
        num_models = len(available_models)

        if num_models == 0:
            print("No confusion matrices available to plot.")
            return

        fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 6))
        if num_models == 1:
            axes = [axes]

        for i, model_name in enumerate(available_models):
            cm = self.performance_metrics[model_name]['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i])
            axes[i].set_title(f"{model_name} Confusion Matrix")
            axes[i].set_xlabel("Predicted Label")
            axes[i].set_ylabel("True Label")

        plt.tight_layout()
        plt.show()

    def compare_prediction_distributions(self):
        prediction_counts = {model_name: {'valid': 0, 'need_second_check': 0, 'invalid': 0}
                             for model_name in self.models.keys()}

        for model_name, evaluations in self.evaluation_results.items():
            for user_id, (user_data, _, _) in evaluations.items():
                counts = user_data['prediction'].value_counts()
                for category in ['valid', 'need_second_check', 'invalid']:
                    prediction_counts[model_name][category] += counts.get(category, 0)

        df = pd.DataFrame(prediction_counts)
        df.plot(kind='bar', stacked=True)
        plt.title("Prediction Distribution Comparison")
        plt.xlabel("Models")
        plt.ylabel("Number of Predictions")
        plt.legend(title="Prediction Category")
        plt.tight_layout()
        plt.show()

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

    def run_comparison(self):
        print("Loading evaluations...")
        self.load_evaluations()

        print("Calculating performance metrics...")
        self.calculate_performance_metrics()

        print("Comparing overall performance...")
        self.compare_overall_performance()

        print("Plotting confusion matrices...")
        self.plot_confusion_matrices()

        print("Comparing prediction distributions...")
        self.compare_prediction_distributions()

        print("Comparing feature importance...")
        self.compare_feature_importance()

        print("Analyzing user consistency...")
        user_consistency = self.compare_user_consistency()

        print("Identifying challenging cases...")
        challenging_cases = self.identify_challenging_cases()

        print("\nAnalysis complete. You can further investigate user_consistency and challenging_cases.")
        return user_consistency, challenging_cases


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

    # You can now analyze user_consistency and challenging_cases further if needed
