import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, \
    average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler

import preprocess_data
from isolation_forest_model import IsolationForestModel
from lof_model import LOFModel
from ocsvm_model import OCSVMModel


class ModelComparator:
    """
    A class for comparing different anomaly detection models.

    This class provides methods to load evaluation results, calculate performance metrics,
    and compare different aspects of the models such as overall performance, feature importance,
    and user consistency.
    """

    def __init__(self, lof_model, iforest_model, ocsvm_model):
        """
        Initialize the ModelComparator with different anomaly detection models.

        Args:
            lof_model (LOFModel): Local Outlier Factor model
            iforest_model (IsolationForestModel): Isolation Forest model
            ocsvm_model (OCSVMModel): One-Class SVM model
        """
        self.models = {
            'LOF': lof_model,
            'Isolation Forest': iforest_model,
            'OCSVM': ocsvm_model
        }
        self.evaluation_results = {}
        self.performance_metrics = {}

    def load_evaluations(self):
        """
        Load evaluation results for each model from pickle files.

        This method searches for evaluation files in model-specific directories,
        loads the data, and stores it in the evaluation_results dictionary.
        It also checks for missing users across different models.
        """
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

        # Print summary and check for missing users
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
        """
        Calculate performance metrics for each model.

        This method computes confusion matrices and classification reports for each model
        based on the loaded evaluation results. The metrics are stored in the performance_metrics dictionary.
        """
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
        """
        Compare and print the overall performance of each model.

        This method prints accuracy, precision, recall, and F1-score for anomaly detection
        for each model based on the calculated performance metrics.
        """
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
        """
        Compare and visualize the feature importance across different models.

        This method creates a bar plot showing the importance of each feature
        for each model based on the feature weights defined in the models.
        """
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
        """
        Compare the consistency of predictions across different models for each user.

        This method calculates a consistency score for each user based on how often
        the models agree on their predictions. It also generates a histogram of
        consistency scores across all users.

        Returns:
            dict: A dictionary of user consistency scores.
        """
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
        """
        Identify cases where the models disagree on their predictions.

        This method finds instances where the different models give conflicting
        predictions for the same user action.

        Returns:
            dict: A dictionary of challenging cases, where keys are user IDs and
                  values are lists of indices where models disagree.
        """
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
        """
        Plot precision-recall curves for each model and apply K-means clustering.

        This method generates precision-recall curves for each model and applies
        K-means clustering to identify distinct regions in the curves. The results
        are visualized in a single plot for easy comparison.
        """
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
            # cluster_labels = kmeans.fit_predict(points)

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
        """
        Run the full comparison pipeline.

        This method executes all comparison steps in sequence, including loading evaluations,
        calculating performance metrics, comparing overall performance, feature importance,
        user consistency, and identifying challenging cases.

        Returns:
            tuple: A tuple containing user consistency scores and challenging cases.
        """
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

    def plot_simplified_performance_by_characteristics_table(self):
        for characteristic in ['session_duration', 'action_count']:
            data = []
            for model_name, evaluations in self.evaluation_results.items():
                user_characteristics = []
                user_f1_scores = []

                for user_id, (user_data, _, _) in evaluations.items():
                    true_labels = [1 if label != 0 else 0 for label in user_data['is_denied']]
                    predicted_labels = [1 if pred != 'valid' else 0 for pred in user_data['prediction']]

                    if characteristic == 'session_duration':
                        char_value = user_data['session_duration'].mean()
                    else:  # action_count
                        char_value = len(user_data)

                    user_characteristics.append(char_value)
                    user_f1_scores.append(f1_score(true_labels, predicted_labels, average='weighted'))

                df = pd.DataFrame({'Characteristic': user_characteristics, 'F1-Score': user_f1_scores})

                # Create bins for the characteristic
                df['Bin'] = pd.qcut(df['Characteristic'], q=3, labels=['Low', 'Medium', 'High'])

                # Calculate average F1-Score for each bin
                stats = df.groupby('Bin')['F1-Score'].mean().reset_index()
                stats['Model'] = model_name
                data.append(stats)

            # Combine data from all models
            combined_data = pd.concat(data)

            # Pivot the table to have models as rows and categories as columns
            pivot_table = combined_data.pivot(index='Model', columns='Bin', values='F1-Score')
            pivot_table = pivot_table.reindex(columns=['Low', 'Medium', 'High'])  # Ensure correct order
            pivot_table = pivot_table.round(3)  # Round to 3 decimal places

            # Create the table
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.axis('off')
            table = ax.table(cellText=pivot_table.values,
                             rowLabels=pivot_table.index,
                             colLabels=pivot_table.columns,
                             cellLoc='center',
                             loc='center')

            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.5)

            # Color the header row and column
            for i in range(len(pivot_table.columns)):
                table[(0, i)].set_facecolor('#4472C4')
                table[(0, i)].set_text_props(color='white')
            for i in range(len(pivot_table.index)):
                table[(i + 1, -1)].set_facecolor('#4472C4')
                table[(i + 1, -1)].set_text_props(color='white')

            plt.title(f'Average F1-Score by {characteristic.replace("_", " ").title()} Category', fontsize=16, pad=20)
            plt.tight_layout()
            plt.show()

    def plot_time_series_anomalies(self):
        plt.figure(figsize=(12, 6))
        for model_name, evaluations in self.evaluation_results.items():
            timestamps = []
            anomaly_counts = []

            for user_id, (user_data, _, _) in evaluations.items():
                user_data['timestamp'] = pd.to_datetime(user_data['@timestamp'])
                daily_anomalies = user_data[user_data['prediction'] != 'valid'].groupby(
                    user_data['timestamp'].dt.date).size()
                timestamps.extend(daily_anomalies.index)
                anomaly_counts.extend(daily_anomalies.values)

            df = pd.DataFrame({'date': timestamps, 'anomalies': anomaly_counts})
            df = df.groupby('date').sum().sort_index()

            plt.plot(df.index, df['anomalies'], label=model_name)

        plt.xlabel('Date')
        plt.ylabel('Number of Anomalies Detected')
        plt.title('Time Series of Anomaly Detections by Model')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_performance_table(self):
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        model_names = []
        data = []

        for model_name, metrics_dict in self.performance_metrics.items():
            if metrics_dict['classification_report'] is not None:
                model_names.append(model_name)
                cr = metrics_dict['classification_report']
                row = [cr['accuracy'], cr['anomaly']['precision'], cr['anomaly']['recall'], cr['anomaly']['f1-score']]
                data.append([f'{value:.3f}' for value in row])

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('off')
        table = ax.table(cellText=data,
                         rowLabels=model_names,
                         colLabels=metrics,
                         cellLoc='center',
                         loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Color the header row
        for i in range(len(metrics)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(color='white')

        # Alternate row colors for better readability
        for i in range(len(model_names)):
            if i % 2 == 0:
                for j in range(len(metrics)):
                    table[(i + 1, j)].set_facecolor('#D9E1F2')

        plt.title('Performance Table Comparison', fontsize=16, pad=20)
        plt.tight_layout()
        plt.show()

    def run_enhanced_comparison(self):
        """
        Run enhanced comparison methods.

        This method executes additional comparison techniques, such as
        plotting precision-recall curves with K-means clustering.
        """
        comparison_methods = [
            ("Comparing precision-recall curves with K-means", self.plot_precision_recall_curves_with_kmeans),
            ("Plotting performance by user characteristics", self.plot_simplified_performance_by_characteristics_table),
            ("Plotting time series of anomalies", self.plot_time_series_anomalies),
            ("Plotting performance radar chart", self.plot_performance_table),
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
