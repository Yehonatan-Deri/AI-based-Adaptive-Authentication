
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

    def plot_roc_curves(self):
        plt.figure(figsize=(10, 8))

        for model_name, evaluations in self.evaluation_results.items():
            true_labels = []
            predicted_scores = []
            for user_id, (user_data, _, _) in evaluations.items():
                true_labels.extend([1 if label != 0 else 0 for label in user_data['is_denied']])

                # Convert categorical predictions to numeric scores
                score_map = {'valid': 0, 'need_second_check': 0.5, 'invalid': 1}
                scores = [score_map[pred] for pred in user_data['prediction']]
                predicted_scores.extend(scores)

            # Convert to numpy arrays
            true_labels = np.array(true_labels)
            predicted_scores = np.array(predicted_scores)

            # Print debug information
            print(f"Model: {model_name}")
            print(f"True labels shape: {true_labels.shape}")
            print(f"Predicted scores shape: {predicted_scores.shape}")
            print(f"Unique true labels: {np.unique(true_labels)}")
            print(f"Unique predicted scores: {np.unique(predicted_scores)}")

            # Ensure we have both positive and negative cases
            if len(np.unique(true_labels)) < 2:
                print(f"Warning: Only one class present in true labels for {model_name}")
                continue

            fpr, tpr, _ = roc_curve(true_labels, predicted_scores)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

            # Print the first and last points of the ROC curve
            print(f"First point of ROC curve: ({fpr[0]:.4f}, {tpr[0]:.4f})")
            print(f"Last point of ROC curve: ({fpr[-1]:.4f}, {tpr[-1]:.4f})")

        plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()

    def compare_decision_boundaries(self):
        # Get the feature names and profiler from one of the models
        first_model = next(iter(self.models.values()))
        feature_names = first_model.features
        profiler = first_model.profiler

        # Select only numeric features
        numeric_features = [f for f in feature_names if profiler.df[f].dtype in ['int64', 'float64']]

        if len(numeric_features) < 2:
            print("Not enough numeric features to create a 2D plot.")
            return

        # Use PCA to reduce to 2 dimensions if we have more than 2 numeric features
        if len(numeric_features) > 2:
            pca = PCA(n_components=2)
            X = pca.fit_transform(profiler.df[numeric_features].values)
        else:
            X = profiler.df[numeric_features].values

        y = profiler.df['is_denied'].values

        fig, axes = plt.subplots(1, len(self.models), figsize=(6 * len(self.models), 6))
        if len(self.models) == 1:
            axes = [axes]

        for i, (model_name, model) in enumerate(self.models.items()):
            # Create a mesh grid
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max - x_min) / 100),
                                 np.arange(y_min, y_max, (y_max - y_min) / 100))

            # Predict on the mesh grid
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            Z = []
            for point in mesh_points:
                action_features = dict(zip(numeric_features, point))
                prediction = model.predict_user_action(next(iter(model.user_models.keys())), action_features)
                Z.append(1 if prediction != 'valid' else 0)
            Z = np.array(Z).reshape(xx.shape)

            # Plot decision boundary
            axes[i].contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
            scatter = axes[i].scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='black')
            axes[i].set_title(f'{model_name} Decision Boundary')
            axes[i].set_xlabel('Feature 1')
            axes[i].set_ylabel('Feature 2')

        plt.tight_layout()
        plt.show()

    def compare_anomaly_scores(self):
        all_scores = {model_name: [] for model_name in self.models.keys()}
        for model_name, evaluations in self.evaluation_results.items():
            for user_id, (user_data, _, _) in evaluations.items():
                # Convert predictions to numeric values
                # Assuming 'valid' is 0, 'need_second_check' is 1, and 'invalid' is 2
                score_map = {'valid': 0, 'need_second_check': 1, 'invalid': 2}
                scores = user_data['prediction'].map(score_map)
                all_scores[model_name].extend(scores)

        plt.figure(figsize=(12, 6))
        plt.boxplot([scores for scores in all_scores.values()], labels=all_scores.keys())
        plt.title('Distribution of Anomaly Predictions Across Models')
        plt.ylabel('Anomaly Prediction (0: Valid, 1: Need Second Check, 2: Invalid)')
        plt.show()

        # Print some statistics
        for model_name, scores in all_scores.items():
            print(f"\n{model_name} Statistics:")
            print(f"Mean: {np.mean(scores):.2f}")
            print(f"Median: {np.median(scores):.2f}")
            print(f"Std Dev: {np.std(scores):.2f}")
            print(f"Min: {np.min(scores)}")
            print(f"Max: {np.max(scores)}")

    def compare_model_agreement(self):
        agreement_matrix = np.zeros((len(self.models), len(self.models)))
        model_names = list(self.models.keys())

        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i < j:
                    agreements = 0
                    total = 0
                    for user_id in self.evaluation_results[model1].keys():
                        if user_id in self.evaluation_results[model2]:
                            pred1 = self.evaluation_results[model1][user_id][0]['prediction']
                            pred2 = self.evaluation_results[model2][user_id][0]['prediction']
                            agreements += np.sum(pred1 == pred2)
                            total += len(pred1)

                    agreement_matrix[i, j] = agreement_matrix[j, i] = agreements / total if total > 0 else 0

        plt.figure(figsize=(10, 8))
        sns.heatmap(agreement_matrix, annot=True, xticklabels=model_names, yticklabels=model_names, cmap='YlGnBu')
        plt.title('Model Agreement Matrix')
        plt.show()

    def compare_feature_rankings(self):
        feature_rankings = {model_name: model.feature_weights for model_name, model in self.models.items()}

        # Calculate Kendall's Tau for each pair of models
        model_names = list(self.models.keys())
        kendall_matrix = np.zeros((len(model_names), len(model_names)))

        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i < j:
                    ranking1 = [feature_rankings[model1][f] for f in self.models[model1].features]
                    ranking2 = [feature_rankings[model2][f] for f in self.models[model2].features]
                    tau, _ = kendalltau(ranking1, ranking2)
                    kendall_matrix[i, j] = kendall_matrix[j, i] = tau

        plt.figure(figsize=(10, 8))
        sns.heatmap(kendall_matrix, annot=True, xticklabels=model_names, yticklabels=model_names, cmap='YlGnBu')
        plt.title("Kendall's Tau for Feature Rankings Between Models")
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

        # Calculate percentages
        df_percentage = df.apply(lambda x: x / x.sum() * 100, axis=0)

        # Create a figure for the grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 8))

        # Set the width of each bar and the positions of the bars
        bar_width = 0.25
        model_names = list(self.models.keys())  # Explicitly get model names
        index = np.arange(len(model_names))

        # Create the grouped bars
        ax.bar(index, df_percentage.loc['valid'], bar_width, label='Valid', color='green')
        ax.bar(index + bar_width, df_percentage.loc['need_second_check'], bar_width, label='Need Second Check',
               color='orange')
        ax.bar(index + 2 * bar_width, df_percentage.loc['invalid'], bar_width, label='Invalid', color='red')

        # Add labels and title
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Percentage of Predictions', fontsize=12)
        ax.set_title('Prediction Distribution Comparison', fontsize=14)
        ax.set_xticks(index + bar_width)
        ax.set_xticklabels(model_names, rotation=45, ha='right')

        # Add value labels on the bars
        for i, v in enumerate(df_percentage.loc['valid']):
            ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom')
        for i, v in enumerate(df_percentage.loc['need_second_check']):
            ax.text(i + bar_width, v, f'{v:.1f}%', ha='center', va='bottom')
        for i, v in enumerate(df_percentage.loc['invalid']):
            ax.text(i + 2 * bar_width, v, f'{v:.1f}%', ha='center', va='bottom')

        # Add a legend
        ax.legend()

        # Adjust layout to prevent cutting off x-axis labels
        plt.tight_layout()
        plt.show()

        # Print the exact percentages
        print("\nPrediction Distribution Percentages:")
        print(df_percentage.round(2).to_string())

        return df_percentage
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

        for model_name, evaluations in self.evaluation_results.items():
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

    def run_enhanced_comparison(self):
        # self.run_comparison()  # Run the existing comparison

        comparison_methods = [
            ("Plotting ROC curves", self.plot_roc_curves),
            ("Comparing decision boundaries", self.compare_decision_boundaries),
            # ("Comparing anomaly scores", self.compare_anomaly_scores),
            # ("Analyzing model agreement", self.compare_model_agreement),
            # ("Comparing feature rankings", self.compare_feature_rankings)
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
