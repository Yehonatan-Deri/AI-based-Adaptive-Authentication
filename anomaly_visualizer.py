import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
from contextlib import contextmanager
from matplotlib.backends.backend_agg import FigureCanvasAgg


class AnomalyVisualizer:
    def __init__(self):
        self.categorical_features = ['phone_versions', 'location_or_ip']

    @contextmanager
    def _suppress_warnings(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            yield

    def visualize_user_anomalies(self, user_id, user_data, features):
        if user_data.empty:
            print(f"No data found for user {user_id}")
            return

        with self._suppress_warnings():
            self._plot_os_distribution(user_id, user_data)
            self._plot_denial_rate(user_id, user_data)
            self._plot_phone_versions(user_id, user_data)
            self._plot_correlation_heatmap(user_id, user_data, features)

    def _plot_os_distribution(self, user_id, user_data):
        ios_sum = user_data['iOS sum'].sum()
        android_sum = user_data['Android sum'].sum()
        total = ios_sum + android_sum

        plt.figure(figsize=(10, 6))
        plt.pie([ios_sum, android_sum], labels=['iOS', 'Android'], autopct='%1.1f%%')
        plt.title(f"OS Distribution for User {user_id}")
        plt.figtext(0.5, 0.01, f"Total actions: {total}", ha='center')
        plt.show()

    def _plot_denial_rate(self, user_id, user_data):
        denied = user_data['is_denied'].sum()
        total = len(user_data)
        approval_rate = (total - denied) / total * 100

        plt.figure(figsize=(10, 6))
        plt.pie([denied, total - denied], labels=['Denied', 'Approved'], autopct='%1.1f%%')
        plt.title(f"Action Approval Rate for User {user_id}")
        plt.figtext(0.5, 0.01, f"Total actions: {total}, Approval rate: {approval_rate:.2f}%", ha='center')
        plt.show()

    def _plot_phone_versions(self, user_id, user_data):
        version_counts = user_data['phone_versions'].value_counts()
        total = len(user_data)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        table = ax.table(
            cellText=[[version, count, f"{count / total * 100:.2f}%"] for version, count in version_counts.items()],
            colLabels=['Phone Version', 'Count', 'Percentage'],
            cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        plt.title(f"Phone Versions Distribution for User {user_id}")
        plt.figtext(0.5, 0.01, f"Total actions: {total}", ha='center')
        plt.show()

    def _plot_correlation_heatmap(self, user_id, user_data, features):
        numeric_features = [f for f in features if f not in self.categorical_features]
        corr_matrix = user_data[numeric_features].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title(f"Feature Correlation Heatmap for User {user_id}")
        plt.show()

    def visualize_anomaly_comparison(self, user_id, normal_data, anomalous_data, features):
        with self._suppress_warnings():
            self._compare_os_distribution(user_id, normal_data, anomalous_data)
            self._compare_denial_rate(user_id, normal_data, anomalous_data)
            self._compare_numeric_features(user_id, normal_data, anomalous_data, features)

    def _compare_os_distribution(self, user_id, normal_data, anomalous_data):
        normal_ios = normal_data['iOS sum'].sum()
        normal_android = normal_data['Android sum'].sum()
        anomalous_ios = anomalous_data['iOS sum'].sum()
        anomalous_android = anomalous_data['Android sum'].sum()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.pie([normal_ios, normal_android], labels=['iOS', 'Android'], autopct='%1.1f%%')
        ax1.set_title("Normal Data")
        ax2.pie([anomalous_ios, anomalous_android], labels=['iOS', 'Android'], autopct='%1.1f%%')
        ax2.set_title("Anomalous Data")
        plt.suptitle(f"OS Distribution Comparison for User {user_id}")
        plt.show()

    def _compare_denial_rate(self, user_id, normal_data, anomalous_data):
        normal_denied = normal_data['is_denied'].sum()
        normal_total = len(normal_data)
        anomalous_denied = anomalous_data['is_denied'].sum()
        anomalous_total = len(anomalous_data)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.pie([normal_denied, normal_total - normal_denied], labels=['Denied', 'Approved'], autopct='%1.1f%%')
        ax1.set_title("Normal Data")
        ax2.pie([anomalous_denied, anomalous_total - anomalous_denied], labels=['Denied', 'Approved'],
                autopct='%1.1f%%')
        ax2.set_title("Anomalous Data")
        plt.suptitle(f"Denial Rate Comparison for User {user_id}")
        plt.show()

    def _compare_numeric_features(self, user_id, normal_data, anomalous_data, features):
        numeric_features = [f for f in features if
                            f not in self.categorical_features and f not in ['iOS sum', 'Android sum', 'is_denied']]

        for feature in numeric_features:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=normal_data, x=feature, kde=True, label='Normal', color='blue', alpha=0.5)
            sns.histplot(data=anomalous_data, x=feature, kde=True, label='Anomalous', color='red', alpha=0.5)
            plt.title(f"Comparison of {feature} for User {user_id}")
            plt.xlabel(feature)
            plt.ylabel("Density")
            plt.legend()
            plt.show()
    def visualize_anomaly_comparison(self, user_id, normal_data, anomalous_data):
        with self._suppress_warnings():
            # Session Duration Distribution
            self.visualize_feature_distribution(user_id, normal_data, anomalous_data, 'session_duration')

            # Hour of Login Distribution
            self.visualize_feature_distribution(user_id, normal_data, anomalous_data, 'hour_of_timestamp')

        # Visualize locations on map
        self.visualize_locations_on_map(user_id, normal_data, anomalous_data)
