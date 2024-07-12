import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
from contextlib import contextmanager


class AnomalyVisualizer:
    def __init__(self):
        self.categorical_features = ['phone_versions', 'location_or_ip']

    @contextmanager
    def _suppress_warnings(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            yield

    def _print_section(self, title):
        print("\n" + "=" * 50)
        print(f"{title:^50}")
        print("=" * 50)

    def _print_table(self, data, headers):
        max_lengths = [max(len(str(item)) for item in col) for col in zip(*data, headers)]

        format_str = " | ".join(f"{{:<{length}}}" for length in max_lengths)
        print(format_str.format(*headers))
        print("-" * (sum(max_lengths) + 3 * (len(headers) - 1)))

        for row in data:
            print(format_str.format(*[str(item) for item in row]))

    def visualize_os_distribution(self, user_id, user_data):
        ios_sum = user_data['iOS sum'].sum()
        android_sum = user_data['Android sum'].sum()
        total = ios_sum + android_sum

        self._print_section(f"OS Distribution for User {user_id}")
        data = [
            ["iOS", ios_sum, f"{ios_sum / total * 100:.2f}%"],
            ["Android", android_sum, f"{android_sum / total * 100:.2f}%"]
        ]
        self._print_table(data, ["OS", "Count", "Percentage"])

    def visualize_action_approval(self, user_id, user_data):
        approved = len(user_data) - user_data['is_denied'].sum()
        denied = user_data['is_denied'].sum()
        total = approved + denied

        self._print_section(f"Action Approval Rate for User {user_id}")
        data = [
            ["Approved", approved, f"{approved / total * 100:.2f}%"],
            ["Denied", denied, f"{denied / total * 100:.2f}%"]
        ]
        self._print_table(data, ["Action", "Count", "Percentage"])

    def visualize_phone_versions(self, user_id, user_data):
        version_counts = user_data['phone_versions'].value_counts().sort_values(ascending=False)
        total = len(user_data)

        self._print_section(f"Phone Versions Distribution for User {user_id}")
        data = [[version, count, f"{count / total * 100:.2f}%"] for version, count in version_counts.items()]
        self._print_table(data, ["Phone Version", "Count", "Percentage"])

    def visualize_feature_distribution(self, user_id, normal_data, anomalous_data, feature):
        with self._suppress_warnings():
            plt.figure(figsize=(12, 6))
            sns.histplot(normal_data[feature], kde=True, color="blue", label="Normal", alpha=0.5)
            sns.histplot(anomalous_data[feature], kde=True, color="red", label="Anomalous", alpha=0.5)
            plt.title(f"Distribution of {feature} for User {user_id}")
            plt.xlabel(feature)
            plt.ylabel("Density")
            plt.legend()
            plt.tight_layout()
            plt.show()

    def visualize_anomaly_comparison(self, user_id, normal_data, anomalous_data):
        with self._suppress_warnings():
            # Session Duration Distribution
            self.visualize_feature_distribution(user_id, normal_data, anomalous_data, 'session_duration')

            # Hour of Login Distribution
            self.visualize_feature_distribution(user_id, normal_data, anomalous_data, 'hour_of_timestamp')

    def visualize_user_anomalies(self, user_id, user_data, features):
        with self._suppress_warnings():
            self.visualize_os_distribution(user_id, user_data)
            self.visualize_action_approval(user_id, user_data)
            self.visualize_phone_versions(user_id, user_data)
