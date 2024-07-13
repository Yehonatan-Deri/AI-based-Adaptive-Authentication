import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
from contextlib import contextmanager


class AnomalyVisualizer:
    def __init__(self):
        self.categorical_features = ['phone_versions', 'location_or_ip']

    @staticmethod
    def create_boxed_output(content, title=None):
        lines = content.split('\n')
        width = max(len(line) for line in lines) + 4  # +4 for padding
        if title:
            width = max(width, len(title) + 4)
        box_top = '=' * width
        box_bottom = '=' * width
        padded_lines = [f"║ {line:<{width - 4}} ║" for line in lines]
        if title:
            title_line = f"║ {title:^{width - 4}} ║"
            return '\n'.join([box_top, title_line, box_top] + padded_lines + [box_bottom])
        else:
            return '\n'.join([box_top] + padded_lines + [box_bottom])

    @contextmanager
    def _suppress_warnings(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            yield

    def _print_table(self, data, headers):
        max_lengths = [max(len(str(item)) for item in col) for col in zip(*data, headers)]
        format_str = " | ".join(f"{{:<{length}}}" for length in max_lengths)
        table_content = format_str.format(*headers) + '\n'
        table_content += "-" * (sum(max_lengths) + 3 * (len(headers) - 1)) + '\n'
        for row in data:
            table_content += format_str.format(*[str(item) for item in row]) + '\n'
        return table_content.strip()

    def visualize_os_distribution(self, user_id, user_data):
        ios_sum = user_data['iOS sum'].sum()
        android_sum = user_data['Android sum'].sum()
        total = ios_sum + android_sum
        data = [
            ["iOS", ios_sum, f"{ios_sum / total * 100:.2f}%"],
            ["Android", android_sum, f"{android_sum / total * 100:.2f}%"]
        ]
        table_content = self._print_table(data, ["OS", "Count", "Percentage"])
        print(self.create_boxed_output(table_content, f"OS Distribution for User {user_id}"))

    def visualize_action_approval(self, user_id, user_data):
        approved = len(user_data) - user_data['is_denied'].sum()
        denied = user_data['is_denied'].sum()
        total = approved + denied
        data = [
            ["Approved", approved, f"{approved / total * 100:.2f}%"],
            ["Denied", denied, f"{denied / total * 100:.2f}%"]
        ]
        table_content = self._print_table(data, ["Action", "Count", "Percentage"])
        print(self.create_boxed_output(table_content, f"Action Approval Rate for User {user_id}"))

    def visualize_phone_versions(self, user_id, user_data):
        version_counts = user_data['phone_versions'].value_counts().sort_values(ascending=False)
        total = len(user_data)
        data = [[version, count, f"{count / total * 100:.2f}%"] for version, count in version_counts.items()]
        table_content = self._print_table(data, ["Phone Version", "Count", "Percentage"])
        print(self.create_boxed_output(table_content, f"Phone Versions Distribution for User {user_id}"))

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
            self.visualize_feature_distribution(user_id, normal_data, anomalous_data, 'session_duration')
            self.visualize_feature_distribution(user_id, normal_data, anomalous_data, 'hour_of_timestamp')

    def visualize_user_anomalies(self, user_id, user_data, features):
        with self._suppress_warnings():
            self.visualize_os_distribution(user_id, user_data)
            self.visualize_action_approval(user_id, user_data)
            self.visualize_phone_versions(user_id, user_data)

    def print_evaluation_results(self, results):
        content = "Evaluation Results:\n"
        total = sum(results.values())
        for category, count in results.items():
            percentage = (count / total) * 100 if total > 0 else 0
            content += f"{category.capitalize()}: {count} ({percentage:.2f}%)\n"
        print(self.create_boxed_output(content.strip(), "Model Evaluation"))

    def print_random_users(self, user_results):
        for category in ['need_second_check', 'invalid']:
            users = [user for user, results in user_results.items() if results[category] > 0]
            if users:
                print(self.create_boxed_output(f"Displaying up to 3 random users from {category} category:",
                                               "Random Users Sample"))
                for user in np.random.choice(users, min(3, len(users)), replace=False):
                    content = f"User ID: {user}\n"
                    for key, value in user_results[user].items():
                        content += f"{key.capitalize()}: {value}\n"
                    print(self.create_boxed_output(content.strip()))
            else:
                print(self.create_boxed_output(f"No users found in {category} category", "Random Users Sample"))
