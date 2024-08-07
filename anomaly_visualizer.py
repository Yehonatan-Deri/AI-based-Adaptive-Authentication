import warnings
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class AnomalyVisualizer:
    """
    A class for visualizing anomalies in user data.

    This class provides methods to visualize various aspects of user data,
    including OS distribution, action approval rates, phone versions,
    session durations, and feature distributions.
    """

    def __init__(self):
        """
        Initialize the AnomalyVisualizer with categorical features.
        """
        self.categorical_features = ['phone_versions', 'location_or_ip']

    @staticmethod
    def create_boxed_output(content, title=None):
        """
        Create a boxed output for displaying information.

        Args:
            content (str): The content to be displayed in the box.
            title (str, optional): The title of the box. Defaults to None.

        Returns:
            str: A string representation of the boxed output.
        """
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
        """
        A context manager to suppress specific warnings.

        This method is used to temporarily suppress UserWarnings and FutureWarnings
        when executing code that may generate these warnings.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            yield

    def _print_table(self, data, headers):
        """
        Create a formatted string representation of a table.

        Args:
            data (list of lists): The data to be displayed in the table.
            headers (list): The headers for the table columns.

        Returns:
            str: A string representation of the formatted table.
        """
        max_lengths = [max(len(str(item)) for item in col) for col in zip(*data, headers)]
        format_str = " | ".join(f"{{:<{length}}}" for length in max_lengths)
        table_content = format_str.format(*headers) + '\n'
        table_content += "-" * (sum(max_lengths) + 3 * (len(headers) - 1)) + '\n'
        for row in data:
            table_content += format_str.format(*[str(item) for item in row]) + '\n'
        return table_content.strip()

    def visualize_os_distribution(self, user_id, user_data):
        """
        Visualize the distribution of operating systems for a user.

        Args:
            user_id (str): The ID of the user.
            user_data (pandas.DataFrame): The data for the specified user.
        """
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
        """
        Visualize the approval rate of actions for a user.

        Args:
            user_id (str): The ID of the user.
            user_data (pandas.DataFrame): The data for the specified user.
        """
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
        """
        Visualize the distribution of phone versions for a user.

        Args:
            user_id (str): The ID of the user.
            user_data (pandas.DataFrame): The data for the specified user.
        """
        version_counts = user_data['phone_versions'].value_counts().sort_values(ascending=False)
        total = len(user_data)
        data = [[version, count, f"{count / total * 100:.2f}%"] for version, count in version_counts.items()]
        table_content = self._print_table(data, ["Phone Version", "Count", "Percentage"])
        print(self.create_boxed_output(table_content, f"Phone Versions Distribution for User {user_id}"))

    def visualize_session_duration(self, user_id, normal_data, anomalous_data):
        """
        Visualize the distribution of session durations for normal and anomalous data.

        Args:
            user_id (str): The ID of the user.
            normal_data (pandas.DataFrame): The normal data for the specified user.
            anomalous_data (pandas.DataFrame): The anomalous data for the specified user.
        """
        with self._suppress_warnings():
            plt.figure(figsize=(12, 6))
            sns.histplot(normal_data['session_duration'], kde=True, color="blue", label="Normal", alpha=0.5)
            sns.histplot(anomalous_data['session_duration'], kde=True, color="red", label="Anomalous", alpha=0.5)
            plt.title(f"Distribution of session_duration for User {user_id}")
            plt.xlabel("session_duration")
            plt.ylabel("Density")
            plt.legend()
            plt.tight_layout()
            plt.show()

    def visualize_feature_distributions(self, user_id, user_data):
        """
        Visualize the distribution of features for all data and anomalies.

        Args:
            user_id (str): The ID of the user.
            user_data (pandas.DataFrame): The data for the specified user.
        """
        with self._suppress_warnings():
            # Session Duration
            plt.figure(figsize=(12, 6))
            sns.histplot(data=user_data, x='session_duration', kde=True, color="blue", label="All Data")
            sns.histplot(data=user_data[user_data['prediction'] != 'valid'], x='session_duration',
                         kde=True, color="red", label="Anomalies")
            plt.title(f"Distribution of session_duration for User {user_id}")
            plt.xlabel("session_duration")
            plt.ylabel("Count")
            plt.legend()
            plt.tight_layout()
            plt.show()

            # Hour of Login
            plt.figure(figsize=(12, 6))
            sns.histplot(data=user_data, x='hour_of_timestamp', kde=True, color="blue", label="All Data")
            sns.histplot(data=user_data[user_data['prediction'] != 'valid'], x='hour_of_timestamp',
                         kde=True, color="red", label="Anomalies")
            plt.title(f"Distribution of login hour for User {user_id}")
            plt.xlabel("hour_of_timestamp")
            plt.ylabel("Count")
            plt.legend()
            plt.tight_layout()
            plt.show()

    def visualize_user_anomalies(self, user_id, user_data, features):
        """
        Visualize various aspects of user anomalies.

        This method calls other visualization methods to provide a comprehensive
        view of the user's data and potential anomalies.

        Args:
            user_id (str): The ID of the user.
            user_data (pandas.DataFrame): The data for the specified user.
            features (list): The features to be visualized.
        """
        with self._suppress_warnings():
            self.visualize_os_distribution(user_id, user_data)
            self.visualize_action_approval(user_id, user_data)
            self.visualize_phone_versions(user_id, user_data)

    def print_evaluation_results(self, results):
        """
        Print the evaluation results of the anomaly detection model.

        Args:
            results (dict): A dictionary containing the counts of different prediction categories.
        """
        content = "Evaluation Results:\n"
        total = sum(results.values())
        for category, count in results.items():
            percentage = (count / total) * 100 if total > 0 else 0
            content += f"{category.capitalize()}: {count} ({percentage:.2f}%)\n"
        print(self.create_boxed_output(content.strip(), "Model Evaluation"))

    def print_random_users(self, user_results):
        """
        Print information about randomly selected users from specific categories.

        This method selects up to 3 random users from the 'need_second_check' and 'invalid'
        categories and displays their results.

        Args:
            user_results (dict): A dictionary containing results for each user.
        """
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
