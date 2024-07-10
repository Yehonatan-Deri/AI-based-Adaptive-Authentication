# user_profiling.py
# Description: This module is responsible for profiling users based on their behavior.

import pandas as pd
import preprocess_data
import numpy as np
from datetime import datetime


class UserProfiler:
    def __init__(self, preprocessed_df):
        self.df = preprocessed_df

    def create_user_profile(self, user_id):
        user_data = self.df[self.df['user_id'] == user_id]

        if user_data.empty:
            raise ValueError(f"No data available for user_id: {user_id}")

        profile = {
            'user_id': user_id,
            'average_login_hour': user_data['hour_of_timestamp'].mean(),
            'login_hour_std_dev': user_data['hour_of_timestamp'].std(),
            'device_changes': user_data['phone_versions'].nunique(),
            'total_ios_actions': user_data['iOS sum'].sum(),
            'total_android_actions': user_data['Android sum'].sum(),
            'denial_rate': user_data['is_denied'].mean(),
            'average_session_duration': user_data['session_duration'].mean(),
            'session_duration_std_dev': user_data['session_duration'].std(),
            'locations': user_data['location_or_ip'].unique().tolist(),
            'device_list': user_data['phone_versions'].unique().tolist(),
        }

        return profile

    def create_all_user_profiles(self):
        profiles = []
        for user_id in self.df['user_id'].unique():
            try:
                profile = self.create_user_profile(user_id)
                profiles.append(profile)
            except ValueError:
                continue
        return pd.DataFrame(profiles)

    def analyze_login_times(self, user_id):
        user_data = self.df[self.df['user_id'] == user_id]
        login_hours = user_data['hour_of_timestamp']

        # Calculate the average and standard deviation
        avg_login_hour = login_hours.mean()
        std_dev_login_hour = login_hours.std()

        # Define time periods
        periods = {
            'morning': (5, 12),
            'afternoon': (12, 17),
            'evening': (17, 21),
            'night': (21, 5)
        }

        period_distribution = {}
        for period, (start, end) in periods.items():
            if start < end:
                period_distribution[period] = login_hours[(login_hours >= start) & (login_hours < end)].count()
            else:
                period_distribution[period] = login_hours[(login_hours >= start) | (login_hours < end)].count()

        return avg_login_hour, std_dev_login_hour, period_distribution

    def handle_device_changes(self, user_id):
        user_data = self.df[self.df['user_id'] == user_id]
        device_list = user_data['phone_versions'].unique().tolist()
        device_changes = len(device_list)

        # Assuming frequent changes within a short period are suspicious
        if device_changes > 1:
            device_change_timestamps = user_data[user_data['phone_versions'].duplicated(keep=False)]['@timestamp']
            device_change_intervals = device_change_timestamps.diff().dropna().dt.total_seconds()
            frequent_changes = (device_change_intervals < 3600).sum()  # Example: changes within an hour

            if frequent_changes > 0:
                return device_list, True  # Suspicious device changes
        return device_list, False  # Not suspicious

    def calculate_session_duration(self, user_id):
        user_data = self.df[self.df['user_id'] == user_id]
        avg_session_duration = user_data['session_duration'].mean()
        std_dev_session_duration = user_data['session_duration'].std()
        return avg_session_duration, std_dev_session_duration

    def calculate_denial_rate(self, user_id):
        user_data = self.df[self.df['user_id'] == user_id]
        denial_rate = user_data['is_denied'].mean()

        # Calculate trends over time (e.g., monthly)
        user_data = user_data.copy()  # Avoid SettingWithCopyWarning
        user_data.loc[:, 'month'] = user_data['@timestamp'].dt.to_period('M')
        monthly_denial_rate = user_data.groupby('month')['is_denied'].mean()

        # Convert monthly denial rate to a dictionary for better readability
        monthly_denial_rate_dict = monthly_denial_rate.to_dict()

        return denial_rate, monthly_denial_rate_dict

    def track_location_changes(self, user_id):
        user_data = self.df[self.df['user_id'] == user_id]
        location_list = user_data['location_or_ip'].unique().tolist()
        return location_list

    def validate_data(self):
        # Add validation rules as needed
        pass


if __name__ == "__main__":
    """
    main is for test purposes only
    """
    preprocessed_file_path = 'csv_dir/jerusalem_location_15.csv'
    # preprocessed_df = pd.read_csv(preprocessed_file_path)
    preprocessed_df = preprocess_data.Preprocessor(preprocessed_file_path).preprocess()
    profiler = UserProfiler(preprocessed_df)

    # region Debug test cases
    example_user_id = 'aca17b2f-0840-4e47-a24a-66d47f9f16d7'
    profile = profiler.create_user_profile(example_user_id)
    print(profile)
    avg_login_hour, std_dev_login_hour, period_distribution = profiler.analyze_login_times(example_user_id)
    print(
        f"Avg Login Hour: {avg_login_hour}, Std Dev Login Hour: {std_dev_login_hour}, Period Distribution: {period_distribution}")
    device_list, is_suspicious = profiler.handle_device_changes(example_user_id)
    print(f"Device List: {device_list}, Suspicious: {is_suspicious}")
    avg_session_duration, std_dev_session_duration = profiler.calculate_session_duration(example_user_id)
    print(f"Avg Session Duration: {avg_session_duration}, Std Dev Session Duration: {std_dev_session_duration}")
    denial_rate, monthly_denial_rate = profiler.calculate_denial_rate(example_user_id)
    print(f"Denial Rate: {denial_rate}, Monthly Denial Rate: {monthly_denial_rate}")
    location_list = profiler.track_location_changes(example_user_id)
    print(f"Location List: {location_list}")
    # endregion
