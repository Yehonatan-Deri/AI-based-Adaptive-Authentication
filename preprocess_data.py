# preprocess_data.py
# Description: This module is responsible for preprocessing the raw log data.
import re

import numpy as np
import pandas as pd


class Preprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        self.df['@timestamp'] = pd.to_datetime(self.df['@timestamp'].str.replace('@', '').str.strip(),
                                               format='%b %d, %Y %H:%M:%S.%f')
        # Remove rows where the message contains "I <query>"
        self.df = self.df[~self.df['message'].str.contains("I <query>", na=False)]
        # Reset the index after removing rows
        self.df = self.df.reset_index(drop=True)
        return self.df

    def parse_message(self):
        self.df['auth_id'] = self.df['message'].str.extract(r'auth\s+([A-Za-z0-9]+)')
        self.df['is_denied'] = self.df['message'].str.contains('denied', case=False, na=False)
        self.df['is_approved'] = self.df['message'].str.contains('approved', case=False, na=False)

        # Extract log_type (start/finish) from the message
        self.df['log_type'] = np.where(self.df['message'].str.contains('start', case=False, na=False), 'start',
                                       np.where(self.df['message'].str.contains('finish', case=False, na=False),
                                                'finish', np.nan))

    def fill_missing_values(self):
        # Fill missing user_id in finish actions by matching start actions based on auth_id
        start_actions = self.df[self.df['log_type'] == 'start'][['auth_id', 'user_id']]

        # Remove duplicates before merging
        start_actions = start_actions.drop_duplicates(subset=['auth_id'])

        self.df = pd.merge(self.df, start_actions, on='auth_id', suffixes=('', '_start'), how='left')
        self.df['user_id'] = self.df['user_id'].fillna(self.df['user_id_start'])
        self.df.drop(columns=['user_id_start'], inplace=True)

    def remove_incomplete_sessions(self):
        # Find start messages with no corresponding finish messages
        auth_ids_with_finish = set(self.df[self.df['log_type'] == 'finish']['auth_id'].unique())
        auth_ids_with_start = set(self.df[self.df['log_type'] == 'start']['auth_id'].unique())

        # Identify unmatched auth_ids
        unmatched_starts = auth_ids_with_start - auth_ids_with_finish
        unmatched_finishes = auth_ids_with_finish - auth_ids_with_start

        # Remove unmatched start messages
        self.df = self.df[~((self.df['log_type'] == 'start') & (self.df['auth_id'].isin(unmatched_starts)))]

        # Remove unmatched finish messages
        self.df = self.df[~((self.df['log_type'] == 'finish') & (self.df['auth_id'].isin(unmatched_finishes)))]

        # Check for duplicates and remove them
        auth_id_counts = self.df['auth_id'].value_counts()
        more_than_two_pairs = auth_id_counts[auth_id_counts > 2].index

        # Remove rows with auth_id having more than 2 occurrences
        self.df = self.df[~self.df['auth_id'].isin(more_than_two_pairs)]

    def drop_unwanted_columns(self):
        columns_to_keep = ['@timestamp', 'Android sum', 'auth_id', 'iOS sum', 'log_type', 'message', 'user_id']
        self.df = self.df[columns_to_keep]

    def extract_location_or_ip(self):
        ip_pattern = re.compile(r'ip: (null|\d{1,3}(?:\.\d{1,3}){3})')
        location_pattern = re.compile(r'at (.*? \(\d{1,3}(?:\.\d{1,3}){3}\))')

        def extract_info(message):
            # Check for location pattern
            location_match = location_pattern.search(message)
            if location_match:
                return location_match.group(1)

            # Check for IP pattern
            ip_match = ip_pattern.search(message)
            if ip_match:
                return f"ip: {ip_match.group(1)}"

            return None

        self.df['location_or_ip'] = self.df['message'].apply(extract_info)

    def extract_phone_type_and_version(self):
        def extract_info(message):
            if "iOS" in message:
                return "iOS"
            elif "Android" in message:
                return "Android"
            return None

        self.df['phone_type'] = self.df['message'].apply(extract_info)
        self.df['Android sum'] = np.where((self.df['log_type'] == 'finish') & (self.df['phone_type'] == 'Android'), 1,
                                          0)
        self.df['iOS sum'] = np.where((self.df['log_type'] == 'finish') & (self.df['phone_type'] == 'iOS'), 1, 0)

        def extract_versions(message):
            if "Android" in message:
                match = re.search(r'Android \d+; (.*?),', message)
                if match:
                    return match.group(1)
            elif "iOS" in message:
                match = re.search(r'iOS \d+(?:\.\d+)*;\s([^,)]+)', message)
                if match:
                    return match.group(1)
            return None

        self.df['phone_versions'] = self.df['message'].apply(extract_versions)

    def extract_hour_of_timestamp(self):
        self.df['hour_of_timestamp'] = self.df['@timestamp'].dt.hour

    def calculate_session_duration(self):
        # Calculate session duration for each auth_id
        start_times = self.df[self.df['log_type'] == 'start'][['auth_id', '@timestamp']].rename(
            columns={'@timestamp': 'start_time'})
        finish_times = self.df[self.df['log_type'] == 'finish'][['auth_id', '@timestamp']].rename(
            columns={'@timestamp': 'finish_time'})

        # Remove duplicates before merging
        start_times = start_times.drop_duplicates(subset=['auth_id'])
        finish_times = finish_times.drop_duplicates(subset=['auth_id'])

        session_times = pd.merge(start_times, finish_times, on='auth_id')
        session_times['session_duration'] = (
                session_times['finish_time'] - session_times['start_time']).dt.total_seconds()

        # Merge session duration back into the original dataframe
        self.df = pd.merge(self.df, session_times[['auth_id', 'session_duration']], on='auth_id', how='left')

        # Keep session duration only for finish messages
        self.df.loc[self.df['log_type'] != 'finish', 'session_duration'] = np.nan

    def combine_start_finish(self):
        # Extract start and finish rows
        start_rows = self.df[self.df['log_type'] == 'start']
        finish_rows = self.df[self.df['log_type'] == 'finish']

        # Merge start and finish rows on auth_id
        combined_df = pd.merge(start_rows, finish_rows, on='auth_id', suffixes=('_start', '_finish'))

        # Create combined columns
        combined_df['@timestamp'] = combined_df['@timestamp_start']
        combined_df['session_duration'] = combined_df['session_duration_finish']
        combined_df['Android sum'] = combined_df['Android sum_finish']
        combined_df['iOS sum'] = combined_df['iOS sum_finish']
        combined_df['is_denied'] = combined_df['is_denied_finish']
        combined_df['is_approved'] = combined_df['is_approved_finish']
        combined_df['location_or_ip'] = combined_df['location_or_ip_finish']
        combined_df['phone_type'] = combined_df['phone_type_finish']
        combined_df['phone_versions'] = combined_df['phone_versions_finish']
        combined_df['hour_of_timestamp'] = combined_df['hour_of_timestamp_start']

        # Select required columns
        final_columns = ['@timestamp', 'auth_id', 'user_id_start', 'Android sum', 'iOS sum', 'is_denied', 'is_approved',
                         'location_or_ip', 'phone_type', 'phone_versions', 'hour_of_timestamp', 'session_duration']
        self.df = combined_df[final_columns].copy()
        self.df.rename(columns={'user_id_start': 'user_id'}, inplace=True)

    def calculate_approval_stats(self):
        total_entries = len(self.df)
        approved_count = self.df['is_approved'].sum()
        denied_count = self.df['is_denied'].sum()

        approved_percentage = (approved_count / total_entries) * 100
        denied_percentage = (denied_count / total_entries) * 100

        print(f"Total entries: {total_entries}")
        print(f"Approved entries: {approved_count} ({approved_percentage:.2f}%)")
        print(f"Denied entries: {denied_count} ({denied_percentage:.2f}%)")

    def preprocess(self):
        self.load_data()
        self.drop_unwanted_columns()
        self.parse_message()
        self.remove_incomplete_sessions()
        self.extract_location_or_ip()
        self.extract_phone_type_and_version()
        self.extract_hour_of_timestamp()
        self.calculate_session_duration()
        self.fill_missing_values()
        self.combine_start_finish()
        # self.calculate_approval_stats()
        return self.df


if __name__ == "__main__":
    """
    main is for test purposes only
    """
    file_path = 'csv_dir/jerusalem_location_15.csv'
    # file_path = r"C:\Users\John's PC\Desktop\logs_csv.csv"
    preprocessor = Preprocessor(file_path)
    df = preprocessor.preprocess()
    print(df.head())
