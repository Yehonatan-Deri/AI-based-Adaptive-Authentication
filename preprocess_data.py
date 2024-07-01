# preprocess_data.py
import re

import pandas as pd
import numpy as np
from datetime import datetime

class Preprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        self.df['@timestamp'] = pd.to_datetime(self.df['@timestamp'].str.replace('@', '').str.strip(), format='%b %d, %Y %H:%M:%S.%f')
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
        start_actions = self.df[self.df['message'].str.contains('start', case=False, na=False)][['auth_id', 'user_id']]
        self.df = self.df.merge(start_actions, on='auth_id', suffixes=('', '_start'), how='left')
        self.df['user_id'] = self.df['user_id'].fillna(self.df['user_id_start'])
        self.df.drop(columns=['user_id_start'], inplace=True)

    def mark_incomplete_sessions_as_denied(self):
        # Find start messages with no corresponding finish messages
        auth_ids_with_finish = self.df[self.df['log_type'] == 'finish']['auth_id'].unique()
        unfinished_starts = self.df[(self.df['log_type'] == 'start') & (~self.df['auth_id'].isin(auth_ids_with_finish))]

        # Mark these start messages as denied
        self.df.loc[unfinished_starts.index, 'is_denied'] = True

        # Identify sessions without any 'denied' or 'approved' messages
        incomplete_sessions = self.df.groupby('auth_id').filter(
            lambda x: (x['is_denied'].sum() == 0) & (x['is_approved'].sum() == 0))
        incomplete_auth_ids = incomplete_sessions['auth_id'].unique()

        # Mark these sessions as denied
        self.df.loc[self.df['auth_id'].isin(incomplete_auth_ids), 'is_denied'] = True

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

    def preprocess(self):
        self.load_data()
        self.drop_unwanted_columns()
        self.parse_message()
        self.fill_missing_values()
        self.handle_incomplete_sessions()
        return self.df

if __name__ == "__main__":
    file_path = 'csv_dir/jerusalem_location_15.csv'
    preprocessor = Preprocessor(file_path)
    df = preprocessor.preprocess()
    print(df.head())
