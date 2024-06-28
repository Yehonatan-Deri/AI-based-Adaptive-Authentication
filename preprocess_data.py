# preprocess_data.py

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

    def handle_incomplete_sessions(self):
        incomplete_sessions = self.df.groupby('auth_id').filter(
            lambda x: (x['is_denied'].sum() == 0) & (x['is_approved'].sum() == 0))
        incomplete_auth_ids = incomplete_sessions['auth_id'].unique()
        self.df.loc[self.df['auth_id'].isin(incomplete_auth_ids), 'is_denied'] = True

    def drop_unwanted_columns(self):
        columns_to_keep = ['@timestamp', 'Android sum', 'auth_id', 'iOS sum', 'log_type', 'message', 'user_id']
        self.df = self.df[columns_to_keep]

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
