import pandas as pd

TO_DROP = [
    '@version',
    '_index',
    '_score',
    '_type',
    'agent.type',
    'agent.version',
    'cloud.account.id',
    'cloud.availability_zone',
    'cloud.image.id',
    'cloud.instance.id',
    'cloud.machine.type',
    'cloud.provider',
    'cloud.region',
    'ecs.version',
    'host.architecture',
    'host.containerized',
    'host.id',
    'host.name',
    'host.os.codename',
    'host.os.family',
    'host.os.kernel',
    'host.os.name',
    'host.os.platform',
    'host.os.version',
    'journald.custom.selinux_context',
    'journald.custom.stream_id',
    'process.capabilites',
    'process.cmd',
    'process.executable',
    'process.name',
    'process.uid',
    'syslog.facility',
    'syslog.identifier',
    'syslog.priority',
    'systemd.cgroup',
    'systemd.slice',
    'systemd.transport',
    'systemd.unit',
    'tags'
]
datetime_format = "%b %d, %Y @ %H:%M:%S.%f"


def load_df(df_path: str,
            to_drop: bool = False) -> pd.DataFrame:
    """
    Load a DataFrame from a CSV file and drop the columns in the TO_DROP list
    :param df_path: path of the data frame
    :param to_drop: boolean to drop the columns
    :return: DataFrame with the columns dropped
    """
    df = pd.read_csv(df_path)
    if to_drop:
        # Filter the columns to drop only those that exist in the DataFrame
        columns_to_drop = [col for col in TO_DROP if col in df.columns]

        # Drop the columns and update the DataFrame
        df = df.drop(columns=columns_to_drop)
        df.to_csv('data_after_dropped.csv', index=False, encoding='utf-8-sig')
    return df


if __name__ == "__main__":
    df = load_df(df_path='data_after_dropped.csv',
                 to_drop=False)

    # region initialize data
    # Filter Start and Finish Events
    start_events = df[df['log_type'] == 'Starting auth']
    finish_events = df[df['log_type'] == 'Finishing auth']

    # make copies of the start and finish events
    start_events_copy = start_events.copy()
    finish_events_copy = finish_events.copy()

    # reset the index # TODO is it necessary?
    start_events = start_events.reset_index(drop=True)
    finish_events = finish_events.reset_index(drop=True)
    start_events_copy = start_events_copy.reset_index(drop=True)
    finish_events_copy = finish_events_copy.reset_index(drop=True)

    start_events_copy['@timestamp'] = pd.to_datetime(start_events_copy['@timestamp'], format=datetime_format)
    finish_events_copy['@timestamp'] = pd.to_datetime(finish_events_copy['@timestamp'], format=datetime_format)
    # endregion

    # Calculate Average Hour of Login
    start_events_copy['hour_of_day_start'] = start_events_copy['@timestamp'].dt.hour

    # Merge Start and Finish Events based on 'auth_id'
    merged_data = start_events_copy.merge(finish_events_copy, on='auth_id', suffixes=('_start', '_finish'))

    # region Calculate time difference of related events
    """ 
        Calculate time difference between the start and finish timestamp of events by
         subtracting the start timestamp from the finish timestamp.
    """
    merged_data['time_difference'] = ((merged_data['@timestamp_finish'] - merged_data['@timestamp_start']).
                                      dt.total_seconds())
    # endregion

    # region Calculate average login duration time per user
    """
        Calculate the average login time per user by grouping the data by 'user_id_start'
        and calculating the mean of the 'time_difference' column.
    """
    avg_login_time_per_user = merged_data.groupby('user_id_start')['time_difference'].mean().reset_index()
    avg_login_time_per_user.columns = ['user_id', 'avg_login_time_seconds']
    # endregion

    # region Calculate Average Hour of Login per User
    avg_hour_of_login = start_events_copy.groupby('user_id')['hour_of_day_start'].mean().reset_index()
    avg_hour_of_login.columns = ['user_id_start', 'avg_hour_of_login']

    merged_data = merged_data.merge(avg_hour_of_login, on='user_id_start', how='left')
    # endregion

    # Merge this information back into the main user_profiles DataFrame
    user_profiles_df = user_profiles_df.merge(avg_login_time_per_user, on='user_id', how='left')

    # endregion

    # Feature Engineering
    df['@timestamp'] = pd.to_datetime(df['@timestamp'], format=datetime_format)
    df['event.created'] = pd.to_datetime(df['event.created'], format=datetime_format)
    df['time_difference'] = (df['event.created'] - df['@timestamp']).dt.total_seconds()

    # Feature selection
    features = ['time_difference', 'user_agent.device.name', 'user_agent.os.full', 'user_id']
    X = df[features]

    # separate the data for each user_id and store  them in a dictionary
    user_data = {}
    for user_id in df['user_id'].unique():
        user_data[user_id] = df[df['user_id'] == user_id]

    user_profiles = {}  # Dictionary to store user profiles

    for user_id, data in user_data.items():
        # Calculate user-specific statistics
        avg_login_duration = data['time_difference'].mean()
        mode_values = data['user_agent.device.name'].mode()
        if mode_values.empty:
            most_common_device = ''
        else:
            most_common_device = data['user_agent.device.name'].mode().values[0]

        os_values = data['user_agent.os.full'].mode()
        if os_values.empty:
            os_used = ''
        else:
            os_used = data['user_agent.os.full'].mode().values[0]
        num_logins = data.shape[0]

        # Create a user profile dictionary
        profile = {
            'user_id': user_id,
            'avg_login_duration': avg_login_duration,
            'most_common_device': most_common_device,
            'os_used': os_used,
            'num_logins': num_logins
            # Add more features as needed
        }

        # Store the profile in user_profiles dictionary
        user_profiles[user_id] = profile

    # Convert user_profiles dictionary to DataFrame for easy manipulation
    user_profiles_df = pd.DataFrame.from_dict(user_profiles, orient='index')

    # # Optionally, you can save the user profiles to a CSV file
    # user_profiles_df.to_csv('user_profiles.csv', index=False)
    print(df)

    # X = df.drop('label', axis=1)
    # y = df['label']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # model = xgb.XGBClassifier(use_label_encoder=False)
    # model.fit(X_train, y_train)
    # predictions = model.predict(X_test)
    # accuracy = accuracy_score(y_test, predictions)
    # print("Accuracy: %.2f%%" % (accuracy * 100.0))
