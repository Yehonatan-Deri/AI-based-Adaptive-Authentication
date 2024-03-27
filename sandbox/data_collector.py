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


def extracr_data_and_merge_auths(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the data from the start and finish sections of the DataFrame and merge them
    :param df: DataFrame
    :return: Merged DataFrame
    """
    # region initialize data
    # Filter Start and Finish Events
    start_events = df[df['log_type'] == 'Starting auth']
    finish_events = df[df['log_type'] == 'Finishing auth']

    # reset the index of the dataframes
    start_events = start_events.reset_index(drop=True)
    finish_events = finish_events.reset_index(drop=True)

    start_events['@timestamp'] = pd.to_datetime(start_events['@timestamp'], format=datetime_format)
    finish_events['@timestamp'] = pd.to_datetime(finish_events['@timestamp'], format=datetime_format)
    # endregion

    # Extract the Hour of Login from the '@timestamp' column
    """
    Extract the hour of the day from the '@timestamp' column and store it in a new column 'hour_of_day_start'.
    """
    start_events['hour_of_day_start'] = start_events['@timestamp'].dt.hour

    # Merge Start and Finish Events based on 'auth_id'
    merged_data = start_events.merge(finish_events, on='auth_id', suffixes=('_start', '_finish'))

    # region Calculate time difference of start to finish related by auth_id
    """ 
        Calculate time difference between the start and finish of a auth by timestamp of events.
         subtracting the start timestamp from the finish timestamp.
    """
    merged_data['time_start_finish_by_auth_id'] = \
        ((merged_data['@timestamp_finish'] - merged_data['@timestamp_start']).dt.total_seconds())
    # endregion

    # region Calculate average login duration time per user
    """
        Calculate the average login time per user by grouping the data by 'user_id_start'
        and calculating the mean of the 'time_start_finish_by_auth_id' column.
    """
    avg_login_time_per_user = merged_data.groupby('user_id_start')['time_start_finish_by_auth_id'].mean().reset_index()
    avg_login_time_per_user.columns = ['user_id_start', 'avg_start_finish_login_time_seconds']

    merged_data = merged_data.merge(avg_login_time_per_user, on='user_id_start', how='left')
    # endregion

    # region Calculate Average Hour of Login per User
    """
        Calculate the average hour of login per user by grouping the data by 'user_id_start'
        and calculating the mean of the 'hour_of_day_start' column.
    """
    avg_hour_of_login = start_events.groupby('user_id')['hour_of_day_start'].mean().reset_index()
    avg_hour_of_login.columns = ['user_id_start', 'avg_hour_of_login']

    merged_data = merged_data.merge(avg_hour_of_login, on='user_id_start', how='left')
    # endregion

    return merged_data


def create_user_profiles(merged_data: pd.DataFrame,
                         save_csv: bool) -> pd.DataFrame:
    """
    Create user profiles from the merged data

    :param merged_data:
    :param save_csv:
    :return:
    """
    # Create a user profile dictionary for each user containing the following features:
    # - user_id
    # - most_common_device
    # - os_used
    # - avg_hour_of_login
    # - avg_start_finish_login_time_seconds
    # - avg_login_duration
    # - num_logins
    # and store them in a dictionary with the user_id as the key. then convert the dictionary to a DataFrame.
    # Finally, save the user profiles to a CSV file.

    # # Feature selection
    features = ['avg_start_finish_login_time_seconds', 'user_agent.device.name_start', 'user_agent.os.full_start',
                'user_id_start', 'avg_hour_of_login']
    X = merged_data[features]

    # # separate the data from merged_data for each user_id and store  them in a dictionary
    user_data = {}
    for user_id in merged_data['user_id_start'].unique():
        user_data[user_id] = merged_data[merged_data['user_id_start'] == user_id]

    user_profiles = {}  # Dictionary to store user profiles

    """
    from the start section of the merged_data, we can get the user_id, avg_start_finish_login_time_seconds,
    and avg_hour_of_login.
    from the finish section of the merged_data, we can get the most_common_device and os_used.
    """
    for user_id, data in user_data.items():
        avg_start_finish_login_duration = data['avg_start_finish_login_time_seconds'].values[0]
        avg_hour_of_login = data['avg_hour_of_login'].values[0]

        mode_values = data['user_agent.device.name_finish'].mode()
        if mode_values.empty:
            most_common_device = ''
        else:
            most_common_device = data['user_agent.device.name_finish'].mode().values[0]

        os_values = data['user_agent.os.full_finish'].mode()
        if os_values.empty:
            os_used = ''
        else:
            os_used = data['user_agent.os.full_finish'].mode().values[0]
        num_logins = data.shape[0]

        # sum the rows of the os_used columns
        os_sum = sum(data['iOS sum_finish'].tolist())
        android_sum = sum(data['Android sum_finish'].tolist())

        # Create a user profile dictionary
        profile = {
            'user_id': user_id,
            'avg_start_finish_login_time_seconds': avg_start_finish_login_duration,
            'avg_hour_of_login': avg_hour_of_login,  # 'avg_hour_of_login
            'most_common_device': most_common_device,
            'os_used': os_used,
            'login_with_android': android_sum,
            'login_with_iOS': os_sum,
            'num_logins': num_logins
            # Add more features as needed
        }

        # Store the profile in user_profiles dictionary
        user_profiles[user_id] = profile

    # Convert user_profiles dictionary to DataFrame for easy manipulation
    user_profiles_df = pd.DataFrame.from_dict(user_profiles, orient='index')

    # Optionally, you can save the user profiles to a CSV file
    if save_csv is not None:
        user_profiles_df.to_csv('user_profiles.csv', index=False)

    return user_profiles_df


if __name__ == "__main__":
    """
    main function for testing the functions
    """
    df = load_df(df_path='data_after_dropped.csv',
                 to_drop=False)

    merged_data = extracr_data_and_merge_auths(df)

    user_profiles_df = create_user_profiles(merged_data=merged_data, save_csv=False)

    # # X = df.drop('label', axis=1)
    # # y = df['label']
    # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # # model = xgb.XGBClassifier(use_label_encoder=False)
    # # model.fit(X_train, y_train)
    # # predictions = model.predict(X_test)
    # # accuracy = accuracy_score(y_test, predictions)
    # # print("Accuracy: %.2f%%" % (accuracy * 100.0))
