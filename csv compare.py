import pandas as pd

csv1 = pd.read_csv('csv_dir/data_after_dropped.csv')
csv2 = pd.read_csv('csv_dir/log-staging-90.csv')

col1 = csv1.columns
col2 = csv2.columns

# find all columns that are in csv1 and in csv2
common_columns = [col for col in col1 if col in col2]
print("both: ", common_columns)
# find all columns that are only in csv1 but not in csv2
only_in_csv1 = [col for col in col1 if col not in col2]
print("only in first", only_in_csv1)
# find all columns that are only in csv2 but not in csv1
only_in_csv2 = [col for col in col2 if col not in col1]
print("only in second", only_in_csv2)