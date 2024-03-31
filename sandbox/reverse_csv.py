import pandas as pd

# import chardet
#
# def detect_encoding(file_path):
#     with open(file_path, 'rb') as f:
#         rawdata = f.read()
#         result = chardet.detect(rawdata)
#     return result['encoding']
#
# # Replace 'your_csv_file.csv' with the path to your CSV file
# csv_file = 'data.csv'
# encoding = detect_encoding(csv_file)
#
# print(f"The encoding of '{csv_file}' is: {encoding}")


df = pd.read_csv('data.csv')
# remove the first row
row_df = df.iloc[1:]  # check if needed !!
# reverse the order of the rows
rev_df = row_df.iloc[::-1]
# reset the index
re_ind_df = rev_df.reset_index(drop=True)
# # save the dataframe to a new csv file utf-8 sig encoding without the index column
re_ind_df.to_csv('rev_data.csv', index=False, encoding='utf-8-sig')
print(df)
