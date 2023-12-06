import os
import pandas as pd

working_dir = 'police_dataset'
db_info_fname = os.path.join(working_dir, 'DBinfo_20231122.xlsx')
save_fname = os.path.join(working_dir, 'label_table.csv')
query_colname = '유류족적이미지경로'
ref_colname = '등록이미지경로'

df = pd.read_excel(db_info_fname, sheet_name='Sheet1', engine='openpyxl')

label_table = {"query": [], "ref": []}
for idx, row in df.iterrows():
    query_img_name = row[query_colname].split('\\')[-1]
    ref_img_name = row[ref_colname].split('\\')[-1]
    
    label_table["query"].append(query_img_name)
    label_table["ref"].append(ref_img_name)

label_table_df = pd.DataFrame.from_dict(label_table, orient='columns')
label_table_df.to_csv(save_fname, index=False)