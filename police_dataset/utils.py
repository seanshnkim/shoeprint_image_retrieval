import os
import pandas as pd
from PIL import Image
import time

def create_label_table(db_info_fname, save_fname, query_colname, ref_colname):
    df = pd.read_excel(db_info_fname, sheet_name='Sheet1', engine='openpyxl')

    label_table = {"query": [], "ref": []}
    for _, row in df.iterrows():
        query_img_name = row[query_colname].split('\\')[-1]
        ref_img_name = row[ref_colname].split('\\')[-1]
        
        label_table["query"].append(query_img_name)
        label_table["ref"].append(ref_img_name)

    label_table_df = pd.DataFrame.from_dict(label_table, orient='columns')
    label_table_df.to_csv(save_fname, index=False)


def open_pair_images(label_table_dir):
    df = pd.read_csv(label_table_dir)
    
    for _, row in df.iterrows():
        query = row['query']
        ref = row['ref']
        query_img = Image.open(os.path.join("police_dataset/query", query))
        ref_img = Image.open(os.path.join("police_dataset/ref", ref))
        
        # open images
        query_img.show(title=query)
        ref_img.show(title=ref)
        
        time.sleep(5)
        query_img.close()
        ref_img.close()


# check if the image files in label_table actually exist.
def check_valid_data(label_table_dir, query_img_dir, ref_img_dir):
    df = pd.read_csv(label_table_dir)
    
    query_imgs = os.listdir(query_img_dir)
    ref_imgs = os.listdir(ref_img_dir)
    
    missing_query_imgs = []
    missing_ref_imgs = []
    
    for _, row in df.iterrows():
        query_img_name = row['query']
        ref_img_name = row['ref']
        
        if query_img_name not in query_imgs:
            # print(f"This query img does NOT exist in dataset: {query_img_name}")
            missing_query_imgs.append(query_img_name)
        
        if ref_img_name not in ref_imgs:
            # print(f"This ref img does NOT exist in dataset: {ref_img_name}")
            missing_ref_imgs.append(ref_img_name)
    
    return missing_query_imgs, missing_ref_imgs


def unify_file_extension(img_dir, targ_ext):
    img_list = sorted(os.listdir(img_dir))
    for img in img_list:
        cur_ext = img.split('.')[-1]
        if cur_ext != targ_ext:
            os.rename(os.path.join(img_dir, img), os.path.join(img_dir, img.rstrip(img.split('.')[-1]) + targ_ext))
            
    return os.listdir(img_dir)


if __name__ == '__main__':
    working_dir = 'police_dataset'
    db_info_fname = os.path.join(working_dir, 'DBinfo_20231122.xlsx')
    save_fname = os.path.join(working_dir, 'label_table.csv')
    query_colname = '유류족적이미지경로'
    ref_colname = '등록이미지경로'
    
    # create_label_table(db_info_fname, save_fname, query_colname, ref_colname)
    
    label_table_dir = os.path.join(working_dir, 'label_table.csv')
    # open_pair_images(label_table_dir)
    
    # missing_queries, missing_refs = check_valid_data(label_table_dir, 'police_dataset/query', 'police_dataset/ref')
    
    # with open('police_dataset/missing_queries.txt', 'w') as f:
    #     for query in missing_queries:
    #         f.write(query + '\n')
    # with open('police_dataset/missing_refs.txt', 'w') as f:
    #     for ref in missing_refs:
    #         f.write(ref + '\n')
    
    unify_file_extension('police_dataset/ref', 'png')