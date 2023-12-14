import os
import shutil
import pandas as pd
from PIL import Image
import time

import sys
import logging 

def create_label_table(db_info_fname, save_fname, query_colname, ref_colname, change_ref_name=True):
    
    df = pd.read_excel(db_info_fname, sheet_name='Sheet1', engine='openpyxl')
    label_table = {"query": [], "ref": []}
    
    ref_tag = 'registration'
    query_tag = 'Leaving'
    
    for idx, row in df.iterrows():
        # check if image file paths are correctly matched with categories(query vs. reference) in excel file.
        if query_tag not in row[query_colname]:
            raise ValueError(f"Query image path is not matched with the category: {idx}th row: {row[query_colname]}")
        
        if ref_tag not in row[ref_colname]:
            if change_ref_name:
                ref_img_name = row['등록밑창이미지이름'] + '.png'
            else:
                raise ValueError(f"Reference image path is not matched with the category: {idx}th row: {row[ref_colname]}")
        
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
def check_existing_data(label_table_dir, query_img_dir, ref_img_dir, save_fname):
    df = pd.read_csv(label_table_dir)
    
    query_imgs = os.listdir(query_img_dir)
    ref_imgs = os.listdir(ref_img_dir)
    
    missing_imgs = {"query": [], "ref": [], "missing":[]}
    
    for _, row in df.iterrows():
        query_img_name = row['query']
        ref_img_name = row['ref']
        
        if query_img_name not in query_imgs:
            missing_imgs["query"].append(query_img_name)
            missing_imgs["ref"].append(ref_img_name)
            missing_imgs["missing"].append("query")
        
        if ref_img_name not in ref_imgs:
            missing_imgs["query"].append(query_img_name)
            missing_imgs["ref"].append(ref_img_name)
            missing_imgs["missing"].append("ref")
    
    missing_imgs = pd.DataFrame.from_dict(missing_imgs, orient='columns')
    missing_imgs.to_csv(save_fname, index=True)
   
    return True


def unify_file_extension(imgs_dir, targ_ext, inplace=True):
    # Verify if targ_ext is a valid image extension.
    if targ_ext.lower() not in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif']:
        raise ValueError(f"Invalid image extension: {targ_ext}")
    
    # log in  sys.stdout
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
    logger = logging.getLogger("unify_file_extension")
    
    img_list = sorted(os.listdir(imgs_dir))
    for img_fname in img_list:
        cur_ext = img_fname.split('.')[-1]
        if cur_ext.lower() != targ_ext:
            raise ValueError(f"Invalid image extension: {cur_ext}")
        elif (cur_ext != targ_ext) and (cur_ext.lower() == targ_ext) :
            if inplace:
                new_img_fname = img_fname.rstrip(cur_ext) + targ_ext
                os.rename(os.path.join(imgs_dir, img_fname), os.path.join(imgs_dir, new_img_fname))
                logger.info(f"Renamed {img_fname} to {new_img_fname}.")
            else:
                # make a new file if inplace is True instead of renaming the file.
                shutil.copyfile(os.path.join(imgs_dir, img_fname), os.path.join(imgs_dir, img_fname.rstrip(cur_ext) + targ_ext))
                logger.info(f"Copied {img_fname} into {new_img_fname}.")
    return True


def missing_in_label_table(imgs_dir, label_table_dir, img_type):
    df = pd.read_csv(label_table_dir)
    imgs = os.listdir(imgs_dir)
    
    missing_imgs = []
    
    if img_type == 'query':
        for img in imgs:
            if img not in df['query'].values:
                missing_imgs.append(img)
    elif img_type == 'ref':
        for img in imgs:
            if img not in df['ref'].values:
                missing_imgs.append(img)
    else:
        raise ValueError(f"Invalid image type: {img_type}")
    
    return missing_imgs


if __name__ == '__main__':
    working_dir = 'police_dataset'
    db_info_fname = os.path.join(working_dir, 'DBinfo_20231122.xlsx')
    
    query_colname = '유류족적이미지경로'
    ref_colname = '등록이미지경로'
    
    time_stamp = time.strftime("%m%d_%H%M", time.localtime())
    lable_table_fname = os.path.join(working_dir, f'label_table_{time_stamp}.csv')
    
    # create_label_table(db_info_fname, lable_table_fname, query_colname, ref_colname)
    
    # label_table_dir = os.path.join(working_dir, 'your label table name')
    # open_pair_images(label_table_dir)
    
    # label_table_dir = 'police_dataset/label_table.csv'
    # check_existing_data(label_table_dir=label_table_dir, \
    #     query_img_dir='police_dataset/query', \
    #     ref_img_dir='police_dataset/ref', \
    #     save_fname=os.path.join('police_dataset', f'missing_imgs_{time_stamp}.csv'))
    
    lable_table_fname = 'police_dataset/label_table_1211_1124.csv'
    # missing_query_train = missing_in_label_table(imgs_dir='police_dataset/query/train', label_table_dir=lable_table_fname, img_type='query')
    # missing_query_test = missing_in_label_table(imgs_dir='police_dataset/query/test', label_table_dir=lable_table_fname, img_type='query')
    missing_query = missing_in_label_table(imgs_dir='police_dataset/query_v2', label_table_dir=lable_table_fname, img_type='query')
    missing_ref = missing_in_label_table(imgs_dir='police_dataset/ref_v2', label_table_dir=lable_table_fname, img_type='ref')
    
    with open(f'missing_query_imgs_v3.txt', 'w') as f:
        for img in missing_query:
            f.write(img + '\n')
    with open(f'missing_ref_imgs_v3.txt', 'w') as f:
        for img in missing_ref:
            f.write(img + '\n')