import os

with open("test_feat_extractor/cropped_dataset.txt", 'r') as f1:
    with open("test_feat_extractor/cropped_dataset2.txt", 'w') as f2:
        for line in f1.readlines():
            f2.write(line.replace("/home/seankim/Documents/", "YOUR/PATH/TO/DATESET/"))

with open("test_feat_extractor/ref_dataset.txt", 'r') as f1:
    with open("test_feat_extractor/ref_dataset2.txt", 'w') as f2:
        for line in f1.readlines():
            f2.write(line.replace("/home/seankim/Documents/", "YOUR/PATH/TO/DATESET/"))