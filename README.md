# Shoeprint Image Retrieval

This repository contains codes for forensic shoeprint image retrieval. It mainly deals with experiments and implementations of relevant papers.


<br>

## test_feat_extractor
- This mainly discusses contents in 'test_feat_extractor' folder.

- For the first trial, Siamaese architecture(Resnet50 pretrained weights) is used as a baseline model.
It is tested on FID-300 dataset, which has 300 query(Scene of Crime) images and 1175 reference images.

- Contrastive loss function is used in the model and it has been implemented following the instructions in http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf.

- To extract descriptive features of the images, I used a model from 'deep-image-retrieval'(https://github.com/naver/deep-image-retrieval).
In the first step, I evaluated similarities between the queries and the references using Euclidean distance.


<br>

## File Structure

- For now, train step is not added in the code, so 'query/train' directory is not necessary to test the model.
- FID-300 reference and query images should be saved in the described structure. FID-300 dataset is an open dataset and can be downloaded at the following link: https://fid.dmi.unibas.ch/

- I used only 100 query images(00201.jpg~00300.jpg) for test, but included the entire(1175) reference images.

```
└── test_feat_extractor
    ├── label_table.csv
    ├── cropped_query_features.npy
    ├── original_query_features.npy
    ├── reference_features.npy
    └── test_naver_feat_extractor.py
    └── extract_features.py
    ├── query
    │   ├── test 
    │   │   ├── 00201.jpg
    │   │   ├── 00202.jpg
    │   │   ├── 00203.jpg
    │   |   ├── ...
    │   |   ├── 00300.jpg
    │   └── train
    │       ├── 00001.jpg
    │       ├── 00002.jpg
    │       ├── 00003.jpg
    │       ├── ...
    │       ├── 00200.jpg
    ├── ref
        ├── 00001.jpg
        ├── 00002.jpg
        ├── 00003.jpg
        ├── ...
        ├── 01175.jpg
```
<br>

## Feature Extraction
- I used a model from 'deep-image-retrieval'(https://github.com/naver/deep-image-retrieval) as a feature extractor of FID-300 images.
- 'extract_features.py' MUST BE run within 'deep-image-retrieval' github repo(https://github.com/naver/deep-image-retrieval), since it requires relevant modules and source codes. However I uploaded the separate code in this repo as well to clarify the feature extraction process.\
The run command and arg options to extract features are:

<br>

```
python extract_features.py  --dataset ImageList('original_dataset.txt')
                --checkpoint dirtorch/data/Resnet-101-AP-GeM.pt
                --output original_query_features.npy
                --whiten Landmarks_clean
                --whitenp 0.25 --gpu 0
```
<br>

## References

[1] **Learning with Average Precision: Training Image Retrieval with a Listwise Loss.**
Jerome Revaud, Jon Almazan, Rafael S. Rezende, Cesar de Souza, ICCV 2019 [\[PDF\]](https://arxiv.org/abs/1906.07589)

[2] FID-300 Dataset: (https://fid.dmi.unibas.ch/)

[3] **Dimensionality Reduction by Learning an Invariant Mapping.**
Hadsell, Raia, Sumit Chopra, and Yann LeCun, IEEE 2006 [\[PDF\]](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)

[4] **Masked Siamese Networks for Label-Efficient Learning.** 
Assran, M., Caron, M., Misra, I., Bojanowski, P., Bordes, F., Vincent, P., ... & Ballas, N. [\[PDF\]](https://arxiv.org/abs/2204.07141).