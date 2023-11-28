import numpy as np

# cur = np.load("test_feat_extractor/np_features_msn/query/00001.npy")
# print(cur.shape)

# for i in range(300):
#     cur = np.load(f"test_feat_extractor/np_features_msn/query/{i+1:05d}.npy")
#     print(cur.shape)
#     squeezed = cur.squeeze(0)
#     np.save(f"test_feat_extractor/np_features_msn/query/{i+1:05d}.npy", squeezed)

for i in range(1175):
    cur = np.load(f"test_feat_extractor/np_features_msn/ref/{i+1:05d}.npy")
    print(cur.shape)
    squeezed = cur.squeeze(0)
    np.save(f"test_feat_extractor/np_features_msn/ref/{i+1:05d}.npy", squeezed)