from transformers import ViTMSNModel, ViTMSNConfig, AutoImageProcessor
import torch
from PIL import Image
import numpy as np
import os

# Initializing a ViT MSN vit-msn-base style configuration
configuration = ViTMSNConfig()

# Initializing a model from the vit-msn-base style configuration
model = ViTMSNModel(configuration)

# Accessing the model configuration
model_configuration = model.config

image_processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-large")
model = ViTMSNModel.from_pretrained("facebook/vit-msn-large")

def preprocess_image(image_path):
    # Open the image file
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Process the image for the model
    processed_image = image_processor(images=image, return_tensors="pt").pixel_values
    return processed_image

def extract_features(image_tensor):
    # Pass the image tensor through the model
    with torch.no_grad():  # We don't need gradients for feature extraction
        outputs = model(pixel_values=image_tensor)
    # Extract the last hidden states as features
    features = outputs.last_hidden_state
    return features


# For query images, we need to extract features for each image in the query set
NUM_QUERY = 300
for i in range(1, NUM_QUERY+1):
    save_fname = os.path.join("test_feat_extractor", "np_features_msn", "query", f"{i:05d}.npy")
    if os.path.exists(save_fname):
        continue
    
    if i <= 200:
        image_path = os.path.join("test_feat_extractor", "query", "train", f"{i:05d}.jpg")
    else:
        image_path = os.path.join("test_feat_extractor", "query", "test", f"{i:05d}.jpg")
        
    image_tensor = preprocess_image(image_path)
    features = extract_features(image_tensor)

    # Save features to numpy file
    features = features.numpy()
    np.save(save_fname, features)


# For reference images, we need to extract features for each image in the reference set
NUM_REF = 1175
for i in range(1, NUM_REF+1):
    save_fname = os.path.join("test_feat_extractor", "np_features_msn", "ref", f"{i:05d}.npy")
    if os.path.exists(save_fname):
        continue
    
    image_path = os.path.join("test_feat_extractor", "ref", f"{i:05d}.png")
    image_tensor = preprocess_image(image_path)
    features = extract_features(image_tensor)

    # Save features to numpy file
    features = features.numpy()
    np.save(save_fname, features)