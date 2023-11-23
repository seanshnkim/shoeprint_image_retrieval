from transformers import ViTMSNModel, ViTMSNConfig, AutoImageProcessor
import torch
from PIL import Image
import cv2

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

# Example usage
image_path = "test_feat_extractor/query/train/00001.jpg"  # Replace with your image path
image_tensor = preprocess_image(image_path)
features = extract_features(image_tensor)

# features now contains the extracted feature vectors
print(features.shape)