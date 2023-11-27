from transformers import ViTMSNModel, ViTMSNConfig, AutoImageProcessor, Trainer, TrainingArguments
from transformers import AutoFeatureExtractor, AutoModel

configuration = ViTMSNConfig()
model = ViTMSNModel(configuration)
model_configuration = model.config

model_ckpt = "facebook/vit-msn-large"

extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
model = ViTMSNModel.from_pretrained(model_ckpt)
hidden_dim = model.config.hidden_size

