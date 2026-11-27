import torch
from model import SmallCNN

model = SmallCNN()
images = torch.randn(64, 3, 32, 32)
logits = model(images)
print(logits.shape)