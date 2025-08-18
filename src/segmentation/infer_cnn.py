import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from src.segmentation.unet import UNet

def segment_with_cnn(model_path, image_path, device="cpu"):
    device = torch.device(device)
    model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(tensor)[0, 0].cpu().numpy()

    mask = (pred > 0.5).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask, (image.width, image.height), interpolation=cv2.INTER_NEAREST)
    return mask_resized
