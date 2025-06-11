import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 1. Load your image
img_path = "image.png"  # <-- Change this to your image path
img = Image.open(img_path).convert("RGB")

# 2. Preprocess image for DeepLabV3
preprocess = transforms.Compose(
    [
        transforms.Resize(520),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0)  # [1, 3, H, W]

# 3. Load pre-trained DeepLabV3 model
model = deeplabv3_resnet101(pretrained=True)
model.eval()

# 4. Run the model
with torch.no_grad():
    output = model(input_batch)["out"][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()


# 5. Colorize the segmentation mask
def decode_segmap(mask):
    label_colors = np.array(
        [
            (0, 0, 0),  # 0=background
            (128, 0, 0),
            (0, 128, 0),
            (128, 128, 0),
            (0, 0, 128),
            (128, 0, 128),
            (0, 128, 128),
            (128, 128, 128),
            (64, 0, 0),
            (192, 0, 0),
            (64, 128, 0),
            (192, 128, 0),
            (64, 0, 128),
            (192, 0, 128),
            (64, 128, 128),
            (192, 128, 128),
            (0, 64, 0),
            (128, 64, 0),
            (0, 192, 0),
            (128, 192, 0),
            (0, 64, 128),
        ]
    )
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, len(label_colors)):
        idx = mask == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


# 6. Visualize
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Predicted Segmentation")
plt.imshow(decode_segmap(output_predictions))
plt.axis("off")

plt.show()
