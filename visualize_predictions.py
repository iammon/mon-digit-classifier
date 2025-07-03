import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.font_manager import FontProperties

# ======= Font Setup =======
font_path = "/usr/share/fonts/truetype/noto/NotoSansMyanmar-Regular.ttf"
myanmar_font = FontProperties(fname=font_path)

# ======= Load Model =======
from models.net import Net

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16

model = Net().to(DEVICE)
model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
model.eval()

# ======= Load Test Data =======
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = datasets.ImageFolder(root="mon_digits/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
class_names = test_dataset.classes

# ======= Get Batch =======
data_iter = iter(test_loader)
images, labels = next(data_iter)
images, labels = images.to(DEVICE), labels.to(DEVICE)

# ======= Predict =======
with torch.no_grad():
    outputs = model(images)
    probs = torch.softmax(outputs, dim=1)
    preds = torch.argmax(probs, dim=1)

# ======= Plot Grid =======
os.makedirs("predictions", exist_ok=True)
fig, axs = plt.subplots(4, 4, figsize=(10, 10))
fig.suptitle("Prediction (Red) vs Ground Truth (Green)", fontsize=14)

for i in range(BATCH_SIZE):
    ax = axs[i // 4, i % 4]
    img = images[i].cpu() * 0.3081 + 0.1307
    img = img.squeeze().numpy()
    pred_idx = preds[i].item()
    true_idx = labels[i].item()
    confidence = probs[i][pred_idx].item()
    pred_label = class_names[pred_idx]
    true_label = class_names[true_idx]

    ax.imshow(img, cmap="gray")
    ax.axis("off")

    # Prediction in red (Mon digit)
    ax.text(
        2, 6,
        f"{pred_label}",
        color="red", fontsize=14, fontproperties=myanmar_font,
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3')
    )

    # Confidence score in red (ASCII)
    ax.text(
        26, 6,
        f"({confidence*100:.1f}%)",
        color="red", fontsize=10,
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3')
    )

    # Ground truth in green (Mon digit)
    ax.text(
        2, 28,
        f"{true_label}",
        color="green", fontsize=12, fontproperties=myanmar_font,
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3')
    )

plt.tight_layout()
plt.subplots_adjust(top=0.92)
save_path = "predictions/grid_predictions_confidence.png"
plt.savefig(save_path)
plt.close()
print(f"✅ Grid saved: {save_path}")
