import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from models.net import Net

# ======= Setup =======
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16

# ======= Load Model =======
model = Net().to(DEVICE)
model.load_state_dict(torch.load("model.pth", map_location=DEVICE))  # You need to save model first!
model.eval()

# ======= Load Test Data =======
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ======= Visualization Function =======
def imshow(img, pred, true, i):
    img = img * 0.3081 + 0.1307  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap="gray")
    plt.title(f"Predicted: {pred} | Actual: {true}")
    plt.axis("off")
    plt.savefig(f"output_{i}_pred{pred}_true{true}.png")
    plt.close()

# ======= Get a Batch and Predict =======
data_iter = iter(test_loader)
images, labels = next(data_iter)
images, labels = images.to(DEVICE), labels.to(DEVICE)

with torch.no_grad():
    outputs = model(images)
    preds = torch.argmax(outputs, dim=1)

# ======= Display =======
for i in range(BATCH_SIZE):
    img = images[i].cpu()
    true = labels[i].item()
    pred = preds[i].item()
    imshow(img, pred, true, i)
