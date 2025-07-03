import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from models.net import Net

# ======= Configuration =======
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======= Data Preparation =======
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure single channel
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.ImageFolder(root='mon_digits/train', transform=transform)
test_dataset = datasets.ImageFolder(root='mon_digits/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ======= Model, Loss, Optimizer =======
model = Net().to(DEVICE)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ======= Training Loop =======
def train():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx} | Loss: {loss.item():.4f}')
        print(f'>> Epoch {epoch+1} complete. Avg Loss: {total_loss / len(train_loader):.4f}')

# ======= Evaluation Function =======
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    print(f'>> Test Accuracy: {100. * correct / total:.2f}%')

# ======= Run Training & Test =======
if __name__ == "__main__":
    train()
    test()
    torch.save(model.state_dict(), "mon_model_v1.pth")
