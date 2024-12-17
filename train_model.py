# train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os

# -----------------------------
# 1. 定义模型
# -----------------------------
class SimpleMNISTModel(nn.Module):
    def __init__(self):
        super(SimpleMNISTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

# -----------------------------
# 2. 训练模型
# -----------------------------
def train_model(num_samples=1000):
    transform = transforms.Compose([transforms.ToTensor()])
    full_train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    
    # 选取前 num_samples 个样本
    train_subset, _ = torch.utils.data.random_split(full_train_dataset, [num_samples, len(full_train_dataset) - num_samples])
    
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True)

    model = SimpleMNISTModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(3):  # 训练3个 Epoch
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # 保存模型
    os.makedirs("model", exist_ok=True)
    model_path = "model/mnist_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at: {model_path}")
    return model_path

if __name__ == "__main__":
    train_model()
