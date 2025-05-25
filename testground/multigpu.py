import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 2) # 假设是分类问题

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

def train_single_gpu():
    # 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        print("CUDA 不可用，使用 CPU 进行训练。")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0") # 默认使用第一张 GPU
        print(f"使用设备: {device}")

    # 2. 准备数据
    # 模拟一个小型数据集
    data_size = 1000
    features = torch.randn(data_size, 10)
    labels = torch.randint(0, 2, (data_size,)) # 0或1分类

    dataset = TensorDataset(features, labels)
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. 实例化模型、损失函数和优化器
    model = SimpleModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. 训练循环
    num_epochs = 5
    print("开始单 GPU 训练...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    print("单 GPU 训练完成。")
    # 5. 保存模型 (通常只在训练结束后保存)
    torch.save(model.state_dict(), "simple_model_single_gpu.pth")
    print("模型已保存到 simple_model_single_gpu.pth")

if __name__ == "__main__":
    train_single_gpu()