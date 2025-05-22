import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import random
import numpy as np

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_loader(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./testground/data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    return trainloader

def train(model, trainloader, device):
    model = model.to(device,non_blocking=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    model.train()
    start_time = time.time()
    for epoch in range(1):  # 单 epoch 就足够用于测试
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device,non_blocking=True), labels.to(device,non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
    duration = time.time() - start_time
    return running_loss, duration

def test_mode(mode_name, deterministic, benchmark):
    print(f"\n--- Testing mode: {mode_name} ---")
    losses, times = [], []

    for run in range(3):  # 测试三次看一致性
        # 设置随机种子 & CuDNN 选项
        seed_all(42)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark

        # 数据 & 模型
        device = torch.device("cuda")
        loader = get_loader()
        model = torchvision.models.resnet18(num_classes=10)

        loss, t = train(model, loader, device)
        losses.append(loss)
        times.append(t)

        print(f"Run {run + 1}: Loss = {loss:.4f}, Time = {t:.2f}s")

    print(f"Avg Loss: {np.mean(losses):.4f}, Std Loss: {np.std(losses):.6f}")
    print(f"Avg Time: {np.mean(times):.2f}s, Std Time: {np.std(times):.2f}s")

if __name__ == "__main__":
    test_mode("TT", deterministic=True, benchmark=True)
    test_mode("TF", deterministic=True, benchmark=False)
    # test_mode("C: Mixed (benchmark on)", deterministic=False, benchmark=False)
