import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

# 1. 定义一个简单的模型 (与单 GPU 相同)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 2)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

def setup_distributed():
    # 从环境变量中获取分布式训练的参数
    # LOCAL_RANK: 当前进程在当前机器上的 GPU ID
    # RANK: 当前进程在整个集群中的全局 rank
    # WORLD_SIZE: 参与训练的总进程数
    # MASTER_ADDR: 主节点的 IP 地址
    # MASTER_PORT: 主节点的端口

    # 如果 LOCAL_RANK 环境变量不存在，则可能是单进程运行，或者未正确使用 torchrun
    if "LOCAL_RANK" not in os.environ:
        raise ValueError("警告: LOCAL_RANK 环境变量未设置。可能不是通过 torchrun 启动，或在单进程模式下运行。")
        # 尝试设置默认值，以便在非分布式模式下也能运行（用于调试）
        # os.environ["LOCAL_RANK"] = "0"
        # os.environ["RANK"] = "0"
        # os.environ["WORLD_SIZE"] = "1"
        # os.environ["MASTER_ADDR"] = "localhost"
        # os.environ["MASTER_PORT"] = "12355" # 随机端口

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # 设置 GPU 设备
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 初始化进程组
    dist.init_process_group(backend="nccl", # NCCL 是 NVIDIA GPU 的推荐后端
                            init_method="env://", # 从环境变量中获取连接信息
                            rank=rank,
                            world_size=world_size)
    
    return local_rank, rank, world_size, device

def cleanup_distributed():
    dist.destroy_process_group()

def train_ddp():
    local_rank, rank, world_size, device = setup_distributed()

    print(f"进程 {rank}/{world_size} (local_rank: {local_rank}) 在设备 {device} 上启动。")

    # 2. 准备数据
    data_size = 1000 * world_size # 模拟更大的数据集，每个进程分摊一部分
    features = torch.randn(data_size, 10)
    labels = torch.randint(0, 2, (data_size,))

    dataset = TensorDataset(features, labels)
    batch_size = 32

    # 使用 DistributedSampler 确保每个进程获取到数据集的不同子集
    # 它会根据 rank 和 world_size 自动分配样本索引
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    # 注意：在 DDP 中，DataLoader 不需要 shuffle=True，因为 sampler 已经处理了洗牌
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4)

    # 3. 实例化模型、损失函数和优化器
    model = SimpleModel().to(device)
    
    # 将模型封装在 DDP 中
    # device_ids 指定模型在哪个 GPU 上运行
    # output_device 指定 DDP 的输出在哪个 GPU 上
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. 训练循环
    num_epochs = 5
    print(f"进程 {rank}: 开始 DDP 训练...")
    for epoch in range(num_epochs):
        # 在每个 epoch 开始时，告诉 DistributedSampler 重新洗牌
        sampler.set_epoch(epoch)
        
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

        # 对所有进程的损失进行 All-reduce，以获取平均损失
        # 这确保了所有进程打印的平均损失是一致的
        # dist.reduce 是一个阻塞操作，等待所有进程完成
        avg_loss = torch.tensor(total_loss / len(dataloader), device=device)
        dist.reduce(avg_loss, dst=0) # 聚合到 rank 0 进程
        
        # 只有主进程打印日志
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss.item() / world_size:.4f}")

    print(f"进程 {rank}: DDP 训练完成。")
    
    # 5. 保存模型 (只由主进程保存)
    # DDP 模型保存的是 DDP 封装后的状态字典，需要通过 .module 获取原始模型的状态字典
    if rank == 0:
        torch.save(model.module.state_dict(), "simple_model_ddp.pth")
        print("模型已由主进程保存到 simple_model_ddp.pth")

    cleanup_distributed()

if __name__ == "__main__":
    train_ddp()