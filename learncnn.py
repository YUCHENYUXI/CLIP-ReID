#%%
import torch
import torch.nn as nn


# %%
# 生成符合3D卷积规范的空数据张量
batch_size = 48
frames = 6  # RGB通道
rgbs = 3     # 帧数
height = 32
width = 32

# 生成递增数值序列 [4](@ref)
values = []
for b in range(batch_size):
    for f in range(frames):
        for r in range(rgbs):
            values.append(b+r)
values = torch.tensor(values,dtype=int)
values = values.view(batch_size, frames, rgbs, 1, 1)       # 扩展维度
#%%

#%%
# 创建数据张量 [6](@ref)
data = values * torch.ones(1, 1, 1, height, width)  # 广播机制填充维度
#%%
print(data.shape)  # torch.Size([48,3,6,55,55])

#%% check
print(data[0,1,0,:,:])

#%%
data = data.view(-1,3,height,width)
#%%
print(data.shape)
#%%
# 定义卷积核：2个输出通道，核大小为2x3x3
conv2d = nn.Conv2d(
    in_channels=3, out_channels=1, 
    kernel_size=3, stride=1, padding=1
)

# 前向传播
output = conv2d(data)
print(output.shape)
#%%
'''
batch_size = 48
frames = 6  # RGB通道
rgbs = 3     # 帧数
height = 32
width = 32
'''
out = output.view(batch_size,frames,1,height,width)
print(out.shape)
#%%
print(out[0,0,0,:,:])

# %%
res = out.mean(dim = 1)
# %%
