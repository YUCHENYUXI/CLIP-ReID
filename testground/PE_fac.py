import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimplifiedSelfAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.d_k = d_k # 头的维度

        # Q, K, V 的线性投影层
        # 这里为了简化，我们假设只有一个头，且 d_k == d_model
        # 如果是多头，这里会是 self.d_model, self.d_model*3
        self.w_q = nn.Linear(d_model, d_k, bias=False)
        self.w_k = nn.Linear(d_model, d_k, bias=False)
        self.w_v = nn.Linear(d_model, d_k, bias=False)

    def forward(self, x):
        # x 的形状: (batch_size, seq_len, d_model)

        # 1. 线性投影到 Q, K, V 空间
        # Q, K, V 的形状: (batch_size, seq_len, d_k)
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # 2. 计算注意力分数 (Scaled Dot-Product Attention)
        # qk_scores 形状: (batch_size, seq_len, seq_len)
        # torch.matmul(q, k.transpose(-2, -1)) 等同于 Q @ K.T
        qk_scores = torch.matmul(q, k.transpose(-2, -1))
        # 缩放
        scaled_qk_scores = qk_scores / math.sqrt(self.d_k)

        # 3. Softmax 归一化，得到注意力权重
        # attention_weights 形状: (batch_size, seq_len, seq_len)
        attention_weights = F.softmax(scaled_qk_scores, dim=-1)

        # 4. 加权求和 Value
        # output 形状: (batch_size, seq_len, d_k)
        output = torch.matmul(attention_weights, v)

        return output, attention_weights # 同时返回 output 和 weights 方便验证


# 参数设置
d_model = 4 # 词嵌入维度
d_k = 4     # 每个头的维度，这里简化为 d_k == d_model

# 初始化自注意力模块 (使用固定的随机种子，确保权重一致)
torch.manual_seed(42)
attention_model = SimplifiedSelfAttention(d_model, d_k)

# 原始序列的词嵌入 (与上面 PE 示例中的相同，这里是 (seq_len, d_model))
original_embeddings = torch.tensor([
    [0.1, 0.2, 0.3, 0.4],    # 0: 我
    [0.5, 0.6, 0.7, 0.8],    # 1: 爱
    [0.9, 1.0, 1.1, 1.2],    # 2: 北京
    [1.3, 1.4, 1.5, 1.6]     # 3: 天安门
], dtype=torch.float32)

# 将其 reshape 为 (batch_size, seq_len, d_model)
original_input = original_embeddings.unsqueeze(0) # batch_size = 1

print(f"--- 原始序列输入 (无位置编码) ---")
original_output, original_weights = attention_model(original_input)
print("原始序列的注意力权重 (softmax):\n", original_weights.squeeze(0))
print("\n原始序列的加权和输出:\n", original_output.squeeze(0))


# 打乱序列的顺序
# 例如，我们交换 "爱" (pos 1) 和 "北京" (pos 2) 的位置
shuffled_embeddings = torch.tensor([
    [0.1, 0.2, 0.3, 0.4],    # 0: 我
    [0.9, 1.0, 1.1, 1.2],    # 1: 北京 (原 pos 2)
    [0.5, 0.6, 0.7, 0.8],    # 2: 爱 (原 pos 1)
    [1.3, 1.4, 1.5, 1.6]     # 3: 天安门
], dtype=torch.float32)

shuffled_input = shuffled_embeddings.unsqueeze(0) # batch_size = 1

print(f"\n--- 打乱序列输入 (无位置编码) ---")
shuffled_output, shuffled_weights = attention_model(shuffled_input)
print("打乱序列的注意力权重 (softmax):\n", shuffled_weights.squeeze(0))
print("\n打乱序列的加权和输出:\n", shuffled_output.squeeze(0))

# 验证结果：
# 注意力权重矩阵和输出矩阵的行是对应的输入元素。
# 如果输入元素打乱了，那么输出矩阵的行也会相应打乱。
# 但对于相同的输入元素，它对应的输出向量应该是一样的。

# 比较原始序列中第1个元素 (爱) 的输出和打乱序列中第2个元素 (爱) 的输出
# 以及原始序列中第2个元素 (北京) 的输出和打乱序列中第1个元素 (北京) 的输出

print("\n--- 验证结果 ---")
# 验证原始序列中 pos=1 的输出 (爱) 与打乱序列中 pos=2 的输出 (爱) 是否一致
if torch.allclose(original_output[0, 1], shuffled_output[0, 2], atol=1e-4):
    print("‘爱’的输出向量在不同顺序下一致。")
else:
    print("‘爱’的输出向量在不同顺序下**不一致**。")
    print("原始序列中'爱'的输出:", original_output[0, 1])
    print("打乱序列中'爱'的输出:", shuffled_output[0, 2])


# 验证原始序列中 pos=2 的输出 (北京) 与打乱序列中 pos=1 的输出 (北京) 是否一致
if torch.allclose(original_output[0, 2], shuffled_output[0, 1], atol=1e-4):
    print("‘北京’的输出向量在不同顺序下一致。")
else:
    print("‘北京’的输出向量在不同顺序下**不一致**。")
    print("原始序列中'北京'的输出:", original_output[0, 2])
    print("打乱序列中'北京'的输出:", shuffled_output[0, 1])

# 验证原始序列中 pos=0 (我) 和 pos=3 (天安门) 的输出
# 由于这两个词的位置未变动，它们的输出应该分别与原始序列中对应位置的输出保持一致。
if torch.allclose(original_output[0, 0], shuffled_output[0, 0], atol=1e-4) and \
   torch.allclose(original_output[0, 3], shuffled_output[0, 3], atol=1e-4):
    print("‘我’和‘天安门’的输出向量在不同顺序下也一致（因为它们位置未变）。")
else:
    print("‘我’或‘天安门’的输出向量在不同顺序下**不一致**。")