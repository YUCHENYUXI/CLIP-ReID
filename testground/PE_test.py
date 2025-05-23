import torch
import numpy as np
import math

def calculate_positional_encoding(pos, d_model):
    """
    根据 Transformer 原始论文的公式计算单个位置的位置编码向量。
    """
    pe_vector = np.zeros(d_model)
    for i in range(d_model):
        if i % 2 == 0:  # 偶数维度使用 sin
            pe_vector[i] = math.sin(pos / (10000 ** (i / d_model)))
        else:  # 奇数维度使用 cos
            pe_vector[i] = math.cos(pos / (10000 ** ((i - 1) / d_model))) # 注意这里是 (i-1)/d_model，因为 i=2k+1, 2k=i-1
    return pe_vector

# 参数设置
d_model = 4
max_seq_len = 5 # 计算0到4的位置编码

print(f"--- 计算位置编码 (D_model={d_model}) ---")
for pos in range(max_seq_len):
    pe = calculate_positional_encoding(pos, d_model)
    print(f"PE_{pos} (Pos {pos}): {pe.round(3)}")

# 你的例子中的预期值 (四舍五入到3位小数)
expected_pe = {
    0: np.array([0.000, 1.000, 0.000, 1.000]),
    1: np.array([0.841, 0.540, 0.010, 0.999]),
    2: np.array([0.909, -0.416, 0.020, 0.999]),
    3: np.array([0.141, -0.990, 0.030, 0.999])
}

print("\n--- 与示例预期值对比 ---")
for pos in expected_pe:
    actual_pe = calculate_positional_encoding(pos, d_model)
    # 使用 np.allclose 来比较浮点数，允许一点误差
    if np.allclose(actual_pe, expected_pe[pos], atol=1e-3):
        print(f"PE_{pos}: 匹配成功!")
    else:
        print(f"PE_{pos}: **匹配失败!**")
        print(f"  计算值: {actual_pe.round(3)}")
        print(f"  预期值: {expected_pe[pos]}")

# 演示词嵌入与位置编码相加
print("\n--- 词嵌入 + 位置编码 ---")
# 假设的词嵌入 (用 PyTorch Tensor)
word_embeddings = torch.tensor([
    [0.1, 0.2, 0.3, 0.4],    # 我 (pos=0)
    [0.5, 0.6, 0.7, 0.8],    # 爱 (pos=1)
    [0.9, 1.0, 1.1, 1.2],    # 北京 (pos=2)
    [1.3, 1.4, 1.5, 1.6]     # 天安门 (pos=3)
], dtype=torch.float32)

# 生成所有需要的位置编码（PyTorch Tensor）
# 这里我们用 NumPy 计算后转为 Tensor，实际可以完全用 PyTorch 实现
positional_encodings_tensor = torch.zeros(word_embeddings.shape[0], d_model, dtype=torch.float32)
for pos in range(word_embeddings.shape[0]):
    positional_encodings_tensor[pos] = torch.tensor(calculate_positional_encoding(pos, d_model), dtype=torch.float32)

print("原始词嵌入:\n", word_embeddings)
print("\n对应位置编码:\n", positional_encodings_tensor.round(3))

# 直接相加
combined_embeddings = word_embeddings + positional_encodings_tensor
print("\n词嵌入 + 位置编码后的结果:\n", combined_embeddings.round(3))

# 与你例子中的预期值对比
expected_combined = torch.tensor([
    [0.100, 1.200, 0.300, 1.400],
    [1.341, 1.140, 0.710, 1.799],
    [1.809, 0.584, 1.120, 2.199],
    [1.441, 0.410, 1.530, 2.599]
], dtype=torch.float32)

if torch.allclose(combined_embeddings, expected_combined, atol=1e-3):
    print("\n词嵌入+位置编码结果：与示例预期值匹配成功！")
else:
    print("\n词嵌入+位置编码结果：**与示例预期值匹配失败！**")
    print("  计算值:\n", combined_embeddings.round(3))
    print("  预期值:\n", expected_combined.round(3))