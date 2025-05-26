import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k  # Key的维度

    def forward(self, Q, K, V, mask=None):
        # Q, K, V形状: (batch_size, seq_len, d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # 可选：掩码（如Transformer解码器掩码未来位置）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)  # 沿最后一维Softmax
        output = torch.matmul(attn_weights, V)   # 加权求和
        
        return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model    # 输入维度（如512）
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性变换层
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 线性变换并分头 (batch_size, seq_len, num_heads, d_k)
        Q = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, d_k)
        
        # 拼接多头结果
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_O(output), attn_weights


def main():
    # 参数设置
    d_model = 64  # 输入维度
    num_heads = 4
    seq_len = 10  # 序列长度
    batch_size = 32

    # 随机生成输入（模拟batch_size=32，序列长度=10，维度=64）
    Q = torch.rand(batch_size, seq_len, d_model)
    K = V = torch.rand(batch_size, seq_len, d_model)  # 自注意力时Q=K=V

    # 多头注意力
    mha = MultiHeadAttention(d_model, num_heads)
    output, attn_weights = mha(Q, K, V)

    print("Output shape:", output.shape)          # (32, 10, 64)
    print("Attention weights shape:", attn_weights.shape)  # (32, 4, 10, 10)


if __name__ == '__main__':
    main()