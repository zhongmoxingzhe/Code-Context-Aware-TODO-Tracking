import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * torch.sum(1 + (recog_logvar - prior_logvar)
                           - torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
                           - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)), dim=1)
    return kld


def gelu(input_tensor):
    """Gaussian Error Linear Unit."""
    cdf = 0.5 * (1.0 + torch.erf(input_tensor / math.sqrt(2.0)))
    return input_tensor * cdf


def norm_log_likelihood(x, mu, logvar):
    return -0.5 * torch.sum(torch.log(torch.tensor(2 * np.pi)) + logvar +
                            torch.div(torch.pow((x - mu), 2), torch.exp(logvar)), dim=1)


def sample_gaussian(mu, logvar):
    epsilon = torch.randn_like(logvar)
    std = torch.exp(0.5 * logvar)
    z = mu + torch.mul(std, epsilon)
    return z


def normalize(inputs, epsilon=1e-8, scope="ln"):
    """Applies layer normalization."""
    if not hasattr(normalize, f'gamma_{scope}'):
        setattr(normalize, f'gamma_{scope}', nn.Parameter(torch.ones(inputs.size(-1))))
        setattr(normalize, f'beta_{scope}', nn.Parameter(torch.zeros(inputs.size(-1))))

    gamma = getattr(normalize, f'gamma_{scope}')
    beta = getattr(normalize, f'beta_{scope}')

    mean = torch.mean(inputs, dim=-1, keepdim=True)
    variance = torch.var(inputs, dim=-1, keepdim=True)
    normalized = (inputs - mean) / torch.sqrt(variance + epsilon)
    outputs = gamma * normalized + beta
    return outputs

class AttentionPooling(nn.Module):
    def __init__(self, num_units):
        super(AttentionPooling, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(num_units, 1))

    def forward(self, x):
        # x: (batch_size, seq_len, num_units)
        # 计算注意力权重
        attention_scores = torch.matmul(x, self.attention_weights)  # (batch_size, seq_len, 1)
        attention_scores = attention_scores.squeeze(-1)  # (batch_size, seq_len)

        # 归一化注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, seq_len)

        # 通过注意力权重加权平均
        weighted_sum = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)  # (batch_size, num_units)
        return weighted_sum
        
class MultiHeadAttention(nn.Module):
    def __init__(self, num_units=None, num_heads=8, dropout_rate=0):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.W_q = nn.Linear(num_units, num_units, bias=False)
        self.W_k = nn.Linear(num_units, num_units, bias=False)
        self.W_v = nn.Linear(num_units, num_units, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(num_units, num_units, bias=False)

    def forward(self, queries, keys, sequence_length, using_mask=False, mymasks=None):
        if self.num_units is None:
            self.num_units = queries.size(-1)

        batch_size = queries.size(0)

        # Linear projections
        Q = self.W_q(queries)
        K = self.W_k(keys)
        V = self.W_v(keys)

        # Split and concat
        Q_ = torch.cat(torch.chunk(Q, self.num_heads, dim=2), dim=0)
        K_ = torch.cat(torch.chunk(K, self.num_heads, dim=2), dim=0)
        V_ = torch.cat(torch.chunk(V, self.num_heads, dim=2), dim=0)

        # Multiplication
        outputs = torch.matmul(Q_, K_.transpose(1, 2))

        # Scale
        outputs = outputs / (K_.size(-1) ** 0.5)

        # Key Masking
        key_masks = torch.ones((batch_size, keys.size(1)), device=device)
        for i, l in enumerate(sequence_length):
            key_masks[i, l:] = 0
        key_masks = key_masks.repeat(self.num_heads, 1)
        key_masks = key_masks.unsqueeze(1).repeat(1, queries.size(1), 1)

        paddings = torch.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = torch.where(key_masks == 0, paddings, outputs)

        if using_mask:
            if mymasks.dim() == 2:
                mymask = mymasks.repeat(self.num_heads, 1, 1)
            else:
                mymask = mymasks
            outputs = torch.where(mymask == 0, paddings, outputs)

        outputs = F.softmax(outputs, dim=-1)

        # Query Masking
        query_masks = torch.ones((batch_size, queries.size(1)), device=device)
        for i, l in enumerate(sequence_length):
            if l < queries.size(1):
                query_masks[i, l:] = 0
        query_masks = query_masks.repeat(self.num_heads, 1)
        query_masks = query_masks.unsqueeze(-1).repeat(1, 1, keys.size(1))

        outputs = outputs * query_masks

        # Weighted sum
        outputs = torch.matmul(outputs, V_)

        # Restore shape
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=2)
        outputs = self.output_layer(outputs)
        outputs = self.dropout(outputs)
        return outputs


def positional_encoding(inputs, length, num_units, zero_pad=True, scale=True):
    device = inputs.device  # 获取 inputs 当前设备
    #print(f"Input tensor device: {device}")

    # 计算位置编码矩阵，并确保所有计算都在 `device` 上
    position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    #print(f"position shape: {position.shape}")  # 打印 position 形状 (length, 1)

    div_term = torch.exp(torch.arange(0, num_units, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / num_units))
    #print(f"div_term shape: {div_term.shape}")  # 打印 div_term 形状 (num_units//2,)

    # 直接在 `device` 上创建 `position_enc`
    position_enc = torch.zeros((length, num_units), dtype=torch.float32, device=device)
    #print(f"position_enc initial shape: {position_enc.shape}")  # (length, num_units)

    position_enc[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
    position_enc[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
    #print(f"position_enc after sin/cos shape: {position_enc.shape}")  # (length, num_units)

    # 检查计算后的 position_enc 设备
    #print(f"After encoding, position_enc device: {position_enc.device}")

    # 追加 zero padding（如果需要）
    if zero_pad:
        zero_tensor = torch.zeros((1, num_units), device=device)
        #print(f"Device of zero_tensor: {zero_tensor.device}")

        position_enc = torch.cat((zero_tensor, position_enc[1:, :]), dim=0)
        #print(f"position_enc after zero_pad shape: {position_enc.shape}")  # (length, num_units)

    # 最终检查
    #print(f"Final position_enc device: {position_enc.device}")

    # 获取 embedding
    lookup_table = position_enc  # 确保在同一设备上
    #print(f"lookup_table shape: {lookup_table.shape}")  # (length, num_units)

    #print(f"inputs shape: {inputs.shape}")  # 打印 inputs 形状

    outputs = F.embedding(inputs.long(), lookup_table)
    #print(f"outputs after embedding shape: {outputs.shape}")  # (batch_size, length, num_units)

    if scale:
        outputs = outputs * math.sqrt(num_units)
        #print(f"outputs after scaling shape: {outputs.shape}")  # (batch_size, length, num_units)

    return outputs


class WEncoderAttention(nn.Module):
    def __init__(self, num_units=None, num_heads=8, dropout_rate=0):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.Q = nn.Linear(num_units, num_units, bias=False)
        self.K = nn.Linear(num_units, num_units, bias=False)
        self.V = nn.Linear(num_units, num_units, bias=False)
        self.output_layer = nn.Linear(num_units, num_units, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, queries, keys, sequence_length, using_mask=False, mymasks=None):
        if self.num_units is None:
            self.num_units = queries.size(-1)

        batch_size = queries.size(0)

        Q = self.Q(queries)
        K = self.K(keys)
        V = self.V(keys)

        x = K * Q
        x = x.view(batch_size, x.size(1), self.num_heads, int(self.num_units / self.num_heads))
        outputs = torch.sum(x, dim=3).transpose(1, 2)
        outputs = outputs / (K.size(-1) ** 0.5)

        if using_mask:
            key_masks = mymasks
            key_masks = key_masks.repeat(1, self.num_heads).view(
                batch_size, self.num_heads, key_masks.size(1))
        else:
            key_masks = torch.zeros((batch_size, keys.size(1)), device=device)
            for i, l in enumerate(sequence_length):
                key_masks[i, :l] = 1
            key_masks = key_masks.repeat(1, self.num_heads).view(
                batch_size, self.num_heads, keys.size(1))

        paddings = torch.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = torch.where(key_masks == 0, paddings, outputs)
        outputs = F.softmax(outputs, dim=2)

        V_ = V.view(batch_size, V.size(1), self.num_heads, int(self.num_units / self.num_heads))
        V_ = V_.transpose(1, 2)
        weight = outputs
        outputs = torch.sum(V_ * outputs.unsqueeze(-1), dim=2)
        outputs = outputs.reshape(batch_size, -1)
        outputs = self.output_layer(outputs)
        outputs = self.dropout(outputs)
        return outputs, weight


def feedforward(inputs, num_units=[2048, 512], is_training=False, dropout_rate=0):
    """Point-wise feed forward net."""
    # Inner layer
    outputs = nn.Conv1d(inputs.size(-1), num_units[0], kernel_size=1)(inputs.transpose(1, 2))
    outputs = F.relu(outputs)

    # Readout layer
    outputs = nn.Conv1d(num_units[0], num_units[1], kernel_size=1)(outputs)
    outputs = outputs.transpose(1, 2)

    outputs = F.dropout(outputs, p=dropout_rate, training=is_training)
    return outputs
