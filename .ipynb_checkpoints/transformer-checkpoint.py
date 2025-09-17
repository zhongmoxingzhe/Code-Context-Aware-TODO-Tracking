import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from modules import *

class DualPathTransformer(nn.Module):
    def __init__(self, hparams):
        super(DualPathTransformer, self).__init__()

        self.vocab_size = hparams.from_vocab_size
        self.emb_dim = hparams.emb_dim
        self.num_units = hparams.num_units if hparams.num_units else hparams.emb_dim
        self.encoder_num_layers = hparams.encoder_num_layers
        self.decoder_num_layers = hparams.decoder_num_layers
        self.num_heads = hparams.num_heads
        self.dropout_rate = hparams.dropout_rate

        self.word_embedding = nn.Embedding(self.vocab_size, self.num_units)

        # 1️⃣ 初始化后检查 NaN
        if torch.isnan(self.word_embedding.weight).any():
            print("🚨 初始化时 word_embedding.weight 包含 NaN！")

        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.num_units,
                nhead=self.num_heads,
                dim_feedforward=self.num_units * 2,
                dropout=self.dropout_rate,
                batch_first=True
            ) for _ in range(self.encoder_num_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=self.num_units,
                nhead=self.num_heads,
                dim_feedforward=self.num_units * 2,
                dropout=self.dropout_rate,
                batch_first=True
            ) for _ in range(self.decoder_num_layers)
        ])

        self.output_layer = nn.Linear(self.num_units, 2)

    def forward(self, input_ids, second_input_ids, input_positions, second_input_positions, 
            input_masks, second_input_masks):
        batch_size, seq_len = input_ids.shape
        _, second_seq_len = second_input_ids.shape
        # ===== 1. 输入校验 =====
        assert not torch.isnan(input_ids).any(), "输入包含 NaN！"
        assert input_ids.max() < self.vocab_size, f"输入超出词表大小！max_id={input_ids.max()}"
        
        # ===== 2. Embedding 保护 =====
        word_emb = self.word_embedding(input_ids)
        second_word_emb = self.word_embedding(second_input_ids)
        
        # 添加 LayerNorm 稳定 embedding
        word_emb = F.layer_norm(word_emb, (self.num_units,), 
                               weight=torch.ones(self.num_units).to(word_emb.device),
                               bias=torch.zeros(self.num_units).to(word_emb.device))
        
        # ===== 3. 位置编码保护 =====
        pos_emb = positional_encoding(input_positions, seq_len, self.num_units)
        pos_emb = pos_emb.clamp(-1e4, 1e4)  # 防极端值
        
        # ===== 4. 输入组合保护 =====
        inputs = (word_emb + pos_emb).float()
        second_inputs = (second_word_emb + pos_emb).float()
        
        # 添加残差连接前的缩放 (√d_model)
        scale = torch.sqrt(torch.tensor(self.num_units, dtype=torch.float32))
        inputs = inputs / scale
        second_inputs = second_inputs / scale
        
        # ===== 5. Encoder 层保护 =====
        for i, layer in enumerate(self.encoder_layers):
            second_inputs = layer(second_inputs, src_key_padding_mask=second_input_masks)
            
            # 添加层间监控
            if torch.isnan(second_inputs).any():
                torch.save({
                    'input': second_inputs,
                    'mask': second_input_masks
                }, f'nan_encoder_layer_{i}.pt')
                raise ValueError(f"Encoder Layer {i} 输出 NaN!")
                
            # 层间激活值裁剪
            second_inputs = torch.clamp(second_inputs, -1e4, 1e4)
        
        # ===== 6. Decoder 层保护 =====
        for i, layer in enumerate(self.decoder_layers):
            inputs = layer(inputs, second_inputs, 
                          tgt_key_padding_mask=input_masks,
                          memory_key_padding_mask=second_input_masks)
            
            # 强制数值范围
            inputs = inputs.float()  # 确保 FP32
            inputs = torch.clamp(inputs, -1e4, 1e4)
            
        # ===== 7. 输出保护 =====
        cls_representation = inputs[:, 0, :]
        logits = self.output_layer(cls_representation)
        
        # 输出值裁剪 (防止 softmax 溢出)
        logits = torch.clamp(logits, -50, 50)  # exp(-50) ≈ 1e-22
        
        return logits
