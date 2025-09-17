import torch
from argparse import Namespace
from transformer import DualPathTransformer

# 假设已经有了模型的参数配置
hparams = Namespace(from_vocab_size=80000, emb_dim=256, num_units=256, encoder_num_layers=2, decoder_num_layers=4, num_heads=8, dropout_rate=0.15)

# 初始化模型
model = DualPathTransformer(hparams)

# 设置优化器和学习率
learning_rate = 0  # 你可以根据需求调整学习率
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 模拟测试输入
input_ids = torch.randint(0, 80000, (4, 512))  # 确保不超 vocab_size
second_input_ids = torch.randint(0, 80000, (4, 512))
input_positions = torch.arange(512).expand(4, -1)
second_input_positions = torch.arange(512).expand(4, -1)
input_masks = torch.ones(4, 512).bool()
second_input_masks = torch.ones(4, 512).bool()

# 运行 forward 传递
logits = model(input_ids, second_input_ids, input_positions, second_input_positions, input_masks, second_input_masks)