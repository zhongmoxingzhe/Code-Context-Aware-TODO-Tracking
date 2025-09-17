import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, input_dim=256, hidden_dims=[128, 64], output_dim=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[1], output_dim),
            #nn.Sigmoid()  # 输出二分类概率 移除看效果
        )

    def forward(self, x):
        return self.layers(x)
