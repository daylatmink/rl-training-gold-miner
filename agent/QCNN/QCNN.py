import torch
import torch.nn as nn
from agent.QCNN.Embedder import Embedder
from einops.layers.torch import Rearrange


class QCNN(nn.Module):
    def __init__(self, d_model = 24, n_actions = 50, d_hidden = 36, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_actions = n_actions
        self.d_hidden = d_hidden
        
        self.item_extractor = nn.Sequential(
            nn.Linear(23, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            Rearrange('b n d -> b d n'),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GroupNorm(num_groups=1, num_channels=d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_hidden, kernel_size=5, padding=2),
            nn.GroupNorm(num_groups=1, num_channels=d_hidden),
            nn.ReLU(),
            nn.Conv1d(d_hidden, d_model, kernel_size=5, padding=2),
            nn.GroupNorm(num_groups=1, num_channels=d_model),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            Rearrange('b d 1 -> b d'),
        )
        
        self.env_extractor = nn.Sequential(
            nn.Linear(10, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, n_actions)
        )
    
    def forward(self, x):
        item_features = self.item_extractor(x[1]) if x[1].shape[1] != 0 else torch.zeros((x[0].shape[0], self.d_model), device=x[0].device)
        env_features = self.env_extractor(x[0])
        combined_features = torch.cat((item_features, env_features), dim=1)
        return self.predictor(combined_features)