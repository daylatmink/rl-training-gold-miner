import torch
import torch.nn as nn
from agent.QCNN.Embedder import Embedder
from einops.layers.torch import Rearrange


class QCNN(nn.Module):
    def __init__(self, d_model = 24, n_actions = 50, d_hidden = 24, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_actions = n_actions
        self.d_hidden = d_hidden
        
        self.item_extractor = nn.Sequential(
            nn.Linear(23, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            Rearrange('b n d -> b d n'),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.GroupNorm(num_groups=1, num_channels=d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.GroupNorm(num_groups=1, num_channels=d_model),
            nn.ReLU(),
        )
        
        self.env_extractor = nn.Sequential(
            nn.Linear(10, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, n_actions)
        )
    
    def forward(self, x, y, z):
        """
        x: [B, 10]
        y: [B, max_items, 23]
        z: [B, max_items]
        """
        # 1) env part
        env_emb = self.env_extractor(x)           # [B, d_model] (ví dụ)

        # 2) item CNN backbone
        h = self.item_extractor(y)                      # [B, d_model, max_items]

        # 3) masked mean pooling theo chiều L (max_items)
        # z: [B, max_items] -> [B, 1, max_items]
        mask = z.unsqueeze(1).to(h.dtype)         # [B, 1, L]

        # zero-out pad  
        h_masked = h * mask                       # [B, d_model, L]

        # tổng và chia cho số item thật
        denom = mask.sum(dim=-1).clamp(min=1.0)   # [B, 1]
        item_emb = h_masked.sum(dim=-1) / denom   # [B, d_model]

        # 4) ghép env + item + head Q
        combined = item_emb + env_emb  # [B, d_model]
        q_values = self.predictor(combined)                # [B, n_actions]

        return q_values