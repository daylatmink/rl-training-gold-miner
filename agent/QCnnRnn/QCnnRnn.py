import torch
import torch.nn as nn
from agent.QCNN.Embedder import Embedder
from einops.layers.torch import Rearrange
import torch.nn.functional as F

class BackboneCnn(nn.Module):
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
    
    def forward(self, x, y, z):
        """
        x: [B, 10] or [B1, B2, ..., 10]
        y: [B, max_items, 23] or [B1, B2, ..., max_items, 23]
        z: [B, max_items] or [B1, B2, ..., max_items]
        
        Supports arbitrary batch dimensions (e.g., [B, T, ...] for sequences)
        """
        # Store original shape and flatten batch dimensions
        orig_shape = x.shape[:-1]  # All dimensions except last
        is_batched = len(orig_shape) > 1
        
        if is_batched:
            # Flatten: [B1, B2, ..., D] -> [B1*B2*..., D]
            x = x.reshape(-1, x.shape[-1])  # [B_flat, 10]
            y = y.reshape(-1, y.shape[-2], y.shape[-1])  # [B_flat, max_items, 23]
            z = z.reshape(-1, z.shape[-1])  # [B_flat, max_items]
        
        # 1) env part
        env_emb = self.env_extractor(x)  # [B_flat, d_model]

        # 2) item CNN backbone
        h = self.item_extractor(y)  # [B_flat, d_model, max_items]

        # 3) masked mean pooling theo chiều L (max_items)
        mask = z.unsqueeze(1).to(h.dtype)  # [B_flat, 1, max_items]

        # zero-out pad  
        h_masked = h * mask  # [B_flat, d_model, max_items]

        # tổng và chia cho số item thật
        denom = mask.sum(dim=-1).clamp(min=1.0)  # [B_flat, 1]
        item_emb = h_masked.sum(dim=-1) / denom  # [B_flat, d_model]

        # 4) ghép env + item
        combined = item_emb + env_emb  # [B_flat, d_model]
        
        # Reshape back to original batch dimensions
        if is_batched:
            combined = combined.reshape(*orig_shape, self.d_model)  # [B1, B2, ..., d_model]
        
        return combined

class MissStreakPenalty(nn.Module):
    def __init__(self, d_model, max_streak=15):
        super().__init__()
        self.bad_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.bad_emb, std=1)
        self.max_streak = max_streak
        
    def straight_forward(self, misses):
        """
        misses: [B, T] - số lần miss liên tiếp tại mỗi bước thời gian
        """
        streak = torch.clamp(misses.float(), 0.0, float(self.max_streak))  # [B, T]
        # normalize về [0,1]
        weight = (streak / self.max_streak).unsqueeze(-1)   # [B, T, 1]
        # nếu streak = 0 -> weight = 0, không phạt
        return weight * self.bad_emb               # [B, T, d_model]

    def forward(self, misses):
        """
        misses: [B, T] - số lần miss liên tiếp tại mỗi bước thời gian
        """
        miss_streak = self.miss_streak(misses)  # [B, T]
        # clip streak
        streak = torch.clamp(miss_streak.float(), 0.0, float(self.max_streak))
        # normalize về [0,1]
        weight = (streak / self.max_streak).unsqueeze(-1)   # [B, T, 1]
        
        # nếu streak = 0 -> weight = 0, không phạt
        return weight * self.bad_emb               # [B, T, d_model]
    
    def compute_miss_streak(self, miss: torch.Tensor) -> torch.Tensor:
        """
        miss: Tensor shape [B, T], 0 = hit, 1 = miss (bool/int/float đều được)
        return: Tensor [B, T], số lần miss liên tiếp, reset về 0 khi gặp hit.
        """

        # 1) Cumsum toàn cục theo thời gian
        c = miss.cumsum(dim=1)          # [B, T]

        # 2) Xác định vị trí "hit" (miss == 0)
        hit = (miss == 0).int()         # [B, T]

        # 3) Tại các vị trí hit, lấy giá trị cumsum làm mốc reset
        baseline = hit * c                  # [B, T]

        # 4) Lan truyền mốc reset về phía sau (segment-wise max)
        baseline, _ = baseline.cummax(dim=1)  # [B, T]

        # 5) Trừ mốc reset → cumsum theo từng đoạn, reset về 0 sau mỗi hit
        streak = c - baseline               # [B, T]

        return streak

    def miss_streak(self, masks: torch.Tensor) -> torch.Tensor:
        # s: [B, T, max_items] (bool hoặc 0/1)
        num_items = masks.sum(dim=-1)          # [B, T]  số item tại mỗi thời điểm

        # so sánh số item giữa t và t-1
        diff = num_items[:, 1:] - num_items[:, :-1]   # [B, T-1]

        # miss_step[b, t] = 1 nếu bước t (từ t -> t+1) là miss, ngược lại 0
        miss = (diff == 0).int()              # [B, T-1]
        return F.pad(self.compute_miss_streak(miss), (1, 0))

class QCnnRnn(nn.Module):
    """
    Q-Network kết hợp CNN backbone với GRU để nhớ lịch sử hành động.
    
    Architecture:
        x_gru = cnn(s_t) + emb(a_t) + g(r_t)
        
    Trong đó:
        - cnn(s_t): CNN backbone extract features từ state
        - emb(a_t): Action embedding (action vừa thực hiện)
        - g(r_t): Miss penalty dựa vào chuỗi miss liên tiếp
    """
    
    def __init__(self, d_model=24, n_actions=50, d_hidden=24, dropout=0.1, max_streak = 20, num_layers=1):
        super().__init__()
        self.d_model = d_model
        self.n_actions = n_actions
        self.d_hidden = d_hidden
        self.num_layers = num_layers
        
        # CNN Backbone
        self.backbone = BackboneCnn(d_model, n_actions, d_hidden, dropout)
        
        # Action embedding: emb(a_t)
        self.action_embedding = nn.Embedding(n_actions + 1, d_model)
        
        # Miss penalty network: g(r_t)
        # Input: miss_count (số lần miss liên tiếp)
        # Output: penalty vector [d_model]
        self.miss_penalty_net = MissStreakPenalty(d_model, max_streak=max_streak)
        # GRU layer
        # Input: x_gru = cnn(s_t) + emb(a_t) + g(r_t)  [d_model]
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Q-value head
        self.predictor = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, n_actions)
        )
    
    def get_gru_input(self, env_feats, item_feats, masks, actions):
        return self.backbone(env_feats, item_feats, masks) + self.action_embedding(actions) + self.miss_penalty_net(masks)
    
    def predict_step(self, env_feats, item_feats, masks, prev_action, miss_count, hidden):
        B, T, D = env_feats.shape
        assert B == 1 and T == 1, "predict_step chỉ hỗ trợ batch_size=1 và seq_len=1"
        
        x = self.backbone(env_feats, item_feats, masks) + self.action_embedding(prev_action) + self.miss_penalty_net.straight_forward(miss_count)  # [1, 1, d_model]
    
        o, new_hidden = self.gru(x, hidden)  # o: [1, 1, d_hidden]
        return self.predictor(o.squeeze(1)), new_hidden  # [1, n_actions], [1, num_layers, d_hidden]
    
    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.d_hidden,
                        dtype=torch.float32, device=device)

    
    