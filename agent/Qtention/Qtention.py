import torch
import torch.nn as nn
from agent.Qtention.Embedder import Embedder

class Qtention(nn.Module):

    def __init__(self, d_model=20, d_ff=24, nhead=4, n_layers=2, dropout=0.0, max_items=30, n_actions=50):
        super().__init__()
        assert d_model % nhead == 0, "d_model phải chia hết cho nhead"
        self.d_model = d_model
        self.n_actions = n_actions
        self.nhead = nhead
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # (B, L, d)
            norm_first=True,     # pre-norm
            
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers, enable_nested_tensor=False)

        # Hai action query tokens (đếm rất ít tham số; nếu muốn loại khỏi budget thì có thể freeze)
        self.act_emb = nn.Parameter(torch.zeros(n_actions, d_model))  # [n_actions, d]

        # Head MLP: d_model -> d_model -> 1 với GELU activation
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

        # Embedding
        self.embedder = Embedder(d_model=d_model, dropout=dropout, max_items=max_items)

        # Khởi tạo BALANCED - Để network TỰ HỌC thay vì bias
        # Đảm bảo TẤT CẢ n_actions được khởi tạo công bằng
        
        # 1. Init head MLP với Xavier (symmetric, không bias)
        with torch.no_grad():
            for m in self.head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.01)  # Small scale
                    nn.init.zeros_(m.bias)
        
        # 2. Init action embeddings SYMMETRIC - Scale theo sqrt(n_actions) cho stability
        with torch.no_grad():
            # Với nhiều actions, dùng std nhỏ hơn để tránh variance lớn
            std = 0.02 / (n_actions ** 0.5)  # Scale theo số action
            nn.init.normal_(self.act_emb, mean=0.0, std=std)

    def forward(self, type_ids, item_feats, mov_idx=None, mov_feats=None, mask=None):
        """
        Forward với preprocessed tensors.
        
        Args:
            type_ids: LongTensor [B, L] - Token type IDs
            item_feats: FloatTensor [B, L, 10] - Item features
            mov_idx: LongTensor [B, l] - Movement indices (optional)
            mov_feats: FloatTensor [B, l, 3] - Movement features (optional)
            mask: BoolTensor [B, L] - Padding mask (optional)
        
        Returns:
            Q logits, shape [B, n_actions]
        """
        x_tokens, mask = self.embedder(type_ids, item_feats, mov_idx, mov_feats, mask)  # [B, L, d], [B, L]
        B = x_tokens.size(0)
        act = self.act_emb.unsqueeze(0).expand(B, -1, -1)   # [B, n_actions, d]
        x = torch.cat([x_tokens, act], dim=1)               # [B, L+n_actions, d]
        
        # Extend mask cho action tokens (False vì không phải padding)
        act_mask = torch.zeros(B, self.n_actions, dtype=torch.bool, device=x.device)  # [B, n_actions]
        full_mask = torch.cat([mask, act_mask], dim=1)      # [B, L+n_actions]
        
        # Pass mask vào encoder: src_key_padding_mask yêu cầu True = ignore
        h = self.encoder(x, src_key_padding_mask=full_mask) # [B, L+n_actions, d]
        
        # Extract action token representations
        h_actions = h[:, -self.n_actions:, :]               # [B, n_actions, d]
        
        # Compute Q values
        Q = self.head(h_actions).squeeze(-1)                # [B, n_actions]
        
        return Q