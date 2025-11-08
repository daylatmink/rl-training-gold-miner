import torch
import torch.nn as nn
from agent.Embedder import Embedder

class Qtention(nn.Module):
    """
    Q-head: 2-layer Transformer encoder (pre-norm), d_model=20, d_ff=24, nhead=4.
    Nhận đầu vào đã ở dạng token embed (B, L, 20).
    Tự thêm 2 action query tokens [ACT0], [ACT1] rồi trả về Q(s,0), Q(s,1).

    Tham số xấp xỉ: ~5,549 (không tính input embeddings/projections).
    """

    def __init__(self, d_model=20, d_ff=24, nhead=4, n_layers=2, dropout=0.0, max_items=30, n_actions=2):
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

        # Head chung d -> 1, áp vào mỗi ACT token để thu Q
        self.head = nn.Linear(d_model, 1)

        # Embedding
        self.embedder = Embedder(d_model=d_model, dropout=dropout, max_items=max_items)

        # Khởi tạo để action 0 có Q-value cao hơn action 1 LUÔN LUÔN
        # Strategy: Tạo gap lớn giữa act_emb[0] và act_emb[1]
        
        # 1. Init head với weights DƯƠNG (quan trọng!)
        with torch.no_grad():
            # Weights dương → Q sẽ tăng theo embedding value
            self.head.weight.data.uniform_(0.0, 0.02)  # Toàn dương
            self.head.bias.data.fill_(0.0)
        
        # 2. Init action embeddings với gap lớn
        with torch.no_grad():
            # ACT0: embedding dương LỚN → Q(act0) cao
            self.act_emb.data[0] = torch.ones(d_model) * 0.5  # Tăng từ 0.1 lên 0.5
            # ACT1: embedding âm NHỎ → Q(act1) thấp
            self.act_emb.data[1] = torch.ones(d_model) * (-0.5)  # Giảm từ -0.1 xuống -0.5
            
            # Thêm một chút noise nhỏ
            self.act_emb.data[0] += torch.randn(d_model) * 0.01
            self.act_emb.data[1] += torch.randn(d_model) * 0.01

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