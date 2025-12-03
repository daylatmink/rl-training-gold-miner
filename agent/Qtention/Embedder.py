import torch
import torch.nn as nn
import math
import numpy as np

class Embedder(nn.Module):
    """
    Embedder: Convert preprocessed state tensors sang token embeddings [B, L, d_model].
    
    OPTIMIZED VERSION: State được preprocess thành [item_feats, type_ids] trước khi forward.
    Forward chỉ cần: embedding[type_ids] + linear(item_feats) → Nhanh hơn nhiều!
    
    Token types:
    - ENV: Global + rope + miner state
    - Gold, Rock, TNT, QuestionBag: Basic items
    - Mole, MoleWithDiamond: Mole variants (dùng subtype)
    - Diamond, Skull, Bone: Other variants (dùng subtype)
    """
    
    # Token type mapping
    TOKEN_TYPES = {
        'ENV': 0,
        'Gold': 1,
        'Rock': 2,
        'TNT': 3,
        'QuestionBag': 4,
        'Mole': 5,
        'MoleWithDiamond': 6,
        'Diamond': 7,
        'Skull': 8,
        'Bone': 9,
        'PAD': 10
    }
    
    # Normalization constants (match với code trong project)
    SCREEN_WIDTH = 1280.0
    SCREEN_HEIGHT = 820.0
    HALF_SCREEN_WIDTH = SCREEN_WIDTH / 2.0
    HALF_SCREEN_HEIGHT = SCREEN_HEIGHT / 2.0
    MAX_SIZE = 200.0        # Gold/Rock có thể lên đến 200 pixels
    HALF_SIZE = 100.0
    MAX_POINT = 1000.0      # Diamond với gem_polish: 900, MoleWithDiamond: 902
    HALF_POINT = 500.0
    MAX_TIME = 60.0         # Mỗi level/episode có 60 giây
    HALF_TIME = 30.0          # Giữa level
    MAX_SCORE = 15000.0     # Score reset mỗi episode, max realistic ~10k-15k trong 60s
                            # (VD: 8 diamonds*900 + các items khác = ~7k-15k)
    MAX_GOAL = 100000.0     # Goal có thể lên rất cao ở level cao
    MAX_LEVEL = 100.0       # Không có giới hạn level (dùng 100 để normalize)
    MAX_DYNAMITE = 5.0      # define.py: MAX_DYNAMITE = 5
    HALF_DYNAMITE = 2.5
    MAX_ROPE_LENGTH = 1000.0  # model/GoldMiner.py: spaces.Box(0, 1000, ...)
    MAX_ROPE_SPEED = 300.0  # entities/rope.py: self.speed = 300 (default)
    MAX_ROPE_WEIGHT = 10.0  # Weight = size/30, max size=200 → max weight≈6.67
    HALF_ROPE_LENGTH = 500.0
    HALF_ROPE_WEIGHT = 5.0
    HALF_PULL_TIME = 5.0    # Estimated pulling time max ~10s
    MAX_DIAGONAL = math.sqrt(SCREEN_WIDTH**2 + SCREEN_HEIGHT**2)
    HALF_DIAGONAL = MAX_DIAGONAL / 2.0

    def __init__(self, d_model=20, max_items=30, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.max_items = max_items
        
        # Token type embedding
        self.token_type_embed = nn.Embedding(len(self.TOKEN_TYPES), d_model)
        
        # Feature projections
        self.env_proj = nn.Linear(10, d_model)  # ENV features
        self.item_proj = nn.Linear(10, d_model)  # Item features 
        self.movement_proj = nn.Linear(3, d_model)  # Movement features for Mole variants
        
        self.dropout = nn.Dropout(dropout)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Initialize
        nn.init.normal_(self.token_type_embed.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.env_proj.weight)
        nn.init.xavier_uniform_(self.item_proj.weight)
        nn.init.xavier_uniform_(self.movement_proj.weight)
        nn.init.zeros_(self.env_proj.bias)
        nn.init.zeros_(self.item_proj.bias)
        nn.init.zeros_(self.movement_proj.bias)
    
    def _normalize_env_features(self, state, device):
        """
        Extract và normalize ENV token features (10).
        Features: time_left, dynamite_count,
                      rope_sin, rope_cos, rope_length, rope_weight, rope_has_item,
                      rope_state_id (one-hot encoded)
        """
        
        global_state = state['global_state']
        rope_state = state['rope_state']
        
        # Rope state encoding: swinging=0, expanding=1, retracting=2
        rope_state_map = {'swinging': 0, 'expanding': 1, 'retracting': 2}
        rope_state_id = rope_state_map.get(rope_state['state'], 0)
        
        # Convert rope direction to sin/cos for better representation
        rope_angle_rad = math.radians(rope_state['direction'])
        rope_sin = math.sin(rope_angle_rad)
        rope_cos = math.cos(rope_angle_rad)
        
        features = [
            (global_state['time_left'] - self.HALF_TIME) / self.HALF_TIME,
            (global_state['dynamite_count'] - self.HALF_DYNAMITE) / self.HALF_DYNAMITE,
            rope_sin,  # sin(direction) ∈ [-1, 1]
            rope_cos,  # cos(direction) ∈ [-1, 1]
            (rope_state['length'] - self.HALF_ROPE_LENGTH) / self.HALF_ROPE_LENGTH,
            (rope_state['weight'] - self.HALF_ROPE_WEIGHT) / self.HALF_ROPE_WEIGHT,
            1.0 if rope_state['has_item'] else 0.0,
            1.0 if rope_state_id == 0 else 0.0,
            1.0 if rope_state_id == 1 else 0.0,
            1.0 if rope_state_id == 2 else 0.0,
        ]

        feat_embed = torch.tensor(features, dtype=torch.float32, device=device)
        type_embed = self.token_type_embed(torch.tensor(self.TOKEN_TYPES['ENV'], device=device))
        return self.env_proj(feat_embed) + type_embed

    def _normalize_item_features(self, item, rope_state, device):
        """
        Extract và normalize item features với tọa độ tương đối.
        Features (10): dx, dy, r, sin_phi, cos_phi, size, point, weight, pull_time, is_move
        
        dx, dy: Vị trí tương đối so với đầu dây (rope end point)
        r: Khoảng cách Euclidean từ rope end đến item
        sin_phi, cos_phi: Góc phương vị (bearing) từ rope end đến item
        weight: Cân nặng của item (size/30)
        pull_time: Ước lượng thời gian kéo về (giây)
        """
        # Get rope end position
        rope_x = rope_state['end_position']['x']
        rope_y = rope_state['end_position']['y']
        
        # Calculate relative position
        dx = item['position']['x'] - rope_x
        dy = item['position']['y'] - rope_y
        
        # Calculate distance and bearing
        r = math.sqrt(dx*dx + dy*dy)
        
        # Bearing angle (góc phương vị)
        phi = math.atan2(dy, dx)  # atan2(y, x) returns angle in radians
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        
        # Calculate weight and pulling time
        weight = item['size'] / 30.0  # Weight formula from rope.py
        rope_speed = rope_state['speed']  # Default 300
        buff_speed = rope_state['buff_speed']  # 1 or 2 (with strength)
        
        # Pulling time estimate: time = distance / speed
        # speed_retracting = (rope_speed * buff_speed) / weight
        if weight > 0:
            speed_retracting = (rope_speed * buff_speed) / weight
            pull_time = r / speed_retracting  # seconds
        else:
            pull_time = 0.0
        
        # Normalize features
        dx_norm = (dx) / self.HALF_SCREEN_WIDTH
        dy_norm = (dy - self.HALF_SCREEN_HEIGHT) / self.HALF_SCREEN_HEIGHT
        r_norm = (r - self.HALF_DIAGONAL) / self.HALF_DIAGONAL  # Max diagonal
        weight_norm = (weight - self.HALF_ROPE_WEIGHT) / self.HALF_ROPE_WEIGHT
        pull_time_norm = (pull_time - self.HALF_PULL_TIME) / self.HALF_PULL_TIME  # Normalize by max ~10 seconds
        
        features = [
            dx_norm,  # Relative x
            dy_norm,  # Relative y
            r_norm,   # Distance
            sin_phi,  # sin(bearing) ∈ [-1, 1]
            cos_phi,  # cos(bearing) ∈ [-1, 1]
            (item['size'] - self.HALF_SIZE) / self.HALF_SIZE,
            (item['point'] - self.HALF_POINT) / self.HALF_POINT,
            weight_norm,  # Weight (centered)
            pull_time_norm,  # Estimated pulling time
            1.0 if item['is_move'] else 0.0,
        ]
        
        feat_embed = torch.tensor(features, dtype=torch.float32, device=device)
        token_type = self._get_token_type(item)
        type_embed = self.token_type_embed(torch.tensor(self.TOKEN_TYPES[token_type], device=device))
        
        # Check if item has movement (Mole variants)
        if 'ranges' not in item or item['ranges'] is None:
            return self.item_proj(feat_embed) + type_embed

        # Add movement features for Mole (ranges already exist at this point)
        dxl, dxr = item['ranges']
        # Calculate relative positions to rope
        dxl_rel = dxl - rope_x
        dxr_rel = dxr - rope_x
        # Normalize
        dxl_norm = (dxl_rel) / self.HALF_SCREEN_WIDTH
        dxr_norm = (dxr_rel) / self.HALF_SCREEN_WIDTH
        mov_features = [
            dxl_norm,
            dxr_norm,
            float(item['direction'])  # -1 or 1
        ]
        mov_embed = torch.tensor(mov_features, dtype=torch.float32, device=device)
        return self.item_proj(feat_embed) + type_embed + self.movement_proj(mov_embed)

    def _get_token_type(self, item):
        """
        Get token type name từ item.
        Ưu tiên subtype, nếu không có thì dùng type.
        """
        if 'subtype' in item:
            token_name = item['subtype']
        else:
            token_name = item['type']
        
        # Fallback nếu không tìm thấy
        if token_name not in self.TOKEN_TYPES:
            token_name = 'PAD'
        
        return token_name
    
    def forward(self, type_ids, item_feats, mov_idx=None, mov_feats=None, mask=None):
        """
        OPTIMIZED: Convert preprocessed tensors sang token embeddings.
        
        Args:
            type_ids: LongTensor [B, L] - Token type IDs (0-10)
            item_feats: FloatTensor [B, L, 10] - Item features
            mov_idx: LongTensor [B, l] - Indices của items có movement (optional)
            mov_feats: FloatTensor [B, l, 3] - Movement features (dxl, dxr, direction) (optional)
            mask: BoolTensor [B, L] - True cho padding positions (optional)
        
        Returns:
            tokens: FloatTensor [B, L, d_model]
            mask: BoolTensor [B, L] - True cho padding positions
        """
        device = item_feats.device
        B, L = type_ids.shape
        
        # Lookup type embeddings: [B, L, d_model]
        type_embeds = self.token_type_embed(type_ids)
        
        # Project item features: [B, L, d_model]
        feat_embeds = self.item_proj(item_feats)
        
        # Add movement features if exists
        if mov_idx is not None and mov_feats is not None:
            # Project movement features: [B, l, d_model]
            mov_embeds = self.movement_proj(mov_feats)
            
            # VECTORIZED: Add mov_embeds vào đúng vị trí trong feat_embeds
            # Create batch indices: [B, l]
            batch_indices = torch.arange(B, device=device).unsqueeze(1).expand_as(mov_idx)
            
            # Flatten indices for scatter_add: [B*l]
            flat_batch_idx = batch_indices.reshape(-1)
            flat_mov_idx = mov_idx.reshape(-1)
            
            # Filter out invalid indices (-1 padding)
            valid_mask = flat_mov_idx >= 0
            flat_batch_idx = flat_batch_idx[valid_mask]
            flat_mov_idx = flat_mov_idx[valid_mask]
            flat_mov_embeds = mov_embeds.reshape(-1, self.d_model)[valid_mask]
            
            # Scatter add: feat_embeds[batch_idx, mov_idx] += mov_embeds
            # Reshape feat_embeds to [B*L, d_model] for easier indexing
            feat_embeds_flat = feat_embeds.view(-1, self.d_model)
            linear_indices = flat_batch_idx * L + flat_mov_idx
            feat_embeds_flat.index_add_(0, linear_indices, flat_mov_embeds)
            feat_embeds = feat_embeds_flat.view(B, L, self.d_model)
        
        # Combine: feat_embeds + type_embeds
        tokens = self.layer_norm(self.dropout(feat_embeds + type_embeds))
        
        # Create mask if not provided
        if mask is None:
            # Assume PAD type_id = 10
            mask = (type_ids == self.TOKEN_TYPES['PAD'])
        
        return tokens, mask
    
    def forward_legacy(self, state):
        """
        LEGACY: Convert state dict sang token embeddings (cho backward compatibility).
        
        Args:
            state: Dict từ get_game_state() hoặc json_to_state()
                   Có thể là single state hoặc batch of states
        
        Returns:
            tokens: FloatTensor [B, L, d_model] hoặc [L, d_model]
            mask: BoolTensor [B, L] - True cho padding positions (để attention bỏ qua)
                  L = 1 (ENV) + num_items
        """
        # Get device from model parameters
        device = next(self.parameters()).device
        
        # Check if batch or single
        is_batch = isinstance(state, list)
        if not is_batch:
            state = [state]
        
        batch_tokens = []
        seq_lengths = []  # Track actual length của mỗi sequence
        
        for s in state:
            tokens = []

            # 1. ENV token
            env_token = self._normalize_env_features(s, device)
            tokens.append(self.layer_norm(self.dropout(env_token)))

            # 2. Item tokens
            rope_state = s['rope_state']
            for item in s['items'][:self.max_items]:  # Limit max items
                item_embed = self._normalize_item_features(item, rope_state, device)
                tokens.append(self.layer_norm(self.dropout(item_embed)))
            
            seq_lengths.append(len(tokens))
            batch_tokens.append(torch.stack(tokens, dim=0))  # [L, d_model]
        
        # Pad to same length for batching
        max_len = max(seq_lengths)
        padded = []
        mask = []  # True for padding positions
        
        for t, length in zip(batch_tokens, seq_lengths):
            if t.size(0) < max_len:
                pad_size = max_len - t.size(0)
                pad_tensor = torch.zeros(pad_size, self.d_model, device=device)  # Create on device
                t = torch.cat([t, pad_tensor], dim=0)
                # Mask: False for real tokens, True for padding
                seq_mask = torch.cat([
                    torch.zeros(length, dtype=torch.bool, device=device),
                    torch.ones(pad_size, dtype=torch.bool, device=device)
                ])
            else:
                seq_mask = torch.zeros(max_len, dtype=torch.bool, device=device)
            
            padded.append(t)
            mask.append(seq_mask)
        
        tokens = torch.stack(padded, dim=0)  # [B, L, d_model]
        mask = torch.stack(mask, dim=0)  # [B, L]
        
        if not is_batch:
            return tokens.squeeze(0), mask.squeeze(0)  # [L, d_model], [L]
        return tokens, mask  # [B, L, d_model], [B, L]
    
    @staticmethod
    def preprocess_state(state, max_items=30, return_batch = False):
        """
        STATIC METHOD: Preprocess state dict thành 4 tensors.
        
        Returns 4 arrays:
        - type_ids [L]: Token type IDs
        - item_feats [L, 10]: Item features (tất cả items đều 10 features)
        - mov_idx [l]: Indices của items có movement
        - mov_feats [l, 3]: Movement features chỉ cho items có movement
        
        Args:
            state: Dict từ get_game_state()
            max_items: Max number of items
        
        Returns:
            type_ids: np.ndarray [L] - Token type IDs (int64)
            item_feats: np.ndarray [L, 10] - Item features (float32)
            mov_idx: np.ndarray [l] - Indices của items có movement (int64)
            mov_feats: np.ndarray [l, 3] - Movement features (float32)
        """
        feats_list = []
        type_ids_list = []
        mov_idx_list = []
        mov_feats_list = []
        
        global_state = state['global_state']
        rope_state = state['rope_state']
        
        # 1. ENV token
        rope_state_map = {'swinging': 0, 'expanding': 1, 'retracting': 2}
        rope_state_id = rope_state_map.get(rope_state['state'], 0)
        rope_angle_rad = math.radians(rope_state['direction'])
        rope_sin = math.sin(rope_angle_rad)
        rope_cos = math.cos(rope_angle_rad)
        
        env_feats = [
            (global_state['time_left'] - Embedder.HALF_TIME) / Embedder.HALF_TIME,
            (global_state['dynamite_count'] - Embedder.HALF_DYNAMITE) / Embedder.HALF_DYNAMITE,
            rope_sin,
            rope_cos,
            (rope_state['length'] - Embedder.HALF_ROPE_LENGTH) / Embedder.HALF_ROPE_LENGTH,
            (rope_state['weight'] - Embedder.HALF_ROPE_WEIGHT) / Embedder.HALF_ROPE_WEIGHT,
            1.0 if rope_state['has_item'] else 0.0,
            1.0 if rope_state_id == 0 else 0.0,
            1.0 if rope_state_id == 1 else 0.0,
            1.0 if rope_state_id == 2 else 0.0,
        ]
        feats_list.append(env_feats)
        type_ids_list.append(Embedder.TOKEN_TYPES['ENV'])
        
        # 2. Item tokens
        rope_x = rope_state['end_position']['x']
        rope_y = rope_state['end_position']['y']
        
        for idx, item in enumerate(state['items'][:max_items]):
            # Calculate relative position
            dx = item['position']['x'] - rope_x
            dy = item['position']['y'] - rope_y
            r = math.sqrt(dx*dx + dy*dy)
            phi = math.atan2(dy, dx)
            sin_phi = math.sin(phi)
            cos_phi = math.cos(phi)
            
            # Calculate weight and pulling time
            weight = item['size'] / 30.0
            rope_speed = rope_state['speed']
            buff_speed = rope_state['buff_speed']
            
            if weight > 0:
                speed_retracting = (rope_speed * buff_speed) / weight
                pull_time = r / speed_retracting
            else:
                pull_time = 0.0
            
            # Normalize
            dx_norm = dx / Embedder.HALF_SCREEN_WIDTH
            dy_norm = (dy - Embedder.HALF_SCREEN_HEIGHT) / Embedder.HALF_SCREEN_HEIGHT
            r_norm = (r - Embedder.HALF_DIAGONAL) / Embedder.HALF_DIAGONAL
            weight_norm = (weight - Embedder.HALF_ROPE_WEIGHT) / Embedder.HALF_ROPE_WEIGHT
            pull_time_norm = (pull_time - Embedder.HALF_PULL_TIME) / Embedder.HALF_PULL_TIME
            
            item_feats = [
                dx_norm,
                dy_norm,
                r_norm,
                sin_phi,
                cos_phi,
                (item['size'] - Embedder.HALF_SIZE) / Embedder.HALF_SIZE,
                (item['point'] - Embedder.HALF_POINT) / Embedder.HALF_POINT,
                weight_norm,
                pull_time_norm,
                1.0 if item['is_move'] else 0.0,
            ]
            
            feats_list.append(item_feats)
            
            # Get type_id
            if 'subtype' in item:
                token_name = item['subtype']
            else:
                token_name = item['type']
            if token_name not in Embedder.TOKEN_TYPES:
                token_name = 'PAD'
            type_ids_list.append(Embedder.TOKEN_TYPES[token_name])
            
            # Check for movement features
            if 'ranges' in item and item['ranges'] is not None:
                dxl, dxr = item['ranges']
                dxl_rel = dxl - rope_x
                dxr_rel = dxr - rope_x
                dxl_norm = dxl_rel / Embedder.HALF_SCREEN_WIDTH
                dxr_norm = dxr_rel / Embedder.HALF_SCREEN_WIDTH
                
                mov_idx_list.append(idx + 1)  # +1 vì ENV token ở index 0
                mov_feats_list.append([dxl_norm, dxr_norm, float(item['direction'])])
        
        # Convert to numpy arrays
        type_ids = torch.tensor(type_ids_list, dtype=torch.int64)
        item_feats = torch.tensor(feats_list, dtype=torch.float32)
        
        if len(mov_idx_list) > 0:
            mov_idx = torch.tensor(mov_idx_list, dtype=torch.int64)
            mov_feats = torch.tensor(mov_feats_list, dtype=torch.float32)
        else:
            # Empty arrays
            mov_idx = torch.tensor([], dtype=torch.int64)
            mov_feats = torch.tensor([], dtype=torch.float32).reshape(0, 3)
        
        if not return_batch:
            return type_ids, item_feats, mov_idx, mov_feats
        return (type_ids[torch.newaxis, :],
                item_feats[torch.newaxis, :, :],
                mov_idx[torch.newaxis, :],
                mov_feats[torch.newaxis, :, :])