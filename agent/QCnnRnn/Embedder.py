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
        'Gold': 0,
        'Rock': 1,
        'TNT': 2,
        'QuestionBag': 3,
        'Mole': 4,
        'MoleWithDiamond': 5,
        'Diamond': 6,
        'Skull': 7,
        'Bone': 8,
        'PAD': 9
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

    def __init__(self):
        super().__init__()

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
    
    @staticmethod
    def preprocess_state(state, max_items=30, return_batch = False):
        """
        STATIC METHOD: Preprocess state dict thành tensors với padding.
        
        Args:
            state: Dict từ get_game_state()
            max_items: Max number of items (sẽ pad đến số này)
            return_batch: Nếu True, thêm batch dimension
        
        Returns:
            env_feats: torch.Tensor [10] hoặc [1, 10] - Environment features
            items_feats: torch.Tensor [max_items, 23] hoặc [1, max_items, 23] - Item features (padded)
            mask: torch.Tensor [max_items] hoặc [1, max_items] - Mask (1=real item, 0=padding)
        """
        feats_list = []
        
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
        
        # 2. Item tokens
        rope_x = rope_state['end_position']['x']
        rope_y = rope_state['end_position']['y']
        
        # Sort items by angle phi (bearing) in ascending order
        items_sorted = sorted(state['items'][:max_items], 
                            key=lambda item: math.atan2(item['position']['y'] - rope_y, 
                                                       item['position']['x'] - rope_x))
        
        for idx, item in enumerate(items_sorted):
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
            
            # Get type_id
            if 'subtype' in item:
                token_name = item['subtype']
            else:
                token_name = item['type']
            if token_name not in Embedder.TOKEN_TYPES:
                raise ValueError(f"Unknown token type: {token_name}")
            
            type_onehot = [0] * len(Embedder.TOKEN_TYPES)
            type_onehot[Embedder.TOKEN_TYPES.get(token_name)] = 1
            item_feats.extend(type_onehot)
            
            # Check for movement features
            if 'ranges' in item and item['ranges'] is not None:
                dxl, dxr = item['ranges']
                dxl_rel = dxl - rope_x
                dxr_rel = dxr - rope_x
                dxl_norm = dxl_rel / Embedder.HALF_SCREEN_WIDTH
                dxr_norm = dxr_rel / Embedder.HALF_SCREEN_WIDTH
    
                item_feats.extend([dxl_norm, dxr_norm, float(item['direction'])])
            else:
                item_feats.extend([0, 0, 0])  # No movement features
            feats_list.append(item_feats)
        
        # Pad to max_items
        num_real_items = len(feats_list)
        num_pad = max_items - num_real_items
        
        # Create mask: 1 for real items, 0 for padding
        mask = [1] * num_real_items + [0] * num_pad
        
        # Pad features with zeros
        feat_dim = len(feats_list[0]) if feats_list else 23  # 10 item + 10 type + 3 movement
        for _ in range(num_pad):
            feats_list.append([0.0] * feat_dim)
        
        # Convert to tensors
        items_feats = torch.tensor(feats_list, dtype=torch.float32)
        env_feats = torch.tensor(env_feats, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        
        if not return_batch:
            return env_feats, items_feats, mask
        return (env_feats[torch.newaxis, :],
                items_feats[torch.newaxis, :, :],
                mask[torch.newaxis, :])