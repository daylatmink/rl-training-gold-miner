# player_manager.py
from state_exporter import get_game_state, save_state_to_json
import torch

class PlayerManager:
    def __init__(self, mining_agent, embedder, shopping_agent):
        self.current_player = 'human'  # or 'ai'
        self.mem = None
        self.mining_agent = mining_agent
        self.embedder = embedder
        self.shopping_agent = shopping_agent
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mining_agent.to(self.device)
        
    def get_mining_action(self, game_scene):
            state = get_game_state(game_scene)
            
            type_ids, item_feats, mov_idx, mov_feats = self.embedder.preprocess_state(state)
            
            # Convert to torch tensors và chuyển sang device
            type_ids_t = type_ids.unsqueeze(0).to(self.device)  # [1, L]
            item_feats_t = item_feats.unsqueeze(0).to(self.device)  # [1, L, d_feats]
            mov_idx_t = mov_idx.unsqueeze(0).to(self.device) if len(mov_idx) > 0 else None
            mov_feats_t = mov_feats.unsqueeze(0).to(self.device) if len(mov_feats) > 0 else None
            
            with torch.no_grad():
                q_values = self.mining_agent(type_ids_t, item_feats_t, mov_idx_t, mov_feats_t)  # [1, n_actions]
                action = q_values.argmax(dim=1).item()
                
            return action
    
    def ask(self, game_scene):
        if not self.mem:
            self.mem = self.get_mining_action(game_scene)
        return self.mem
    
    def clear_mem(self):
        self.mem = None
    
    def switch_player(self):
        self.clear_mem()
        if self.current_player == 'human':
            self.current_player = 'ai'
        else:
            self.current_player = 'human'

    def is_human(self):
        return self.current_player == 'human'

    def is_ai(self):
        return self.current_player == 'ai'
