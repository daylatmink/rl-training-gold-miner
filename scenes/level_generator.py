import random
import json
from typing import List, Dict, Any

class LevelGenerator:
    def __init__(self, config_path="level_config.json"):
        self.config = self.load_config(config_path)
        
    def load_config(self, config_path):
        return {
            "level_types": ["LevelA", "LevelB", "LevelC", "LevelD", "LevelE"],
            
            # 🎯 CONFIG THEO TỪNG DIFFICULTY - HOÀN TOÀN RIÊNG BIỆT
            "entity_types_by_difficulty": {
                "train": {
                    # 🎯 TRAIN: CHỈ 1 ITEM NGẪU NHIÊN - Để agent học cơ bản nhất
                    "MiniRock": {"weight": 0.2, "min": 0, "max": 1},
                    "NormalRock": {"weight": 0.2, "min": 0, "max": 1},
                    "BigRock": {"weight": 0.1, "min": 0, "max": 1},
                    "MiniGold": {"weight": 0.5, "min": 0, "max": 1},
                    "NormalGold": {"weight": 0.5, "min": 0, "max": 1},
                    "BigGold": {"weight": 0.3, "min": 0, "max": 1},
                    "Diamond": {"weight": 0.2, "min": 0, "max": 1},
                    "QuestionBag": {"weight": 0.1, "min": 0, "max": 1},
                    "Mole": {"weight": 0.0, "min": 0, "max": 0},
                    "MoleWithDiamond": {"weight": 0.0, "min": 0, "max": 0},
                    "TNT": {"weight": 0.0, "min": 0, "max": 0},
                    "Skull": {"weight": 0.0, "min": 0, "max": 0},
                    "Bone": {"weight": 0.0, "min": 0, "max": 0}
                },
                "easy": {
                    # 🎯 EASY: Mức độ vừa phải để học - chỉ vàng + đá cơ bản (GIẢM 40%)
                    "MiniRock": {"weight": 0.4, "min": 1, "max": 2},
                    "NormalRock": {"weight": 0.3, "min": 0, "max": 2},
                    "BigRock": {"weight": 0.0, "min": 0, "max": 0},
                    "MiniGold": {"weight": 0.8, "min": 2, "max": 4},
                    "NormalGold": {"weight": 0.8, "min": 2, "max": 3},
                    "BigGold": {"weight": 0.8, "min": 1, "max": 2},
                    "Diamond": {"weight": 0.0, "min": 0, "max": 0},
                    "QuestionBag": {"weight": 0.2, "min": 0, "max": 1},
                    "Mole": {"weight": 0.0, "min": 0, "max": 0},
                    "MoleWithDiamond": {"weight": 0.0, "min": 0, "max": 0},
                    "TNT": {"weight": 0.0, "min": 0, "max": 0},
                    "Skull": {"weight": 0.0, "min": 0, "max": 0},
                    "Bone": {"weight": 0.0, "min": 0, "max": 0}
                },
                "medium": {
                    # 🎯 MEDIUM: Nhiều vàng + đá, có đá to, đá dày tầng trên (GIẢM 35%)
                    "MiniRock": {"weight": 0.5, "min": 2, "max": 4},
                    "NormalRock": {"weight": 0.4, "min": 1, "max": 3},
                    "BigRock": {"weight": 0.3, "min": 0, "max": 2},
                    "MiniGold": {"weight": 0.6, "min": 2, "max": 4},
                    "NormalGold": {"weight": 0.5, "min": 2, "max": 3},
                    "BigGold": {"weight": 0.8, "min": 2, "max": 4},
                    "Diamond": {"weight": 0.0, "min": 0, "max": 0},
                    "QuestionBag": {"weight": 0.3, "min": 0, "max": 2},
                    "Mole": {"weight": 0.0, "min": 0, "max": 0},
                    "MoleWithDiamond": {"weight": 0.0, "min": 0, "max": 0},
                    "TNT": {"weight": 0.0, "min": 0, "max": 0},
                    "Skull": {"weight": 0.0, "min": 0, "max": 0},
                    "Bone": {"weight": 0.0, "min": 0, "max": 0}
                },
                "hard": {
                    # 🎯 HARD: Xuất hiện kim cương, nhiều đá to (GIẢM 35%)
                    "MiniRock": {"weight": 0.3, "min": 1, "max": 3},
                    "NormalRock": {"weight": 0.4, "min": 1, "max": 3},
                    "BigRock": {"weight": 0.5, "min": 1, "max": 3},
                    "MiniGold": {"weight": 0.3, "min": 1, "max": 3},
                    "NormalGold": {"weight": 0.4, "min": 1, "max": 3},
                    "BigGold": {"weight": 0.7, "min": 2, "max": 3},
                    "Diamond": {"weight": 0.8, "min": 1, "max": 3},
                    "QuestionBag": {"weight": 0.2, "min": 0, "max": 2},
                    "Mole": {"weight": 0.0, "min": 0, "max": 0},
                    "MoleWithDiamond": {"weight": 0.0, "min": 0, "max": 0},
                    "TNT": {"weight": 0.0, "min": 0, "max": 0},
                    "Skull": {"weight": 0.0, "min": 0, "max": 0},
                    "Bone": {"weight": 0.0, "min": 0, "max": 0}
                },
                "hard_speed_run": {
                    # 🎯 HARD - CHẠY ĐUA: Nhiều vàng dễ lấy, ít chướng ngại, nhưng ít thời gian
                    "MiniRock": {"weight": 0.1, "min": 0, "max": 2},
                    "NormalRock": {"weight": 0.1, "min": 0, "max": 2},
                    "BigRock": {"weight": 0.0, "min": 0, "max": 0},
                    "MiniGold": {"weight": 0.8, "min": 8, "max": 12},
                    "NormalGold": {"weight": 0.7, "min": 6, "max": 10},
                    "BigGold": {"weight": 0.5, "min": 3, "max": 6},
                    "Diamond": {"weight": 0.3, "min": 1, "max": 3},
                    "QuestionBag": {"weight": 0.4, "min": 2, "max": 5},
                    "Mole": {"weight": 0.0, "min": 0, "max": 0},
                    "MoleWithDiamond": {"weight": 0.0, "min": 0, "max": 0},
                    "TNT": {"weight": 0.0, "min": 0, "max": 0},
                    "Skull": {"weight": 0.0, "min": 0, "max": 0},
                    "Bone": {"weight": 0.0, "min": 0, "max": 0}
                },
                "hard_treasure_hunt": {
                    # 🎯 HARD - ĐI TÌM KHO BÁU: Nhiều diamond ẩn, ít vàng, nhiều bẫy
                    "MiniRock": {"weight": 0.3, "min": 2, "max": 4},
                    "NormalRock": {"weight": 0.4, "min": 2, "max": 5},
                    "BigRock": {"weight": 0.5, "min": 3, "max": 5},
                    "MiniGold": {"weight": 0.1, "min": 1, "max": 3},
                    "NormalGold": {"weight": 0.1, "min": 0, "max": 2},
                    "BigGold": {"weight": 0.0, "min": 0, "max": 0},
                    "Diamond": {"weight": 0.9, "min": 4, "max": 8},      # RẤT NHIỀU diamond
                    "QuestionBag": {"weight": 0.3, "min": 1, "max": 3},
                    "Mole": {"weight": 0.2, "min": 1, "max": 3},
                    "MoleWithDiamond": {"weight": 0.0, "min": 0, "max": 0},
                    "TNT": {"weight": 0.4, "min": 2, "max": 4},
                    "Skull": {"weight": 0.2, "min": 1, "max": 3},
                    "Bone": {"weight": 0.2, "min": 1, "max": 3}
                },
                "hard_lottery": {
                    # 🎯 Toàn túi bí ẩn, hoặc trúng lớn hoặc thua đậm
                    "MiniRock": {"weight": 0.1, "min": 0, "max": 2},
                    "NormalRock": {"weight": 0.1, "min": 0, "max": 2},
                    "BigRock": {"weight": 0.0, "min": 0, "max": 0},
                    "MiniGold": {"weight": 0.0, "min": 0, "max": 0},
                    "NormalGold": {"weight": 0.0, "min": 0, "max": 0},
                    "BigGold": {"weight": 0.0, "min": 0, "max": 0},
                    "Diamond": {"weight": 0.0, "min": 0, "max": 0},
                    "QuestionBag": {"weight": 1.0, "min": 15, "max": 25}, # TOÀN TÚI!
                    "Mole": {"weight": 0.0, "min": 0, "max": 0},
                    "MoleWithDiamond": {"weight": 0.0, "min": 0, "max": 0},
                    "TNT": {"weight": 0.2, "min": 1, "max": 3},
                    "Skull": {"weight": 0.0, "min": 0, "max": 0},
                    "Bone": {"weight": 0.0, "min": 0, "max": 0}
                },
                "expert_diamond_moles": {
                    # 🎯 EXPERT DẠNG 1: Mole với diamond, TNT, ít kim cương thường
                    "MiniRock": {"weight": 0.3, "min": 2, "max": 4},
                    "NormalRock": {"weight": 0.4, "min": 2, "max": 5},
                    "BigRock": {"weight": 0.5, "min": 2, "max": 4},
                    "MiniGold": {"weight": 0.0, "min": 1, "max": 3},     # Rất ít vàng
                    "NormalGold": {"weight": 0.0, "min": 0, "max": 2},
                    "BigGold": {"weight": 0.0, "min": 0, "max": 0},      # Không có vàng lớn
                    "Diamond": {"weight": 0.3, "min": 1, "max": 2},      # 🎯 TĂNG: Đảm bảo có ít nhất 1 diamond
                    "QuestionBag": {"weight": 0.1, "min": 0, "max": 1},
                    "Mole": {"weight": 0.3, "min": 1, "max": 3},
                    "MoleWithDiamond": {"weight": 0.8, "min": 2, "max": 4},  # Nhiều mole với diamond
                    "TNT": {"weight": 0.6, "min": 1, "max": 3},          # Nhiều TNT
                    "Skull": {"weight": 0.0, "min": 0, "max": 0},
                    "Bone": {"weight": 0.0, "min": 0, "max": 0}
                },
                "expert_gold_rocks": {
                    # 🎯 EXPERT DẠNG 2: Không kim cương, chỉ vàng + đá + ít TNT + skull/bone
                    "MiniRock": {"weight": 0.5, "min": 3, "max": 6},
                    "NormalRock": {"weight": 0.6, "min": 3, "max": 6},
                    "BigRock": {"weight": 0.7, "min": 2, "max": 5},      # Rất nhiều đá to
                    "MiniGold": {"weight": 0.4, "min": 2, "max": 5},
                    "NormalGold": {"weight": 0.3, "min": 1, "max": 4},
                    "BigGold": {"weight": 0.2, "min": 0, "max": 2},
                    "Diamond": {"weight": 0.0, "min": 0, "max": 0},      # KHÔNG có diamond
                    "QuestionBag": {"weight": 0.1, "min": 0, "max": 1},
                    "Mole": {"weight": 0.0, "min": 0, "max": 0},
                    "MoleWithDiamond": {"weight": 0.0, "min": 0, "max": 0},
                    "TNT": {"weight": 0.3, "min": 1, "max": 2},          # Ít TNT
                    "Skull": {"weight": 0.4, "min": 1, "max": 3},        # Có skull
                    "Bone": {"weight": 0.4, "min": 1, "max": 3}          # Có bone
                },
                "expert_crystal_cave": {
                    # 🎯 EXPERT - HANG ĐỘNG PHÁT SÁNG: Toàn kim cương và đá quý
                    "MiniRock": {"weight": 0.2, "min": 1, "max": 3},
                    "NormalRock": {"weight": 0.2, "min": 1, "max": 3},
                    "BigRock": {"weight": 0.3, "min": 1, "max": 3},
                    "MiniGold": {"weight": 0.0, "min": 0, "max": 0},
                    "NormalGold": {"weight": 0.0, "min": 0, "max": 0},
                    "BigGold": {"weight": 0.0, "min": 0, "max": 0},
                    "Diamond": {"weight": 0.9, "min": 6, "max": 12},     # SIÊU NHIỀU diamond
                    "QuestionBag": {"weight": 0.3, "min": 2, "max": 4},
                    "Mole": {"weight": 0.4, "min": 2, "max": 4},
                    "MoleWithDiamond": {"weight": 0.6, "min": 3, "max": 6},
                    "TNT": {"weight": 0.5, "min": 2, "max": 4},
                    "Skull": {"weight": 0.3, "min": 1, "max": 3},
                    "Bone": {"weight": 0.3, "min": 1, "max": 3}
                },
                "expert_gauntlet": {
                    # 🎯 EXPERT - VÒNG VÂY TỬ THẦN: Toàn chướng ngại vật và quái
                    "MiniRock": {"weight": 0.6, "min": 4, "max": 8},
                    "NormalRock": {"weight": 0.7, "min": 5, "max": 9},
                    "BigRock": {"weight": 0.8, "min": 4, "max": 7},
                    "MiniGold": {"weight": 0.05, "min": 0, "max": 1},
                    "NormalGold": {"weight": 0.05, "min": 0, "max": 1},
                    "BigGold": {"weight": 0.0, "min": 0, "max": 0},
                    "Diamond": {"weight": 0.0, "min": 0, "max": 0},
                    "QuestionBag": {"weight": 0.1, "min": 0, "max": 1},
                    "Mole": {"weight": 0.6, "min": 3, "max": 6},
                    "MoleWithDiamond": {"weight": 0.4, "min": 2, "max": 4},
                    "TNT": {"weight": 0.7, "min": 3, "max": 6},
                    "Skull": {"weight": 0.5, "min": 2, "max": 5},
                    "Bone": {"weight": 0.5, "min": 2, "max": 5}
                },
                "expert_risk_reward": {
                    # 🎯 Mỗi vật giá trị cao đều có TNT bảo vệ
                    "MiniRock": {"weight": 0.2, "min": 1, "max": 3},
                    "NormalRock": {"weight": 0.2, "min": 1, "max": 3},
                    "BigRock": {"weight": 0.0, "min": 0, "max": 0},
                    "MiniGold": {"weight": 0.3, "min": 2, "max": 4},
                    "NormalGold": {"weight": 0.2, "min": 1, "max": 3},
                    "BigGold": {"weight": 0.8, "min": 3, "max": 6},      # BigGold có TNT đi kèm
                    "Diamond": {"weight": 0.9, "min": 4, "max": 8},     # Diamond có TNT đi kèm  
                    "QuestionBag": {"weight": 0.6, "min": 3, "max": 6}, # Túi có TNT đi kèm
                    "Mole": {"weight": 0.0, "min": 0, "max": 0},
                    "MoleWithDiamond": {"weight": 0.0, "min": 0, "max": 0},
                    "TNT": {"weight": 0.9, "min": 8, "max": 15},        # RẤT NHIỀU TNT
                    "Skull": {"weight": 0.0, "min": 0, "max": 0},
                    "Bone": {"weight": 0.0, "min": 0, "max": 0}
                },
                "expert_bone_graveyard": {
                    # 🎯 Toàn xương và sọ, rất ít vật có giá trị
                    "MiniRock": {"weight": 0.2, "min": 1, "max": 3},
                    "NormalRock": {"weight": 0.2, "min": 1, "max": 3},
                    "BigRock": {"weight": 0.0, "min": 0, "max": 0},
                    "MiniGold": {"weight": 0.1, "min": 0, "max": 2},
                    "NormalGold": {"weight": 0.1, "min": 0, "max": 1},
                    "BigGold": {"weight": 0.0, "min": 0, "max": 0},
                    "Diamond": {"weight": 0.0, "min": 0, "max": 0},
                    "QuestionBag": {"weight": 0.2, "min": 1, "max": 3},
                    "Mole": {"weight": 0.3, "min": 1, "max": 3},
                    "MoleWithDiamond": {"weight": 0.0, "min": 0, "max": 0},
                    "TNT": {"weight": 0.4, "min": 2, "max": 4},
                    "Skull": {"weight": 0.9, "min": 8, "max": 15},       # SIÊU NHIỀU SKULL
                    "Bone": {"weight": 0.9, "min": 8, "max": 15}         # SIÊU NHIỀU BONE
                }
            },
            "spawn_rules": {
                "x_range": (50, 1200),
                "y_range": (200, 650),
                "min_distance": 120,  # Tăng lên 120 để items thưa hơn
                "regions": {
                    "top": {"y_range": (200, 350), "density": 0.2},     # Tăng từ 0.3 lên 0.35
                    "middle": {"y_range": (350, 500), "density": 0.3},  # Tăng từ 0.4 lên 0.45
                    "bottom": {"y_range": (500, 650), "density": 0.2}   # Tăng từ 0.3 lên 0.35
                },
                # 🎯 SPAWN RULES ĐẶC BIỆT THEO DIFFICULTY
                "difficulty_spawn_rules": {
                    "medium": {
                        "rock_top_density": 0.7  # 70% đá spawn ở tầng trên
                    },
                    "hard": {
                        "diamond_prefer_bottom": True  # Diamond thích spawn tầng dưới
                    }
                }
            },
            "difficulty_profiles": {
                "train": {"total_entities": (1, 1), "value_ratio": 1.0},      # CHỈ 1 ITEM DUY NHẤT
                "easy": {"total_entities": (5, 8), "value_ratio": 0.8},       # Giảm thêm 30%: 5-8 items
                "medium": {"total_entities": (8, 12), "value_ratio": 0.7},    # Giảm thêm 30%: 8-12 items
                "hard": {"total_entities": (10, 15), "value_ratio": 0.6},     # Giảm thêm 30%: 10-15 items
                "expert": {"total_entities": (12, 18), "value_ratio": 0.5}    # Giảm thêm 30%: 12-18 items
            }
        }
    def generate_level(self, level_id: str, difficulty: str = "medium") -> Dict:
        """Tạo level ngẫu nhiên - HỖ TRỢ TẤT CẢ BIẾN THỂ"""
        
        # XỬ LÝ TẤT CẢ BIẾN THỂ (easy, medium, hard, expert)
        actual_difficulty = difficulty
        
        # NẾU LÀ DIFFICULTY CƠ BẢN → CHỌN BIẾN THỂ TƯƠNG ỨNG
        if difficulty in ["easy", "medium", "hard", "expert"]:
            actual_difficulty = self._calculate_difficulty_weights(difficulty)
        
        # Lấy profile từ difficulty gốc (không phải biến thể)
        profile_key = difficulty  # Dùng difficulty gốc để lấy profile
        if profile_key not in self.config["difficulty_profiles"]:
            profile_key = "medium"
        
        profile = self.config["difficulty_profiles"][profile_key]
        
        level_data = {
            "type": random.choice(self.config["level_types"]),
            "difficulty": difficulty,           # Lưu difficulty gốc
            "actual_difficulty": actual_difficulty,  # 🎯 THÊM: biến thể thực tế
            "entities": []
        }
        
        # Xác định số lượng entity tổng
        total_entities = random.randint(profile["total_entities"][0], profile["total_entities"][1])
        
        # 🎯 Truyền actual_difficulty (có thể là bất kỳ biến thể nào)
        entity_distribution = self._get_entity_distribution(actual_difficulty, total_entities)
        
        # Tạo từng entity
        positions = []
        for entity_type, count in entity_distribution.items():
            for _ in range(count):
                # 🎯 Truyền actual_difficulty cho spawn rules đặc biệt
                pos = self._find_valid_position(positions, 50, actual_difficulty)
                entity_data = {
                    "type": entity_type, 
                    "pos": pos,
                    "x": pos["x"], "y": pos["y"]
                }
                
                # Thêm direction cho các entity di chuyển
                if entity_type in ["Mole", "MoleWithDiamond"]:
                    entity_data["dir"] = random.choice(["Left", "Right"])
                
                level_data["entities"].append(entity_data)
                positions.append(pos)
        
        # DEBUG CHI TIẾT
        print(f"\n{'='*50}")
        print(f"🎯 GENERATED LEVEL DEBUG")
        print(f"{'='*50}")
        print(f"📁 Level ID: {level_id}")
        print(f"🎚️  Base Difficulty: {difficulty}")
        print(f"🎛️  Actual Variant: {actual_difficulty}")
        print(f"📊 Profile: {profile['total_entities'][0]}-{profile['total_entities'][1]} items")
        print(f"🔢 Total entities: {len(level_data['entities'])}")
        
        # Phân tích phân bố
        entity_count = {}
        for entity in level_data['entities']:
            entity_type = entity['type']
            entity_count[entity_type] = entity_count.get(entity_type, 0) + 1
        
        return level_data
    def _get_entity_distribution(self, difficulty: str, total_entities: int) -> Dict:
        """Phân phối entity types dựa trên difficulty - SỬ DỤNG CONFIG RIÊNG"""
        
        # LẤY CONFIG THEO DIFFICULTY
        entity_configs = self.config["entity_types_by_difficulty"]
        
        if difficulty in entity_configs:
            entity_config = entity_configs[difficulty]
        else:
            # Fallback: nếu không tìm thấy config, dùng config cho difficulty cơ bản
            base_difficulty = difficulty.split('_')[-1] if '_' in difficulty else difficulty
            if base_difficulty in ["easy", "medium", "hard"]:
                entity_config = entity_configs[base_difficulty]
            else:
                entity_config = entity_configs["medium"]
        
        # TẠO PRIORITY ORDER
        priority_order = sorted(
            [et for et in entity_config.keys() if entity_config[et]["weight"] > 0],
            key=lambda et: entity_config[et]["weight"],
            reverse=True
        )
        
        distribution = {}
        entities_left = total_entities
        
        # PHÂN PHỐI VẬT PHẨM THEO ĐỘ ƯU TIÊN
        for entity_type in priority_order:
            if entities_left <= 0:
                break
                
            config = entity_config[entity_type]
            min_count = config["min"]
            max_count = min(config["max"], entities_left)
            
            if max_count <= 0:
                continue
                
            # Tính số lượng dựa trên weight và số slot còn lại
            weight_factor = config["weight"]
            calculated_count = max(min_count, int(weight_factor * entities_left * 0.6))
            count = min(calculated_count, max_count)
            
            if count > 0:
                distribution[entity_type] = count
                entities_left -= count
        
        # XỬ LÝ SỐ LƯỢNG CÒN LẠI
        if entities_left > 0:
            # Tạo danh sách các type có thể nhận thêm
            available_types = []
            for entity_type in priority_order:
                current_count = distribution.get(entity_type, 0)
                max_allowed = entity_config[entity_type]["max"]
                if current_count < max_allowed:
                    available_types.append(entity_type)
            
            # Phân phối số entities còn lại
            while entities_left > 0 and available_types:
                for entity_type in available_types:
                    if entities_left <= 0:
                        break
                    
                    current_count = distribution.get(entity_type, 0)
                    max_allowed = entity_config[entity_type]["max"]
                    
                    if current_count < max_allowed:
                        distribution[entity_type] = current_count + 1
                        entities_left -= 1
                        
                        # Nếu đã đạt max, loại khỏi available_types
                        if distribution[entity_type] >= max_allowed:
                            available_types.remove(entity_type)
                
                # Nếu không còn type nào available mà vẫn còn entities
                if entities_left > 0 and not available_types:
                    break
        
        # ĐẢM BẢO MIN COUNT CHO CÁC ENTITY QUAN TRỌNG - TRANSFER NHIỀU LẦN
        for entity_type in priority_order:
            min_count = entity_config[entity_type]["min"]
            max_count = entity_config[entity_type]["max"]
            current_count = distribution.get(entity_type, 0)
            
            # Transfer cho đến khi đạt min hoặc không còn cách nào
            while min_count > 0 and current_count < min_count and current_count < max_count:
                # Tìm entity nào có số lượng nhiều để giảm bớt
                transferred = False
                for reduce_type in priority_order:
                    if reduce_type != entity_type and distribution.get(reduce_type, 0) > entity_config[reduce_type]["min"]:
                        distribution[reduce_type] -= 1
                        distribution[entity_type] = min(distribution.get(entity_type, 0) + 1, max_count)
                        current_count = distribution[entity_type]
                        print(f"   DEBUG: Transferred 1 from {reduce_type} to {entity_type} (now {current_count}/{min_count} min)")
                        transferred = True
                        break
                
                # Nếu không transfer được thì dừng (tránh vòng lặp vô hạn)
                if not transferred:
                    break
        
        return distribution
    def _calculate_difficulty_weights(self, difficulty: str) -> str:
        """Tính toán weights - CHỌN BIẾN THỂ NGẪU NHIÊN CÂN BẰNG"""
        
        if difficulty == "hard":
            # 4 BIẾN THỂ CHO HARD
            hard_variants = [
                "hard",              # Hard gốc
                "hard_speed_run",    # Chạy đua
                "hard_treasure_hunt", # Đi tìm kho báu
                "hard_lottery"       # Xổ số may mắn
            ]
            
            # THÊM LOGIC TRÁNH LẶP LIÊN TIẾP
            selected_variant = self._get_balanced_variant(hard_variants, "hard")
            return selected_variant
            
        elif difficulty == "expert":
            # 6 BIẾN THỂ CHO EXPERT
            expert_variants = [
                "expert_diamond_moles",  # Diamond + Moles
                "expert_gold_rocks",     # Vàng + Đá
                "expert_crystal_cave",   # Hang động phát sáng
                "expert_gauntlet",       # Vòng vây tử thần
                "expert_bone_graveyard", # Bãi chôn xương
                "expert_risk_reward"     # Cân bằng nguy hiểm
            ]
            
            # THÊM LOGIC TRÁNH LẶP LIÊN TIẾP
            selected_variant = self._get_balanced_variant(expert_variants, "expert")
            return selected_variant
            
        else:
            # Easy và Medium giữ nguyên
            return difficulty

    def _get_balanced_variant(self, variants: List[str], difficulty_type: str) -> str:
        """Chọn biến thể cân bằng, tránh lặp lại liên tiếp"""
        
        # Khởi tạo lịch sử nếu chưa có
        history_key = f'_recent_{difficulty_type}_variants'
        if not hasattr(self, history_key):
            setattr(self, history_key, [])
        
        recent_variants = getattr(self, history_key)
        
        # 🎯 NẾU 2 LẦN GẦN ĐÂY CÙNG 1 BIẾN THỂ → CHỌN BIẾN THỂ KHÁC
        if len(recent_variants) >= 2:
            last_two = recent_variants[-2:]
            if len(set(last_two)) == 1:  # Cùng 1 biến thể 2 lần liên tiếp
                # Tìm biến thể khác
                other_variants = [v for v in variants if v != last_two[0]]
                if other_variants:
                    selected = random.choice(other_variants)
                    
                    # Cập nhật lịch sử
                    recent_variants.append(selected)
                    if len(recent_variants) > 3:  # Giữ 3 lần gần nhất
                        recent_variants.pop(0)
                    setattr(self, history_key, recent_variants)
                    
                    return selected
        
        # CHỌN NGẪU NHIÊN BÌNH THƯỜNG
        selected = random.choice(variants)
        
        # Cập nhật lịch sử
        recent_variants.append(selected)
        if len(recent_variants) > 3:  # Giữ 3 lần gần nhất
            recent_variants.pop(0)
        setattr(self, history_key, recent_variants)
        
        return selected
            
    def _find_valid_position(self, existing_positions: List[Dict], max_attempts: int = 50, difficulty: str = "medium") -> Dict:
        """Tìm vị trí hợp lệ không trùng lặp - HỖ TRỢ DIFFICULTY"""
        
        # 🎯 XỬ LÝ SPAWN RULES ĐẶC BIỆT THEO DIFFICULTY
        spawn_rules = self.config["spawn_rules"]
        
        # 🎯 MEDIUM: Đá dày ở tầng trên
        if difficulty == "medium" and "difficulty_spawn_rules" in spawn_rules:
            medium_rules = spawn_rules["difficulty_spawn_rules"].get("medium", {})
            rock_top_density = medium_rules.get("rock_top_density", 0.3)
        else:
            rock_top_density = 0.3  # Mặc định
        
        for attempt in range(max_attempts):
            # 🎯 LOGIC SPAWN ĐẶC BIỆT
            if difficulty == "medium" and random.random() < rock_top_density:
                region = spawn_rules["regions"]["top"]  # Ưu tiên tầng trên cho medium
            else:
                region = random.choice(list(spawn_rules["regions"].values()))
                
            x = random.randint(spawn_rules["x_range"][0], spawn_rules["x_range"][1])
            y = random.randint(region["y_range"][0], region["y_range"][1])
            
            pos = {"x": x, "y": y}
            
            # Kiểm tra khoảng cách tối thiểu
            if all(self._calculate_distance(pos, existing) >= spawn_rules["min_distance"]
                for existing in existing_positions):
                return pos
    
        # Fallback: random position không check distance
        print(f"   ⚠️  WARNING: Failed to find valid position after {max_attempts} attempts")
        x = random.randint(spawn_rules["x_range"][0], spawn_rules["x_range"][1])
        y = random.randint(spawn_rules["y_range"][0], spawn_rules["y_range"][1])
        return {"x": x, "y": y}
            
    def _calculate_distance(self, pos1: Dict, pos2: Dict) -> float:
            """Tính khoảng cách Euclid giữa 2 points"""
            return ((pos1["x"] - pos2["x"])**2 + (pos1["y"] - pos2["y"])**2)**0.5
class ProceduralLevelManager:
    def __init__(self, generator: LevelGenerator):
        self.generator = generator
        self.generated_levels = {}
        self.difficulty_progression = [
            "easy", "easy", "medium", "medium", "medium", 
            "hard", "hard", "hard", "expert", "expert"
        ]
    
    def get_level(self, level_id: str, difficulty: str = None) -> Dict:
        """Lấy level - generate nếu chưa tồn tại"""
        
        if level_id not in self.generated_levels:
            if difficulty is None:
                # Auto difficulty progression
                level_num = self._extract_level_number(level_id)
                
                # 🎯 ÁNH XẠ: Level 0 = train difficulty
                if level_num == 0:
                    difficulty = "train"
                else:
                    # Level vô hạn - lặp lại progression
                    progression_index = (level_num - 1) % len(self.difficulty_progression)
                    difficulty = self.difficulty_progression[progression_index]
            
            # Đảm bảo difficulty hợp lệ
            valid_difficulties = ["train", "easy", "medium", "hard", "expert"]
            if difficulty not in valid_difficulties:
                difficulty = "medium"
            
            self.generated_levels[level_id] = self.generator.generate_level(level_id, difficulty)
        
        return self.generated_levels[level_id]
    
    def generate_infinite_levels(self, base_name: str, count: int, start_difficulty: str = "easy"):
        """Tạo series level vô hạn cho training"""
        levels = {}
        for i in range(count):
            level_id = f"{base_name}_{i+1}"
            # Tăng difficulty theo progression
            difficulty_idx = min(i, len(self.difficulty_progression) - 1)
            difficulty = self.difficulty_progression[difficulty_idx]
            levels[level_id] = self.generator.generate_level(level_id, difficulty)
        
        return levels
    
    def _extract_level_number(self, level_id: str) -> int:
        """Trích xuất số level từ level_id"""
        import re
        # Tìm số trong level_id (hỗ trợ cả level 0)
        match = re.search(r'[_L](\d+)', level_id)
        if match:
            return int(match.group(1))
        # Nếu level_id là số thuần túy
        try:
            return int(level_id)
        except:
            return 1
class RLTrainingEnvironment:
    def __init__(self, level_manager: ProceduralLevelManager):
        self.level_manager = level_manager
        self.current_level = None
        self.level_pool = []
        
    def setup_training_pool(self, num_levels: int = 1000):
        """Tạo pool level cho training"""
        self.level_pool = []
        for i in range(num_levels):
            level_id = f"TRAIN_{i}"
            # Random difficulty để agent học đa dạng
            difficulty = random.choice(["easy", "medium", "hard", "expert"])
            level = self.level_manager.get_level(level_id, difficulty)
            self.level_pool.append(level)
    
    def get_random_level(self) -> Dict:
        """Lấy level ngẫu nhiên từ pool"""
        return random.choice(self.level_pool) if self.level_pool else None
    
    def get_curriculum_level(self, episode: int) -> Dict:
        """Lấy level theo curriculum learning"""
        # Tăng difficulty theo số episode
        if episode < 1000:
            difficulty = "easy"
        elif episode < 5000:
            difficulty = "medium" 
        elif episode < 10000:
            difficulty = "hard"
        else:
            difficulty = "expert"
            
        level_id = f"CURR_{episode}"
        return self.level_manager.get_level(level_id, difficulty)