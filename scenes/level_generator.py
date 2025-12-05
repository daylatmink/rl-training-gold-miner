"""
Level Generator - Sinh level ngẫu nhiên dựa trên thiết kế levels.json
Tạo các level cân bằng với tỉ lệ vàng/đá hợp lý, tránh các trường hợp cực đoan
"""

import random
from typing import List, Dict, Tuple
import math


class LevelGenerator:
    """
    Generator tạo level ngẫu nhiên với thiết kế cân bằng
    Dựa trên phân tích levels.json:
    - Tỉ lệ vàng/đá hợp lý: 50-70% vàng, 25-40% đá, 5-15% special
    - Phân bố không gian đều (tránh chồng chéo)
    - Độ khó tăng dần tự nhiên
    """
    
    # Cấu hình cơ bản cho từng loại entity (dựa trên levels.json)
    ENTITY_CONFIG = {
        # === ROCKS - CHƯỚNG NGẠI VẬT ===
        "MiniRock": {
            "weight": 8,           # Tăng: đá nhỏ xuất hiện nhiều hơn
            "size": 30,            # Kích thước ước tính
            "value": 0,
            "categories": ["rock", "mini"]
        },
        "NormalRock": {
            "weight": 12,          # Tăng: đá vừa xuất hiện rất nhiều
            "size": 45,
            "value": 0,
            "categories": ["rock", "normal"]
        },
        "BigRock": {
            "weight": 5,           # Tăng: đá lớn xuất hiện nhiều hơn
            "size": 60,
            "value": 0,
            "categories": ["rock", "big"]
        },
        
        # === GOLD - VẬT PHẨM GIÁ TRỊ ===
        "MiniGold": {
            "weight": 10,          # Giảm xuống để cân bằng với đá
            "size": 25,
            "value": 50,
            "categories": ["gold", "mini"]
        },
        "NormalGold": {
            "weight": 8,           # Giảm xuống để cân bằng với đá
            "size": 35,
            "value": 250,
            "categories": ["gold", "normal"]
        },
        "BigGold": {
            "weight": 5,           # Giảm xuống để cân bằng với đá
            "size": 50,
            "value": 500,
            "categories": ["gold", "big"]
        },
        
        # === SPECIAL ITEMS ===
        "Diamond": {
            "weight": 2,           # Kim cương hiếm
            "size": 30,
            "value": 600,
            "categories": ["special", "valuable"]
        },
        "QuestionBag": {
            "weight": 2,           # Túi bí ẩn ít, không quá nhiều
            "size": 30,
            "value": 0,            # Random value
            "categories": ["special", "random"]
        },
        
        # === OBSTACLES - NGUY HIỂM (rất hiếm) ===
        "TNT": {
            "weight": 0.5,         # TNT cực hiếm
            "size": 35,
            "value": -200,
            "categories": ["obstacle", "dangerous"]
        },
        "Skull": {
            "weight": 0.3,         # Sọ cực hiếm
            "size": 30,
            "value": 0,
            "categories": ["obstacle", "worthless"]
        },
        "Bone": {
            "weight": 0.3,         # Xương cực hiếm
            "size": 25,
            "value": 0,
            "categories": ["obstacle", "worthless"]
        },
    }
    
    # Cấu hình theo độ khó (dựa trên phân tích levels.json)
    DIFFICULTY_CONFIG = {
        "easy": {
            "total_items": (8, 12),           # 8-12 items
            "gold_ratio": (0.40, 0.55),       # Giảm: 40-55% là vàng (trước 55-70%)
            "rock_ratio": (0.35, 0.50),       # Tăng: 35-50% là đá (trước 25-35%)
            "special_ratio": (0.05, 0.12),    # 5-12% special (diamond, question)
            "obstacle_ratio": (0.0, 0.0),     # 0% obstacles
            "diamond_chance": 0.3,            # 30% có diamond
            "question_max": 1,                # Tối đa 1 question bag
        },
        "medium": {
            "total_items": (10, 15),
            "gold_ratio": (0.35, 0.50),       # Giảm: 35-50% là vàng (trước 50-65%)
            "rock_ratio": (0.40, 0.55),       # Tăng: 40-55% là đá (trước 30-42%)
            "special_ratio": (0.08, 0.15),
            "obstacle_ratio": (0.0, 0.03),    # Tối đa 3% obstacles
            "diamond_chance": 0.5,            # 50% có diamond
            "question_max": 2,
        },
        "hard": {
            "total_items": (12, 18),
            "gold_ratio": (0.30, 0.45),       # Giảm: 30-45% là vàng (trước 45-60%)
            "rock_ratio": (0.40, 0.55),       # Tăng: 40-55% là đá (trước 30-45%)
            "special_ratio": (0.10, 0.18),
            "obstacle_ratio": (0.02, 0.08),   # 2-8% obstacles
            "diamond_chance": 0.7,            # 70% có diamond
            "question_max": 2,
        }
    }
    
    # Vùng spawn theo tầng (y coordinate) - dựa trên levels.json
    SPAWN_REGIONS = {
        "top": {
            "y_range": (200, 350),
            "rock_preference": 0.35,      # 35% đá spawn ở đây
            "gold_preference": 0.20,      # 20% vàng spawn ở đây
        },
        "middle": {
            "y_range": (350, 500),
            "rock_preference": 0.40,
            "gold_preference": 0.45,      # Vàng ưu tiên middle
        },
        "bottom": {
            "y_range": (500, 650),
            "rock_preference": 0.25,
            "gold_preference": 0.35,      # Diamond ưu tiên bottom
        }
    }
    
    # Level types từ levels.json
    LEVEL_TYPES = ["LevelA", "LevelB", "LevelC", "LevelD", "LevelE"]
    
    def __init__(self):
        """Khởi tạo generator"""
        self.min_distance = 85  # Khoảng cách tối thiểu giữa các items
        self.x_range = (60, 1180)  # Vùng spawn x (có margin)
        
    def generate_level(self, level_id: str = "GENERATED", difficulty: str = "medium") -> Dict:
        """
        Tạo level ngẫu nhiên với độ khó chỉ định
        
        Args:
            level_id: ID của level
            difficulty: "easy", "medium", hoặc "hard"
            
        Returns:
            Dict chứa level data với format giống levels.json
        """
        if difficulty not in self.DIFFICULTY_CONFIG:
            difficulty = "medium"
            
        config = self.DIFFICULTY_CONFIG[difficulty]
        
        # Xác định số lượng items
        total_items = random.randint(*config["total_items"])
        
        # Tính số lượng từng loại dựa trên ratios
        num_gold = self._get_count_from_ratio(total_items, config["gold_ratio"])
        num_rocks = self._get_count_from_ratio(total_items, config["rock_ratio"])
        num_special = self._get_count_from_ratio(total_items, config["special_ratio"])
        num_obstacles = self._get_count_from_ratio(total_items, config["obstacle_ratio"])
        
        # Điều chỉnh để tổng = total_items
        current_total = num_gold + num_rocks + num_special + num_obstacles
        remaining = total_items - current_total
        
        if remaining > 0:
            # Phân bố remaining vào vàng/đá theo tỷ lệ
            add_to_gold = int(remaining * 0.6)
            num_gold += add_to_gold
            num_rocks += (remaining - add_to_gold)
        elif remaining < 0:
            # Giảm từ obstacles trước, sau đó special
            if num_obstacles > 0:
                reduction = min(abs(remaining), num_obstacles)
                num_obstacles -= reduction
                remaining += reduction
            
            if remaining < 0 and num_special > 0:
                reduction = min(abs(remaining), num_special)
                num_special -= reduction
        
        # Tạo danh sách entities
        entities = []
        
        # 1. Tạo rocks - phân bố đều mini/normal/big
        entities.extend(self._create_category_items("rock", num_rocks))
        
        # 2. Tạo gold - phân bố đều mini/normal/big
        entities.extend(self._create_category_items("gold", num_gold))
        
        # 3. Tạo special items (Diamond, QuestionBag) - không quá nhiều
        if num_special > 0:
            # Giới hạn question bags
            num_questions = min(random.randint(0, config["question_max"]), num_special)
            num_diamonds = num_special - num_questions
            
            # Chỉ spawn diamond nếu random pass
            if random.random() < config["diamond_chance"] and num_diamonds > 0:
                for _ in range(num_diamonds):
                    entities.append({"type": "Diamond"})
            
            for _ in range(num_questions):
                entities.append({"type": "QuestionBag"})
        
        # 4. Tạo obstacles (TNT, Skull, Bone) - rất hiếm
        if num_obstacles > 0:
            obstacle_types = ["TNT", "Skull", "Bone"]
            for _ in range(num_obstacles):
                entity_type = random.choice(obstacle_types)
                entities.append({"type": entity_type})
        
        # Shuffle để random thứ tự
        random.shuffle(entities)
        
        # Gán vị trí cho các entities
        occupied_positions = []
        final_entities = []
        
        for entity in entities:
            entity_type = entity["type"]
            entity_config = self.ENTITY_CONFIG[entity_type]
            
            # Xác định region ưu tiên
            preferred_region = self._get_preferred_region(entity_type)
            
            # Tìm vị trí hợp lệ
            pos = self._find_valid_position(
                occupied_positions,
                entity_config["size"],
                preferred_region
            )
            
            if pos:
                entity_data = {
                    "type": entity_type,
                    "pos": {"x": pos[0], "y": pos[1]}
                }
                
                final_entities.append(entity_data)
                occupied_positions.append((pos[0], pos[1], entity_config["size"]))
        
        # Tạo level data
        level_data = {
            "type": random.choice(self.LEVEL_TYPES),
            "difficulty": difficulty,
            "entities": final_entities
        }
        
        return level_data
    
    def _get_count_from_ratio(self, total: int, ratio_range: Tuple[float, float]) -> int:
        """Tính số lượng từ ratio range"""
        ratio = random.uniform(*ratio_range)
        count = int(total * ratio)
        return max(0, count)
    
    def _create_category_items(self, category: str, count: int) -> List[Dict]:
        """
        Tạo items từ một category (rock, gold)
        Phân bố dựa trên weight
        """
        if count <= 0:
            return []
        
        # Lọc các entity types thuộc category
        available_types = [
            entity_type for entity_type, config in self.ENTITY_CONFIG.items()
            if category in config["categories"]
        ]
        
        if not available_types:
            return []
        
        # Tạo phân bố dựa trên weight
        weights = [self.ENTITY_CONFIG[t]["weight"] for t in available_types]
        
        # Phân bố items
        items = []
        for _ in range(count):
            entity_type = random.choices(available_types, weights=weights, k=1)[0]
            items.append({"type": entity_type})
        
        return items
    
    def _get_preferred_region(self, entity_type: str) -> str:
        """Xác định region ưu tiên cho entity type"""
        entity_config = self.ENTITY_CONFIG[entity_type]
        
        # Rocks ưu tiên top/middle
        if "rock" in entity_config["categories"]:
            preferences = [
                ("top", self.SPAWN_REGIONS["top"]["rock_preference"]),
                ("middle", self.SPAWN_REGIONS["middle"]["rock_preference"]),
                ("bottom", self.SPAWN_REGIONS["bottom"]["rock_preference"])
            ]
        # Gold và valuable items ưu tiên middle/bottom
        elif "gold" in entity_config["categories"] or "valuable" in entity_config["categories"]:
            preferences = [
                ("top", self.SPAWN_REGIONS["top"]["gold_preference"]),
                ("middle", self.SPAWN_REGIONS["middle"]["gold_preference"]),
                ("bottom", self.SPAWN_REGIONS["bottom"]["gold_preference"])
            ]
        else:
            # Default: random đều
            return random.choice(["top", "middle", "bottom"])
        
        # Chọn region dựa trên preferences
        regions = [p[0] for p in preferences]
        weights = [p[1] for p in preferences]
        
        return random.choices(regions, weights=weights, k=1)[0]
    
    def _find_valid_position(
        self,
        occupied_positions: List[Tuple[int, int, int]],
        entity_size: int,
        preferred_region: str = None
    ) -> Tuple[int, int]:
        """
        Tìm vị trí hợp lệ cho entity
        
        Args:
            occupied_positions: List các vị trí đã chiếm [(x, y, size), ...]
            entity_size: Kích thước entity
            preferred_region: Region ưu tiên ("top", "middle", "bottom")
            
        Returns:
            (x, y) hoặc None nếu không tìm được
        """
        max_attempts = 100
        
        # Xác định y_range
        if preferred_region and preferred_region in self.SPAWN_REGIONS:
            y_min, y_max = self.SPAWN_REGIONS[preferred_region]["y_range"]
        else:
            y_min, y_max = 200, 650
        
        for _ in range(max_attempts):
            x = random.randint(*self.x_range)
            y = random.randint(y_min, y_max)
            
            # Kiểm tra khoảng cách với các vị trí đã chiếm
            valid = True
            for ox, oy, osize in occupied_positions:
                distance = math.sqrt((x - ox)**2 + (y - oy)**2)
                min_required = (entity_size + osize) / 2 + self.min_distance
                
                if distance < min_required:
                    valid = False
                    break
            
            if valid:
                return (x, y)
        
        # Nếu không tìm được trong preferred region, thử random
        if preferred_region:
            return self._find_valid_position(occupied_positions, entity_size, None)
        
        return None
    
    def generate_multiple_levels(self, num_levels: int = 10, difficulty: str = "medium") -> Dict[str, Dict]:
        """
        Tạo nhiều levels cùng lúc
        
        Args:
            num_levels: Số lượng levels cần tạo
            difficulty: Độ khó
            
        Returns:
            Dict với key là level_id, value là level data
        """
        levels = {}
        for i in range(num_levels):
            level_id = f"GEN_{difficulty.upper()}_{i+1}"
            levels[level_id] = self.generate_level(level_id, difficulty)
        
        return levels
    
    def save_levels_to_json(self, levels: Dict, output_path: str = "generated_levels.json"):
        """Lưu levels ra file JSON"""
        import json
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(levels, f, indent=4, ensure_ascii=False)
        
        print(f"✓ Saved {len(levels)} levels to {output_path}")


class ProceduralLevelManager:
    """Manager để quản lý việc tạo và cache levels"""
    
    def __init__(self, generator: LevelGenerator = None):
        self.generator = generator if generator else LevelGenerator()
        self.generated_levels = {}  # Cache các levels đã tạo
        
        # Progression độ khó cho infinite levels
        self.difficulty_progression = [
            "easy", "easy", "medium", "medium", "medium", 
            "hard", "hard", "hard", "hard", "hard"
        ]
    
    def get_level(self, level_id: str, difficulty: str = None) -> Dict:
        """
        Lấy level - generate nếu chưa tồn tại
        
        Args:
            level_id: ID của level (có thể là số hoặc string)
            difficulty: Độ khó ("easy", "medium", "hard")
            
        Returns:
            Level data dict
        """
        # Check cache
        if level_id in self.generated_levels:
            return self.generated_levels[level_id]
        
        # Nếu không có difficulty, tự động tính dựa trên level_id
        if difficulty is None:
            level_num = self._extract_level_number(level_id)
            
            # Ánh xạ level number sang difficulty
            if level_num <= 0:
                difficulty = "easy"
            else:
                # Lặp lại progression
                progression_index = (level_num - 1) % len(self.difficulty_progression)
                difficulty = self.difficulty_progression[progression_index]
        
        # Generate level mới
        level_data = self.generator.generate_level(level_id, difficulty)
        
        # Cache lại
        self.generated_levels[level_id] = level_data
        
        return level_data
    
    def generate_infinite_levels(self, base_name: str, count: int) -> Dict[str, Dict]:
        """
        Tạo series levels cho training
        
        Args:
            base_name: Tên base (vd: "TRAIN")
            count: Số lượng levels
            
        Returns:
            Dict của levels {level_id: level_data}
        """
        levels = {}
        for i in range(count):
            level_id = f"{base_name}_{i+1}"
            
            # Ánh xạ difficulty theo progression
            progression_index = i % len(self.difficulty_progression)
            difficulty = self.difficulty_progression[progression_index]
            
            levels[level_id] = self.get_level(level_id, difficulty)
        
        return levels
    
    def _extract_level_number(self, level_id: str) -> int:
        """Trích xuất số level từ level_id"""
        import re
        
        # Tìm số trong level_id
        match = re.search(r'[_L](\d+)', level_id)
        if match:
            return int(match.group(1))
        
        # Nếu level_id là số thuần túy
        try:
            return int(level_id)
        except:
            return 1
    
    def clear_cache(self):
        """Xóa cache levels"""
        self.generated_levels.clear()
    
    def get_cache_size(self) -> int:
        """Lấy số lượng levels trong cache"""
        return len(self.generated_levels)
