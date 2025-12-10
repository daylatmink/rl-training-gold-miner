"""
Data-Driven Level Generator
Based on statistical analysis of levels.json (30 levels, L1-L10)
"""

import random
from typing import List, Dict, Tuple
from define import *


class DataDrivenLevelGenerator:
    """
    Generator based on actual level statistics:
    - L1-L3: avg 15-20 entities, 40-45% gold, 30-35% rocks, rest special/diamonds
    - L4-L6: avg 17-21 entities, introduce Mole, more diamonds
    - L7-L10: avg 18-26 entities, lots of TNT/MoleWithDiamond, obstacles appear
    """
    
    # Statistical entity distribution from levels.json analysis (reduced counts)
    LEVEL_TEMPLATES = {
        'L1': {  # L1 group statistics
            'total_items': (10, 12),
            'entities': {
                'MiniGold': 3.0,
                'NormalGold': 2.0,
                'BigGold': 1.0,
                'MiniRock': 1.5,
                'NormalRock': 1.5,
                'QuestionBag': 1.0,
                'Diamond': 0.3,
            }
        },
        'L2': {
            'total_items': (12, 14),
            'entities': {
                'MiniGold': 4.0,
                'NormalGold': 2.0,
                'BigGold': 1.0,
                'MiniRock': 2.0,
                'NormalRock': 2.5,
                'QuestionBag': 0.5,
                'Diamond': 0.3,
            }
        },
        'L3': {
            'total_items': (10, 12),
            'entities': {
                'MiniGold': 2.5,
                'NormalGold': 2.0,
                'BigGold': 0.5,
                'MiniRock': 2.0,
                'NormalRock': 2.5,
                'QuestionBag': 0.5,
                'Diamond': 0.5,
            }
        },
        'L4': {  # Mole introduced
            'total_items': (10, 12),
            'entities': {
                'MiniGold': 2.5,
                'NormalGold': 1.0,
                'NormalGoldPlus': 1.5,
                'MiniRock': 0.5,
                'NormalRock': 2.0,
                'QuestionBag': 1.5,
                'Mole': 2.0,
            }
        },
        'L5': {
            'total_items': (12, 14),
            'entities': {
                'MiniGold': 2.5,
                'NormalGold': 2.0,
                'NormalGoldPlus': 1.0,
                'BigGold': 1.0,
                'MiniRock': 0.5,
                'NormalRock': 2.0,
                'Diamond': 2.0,
                'QuestionBag': 0.5,
                'Mole': 1.0,
            }
        },
        'L6': {  # MoleWithDiamond introduced
            'total_items': (12, 14),
            'entities': {
                'MiniGold': 4.0,
                'NormalGold': 2.0,
                'BigGold': 0.5,
                'MiniRock': 2.0,
                'NormalRock': 2.0,
                'QuestionBag': 0.5,
                'MoleWithDiamond': 2.0,
            }
        },
        'L7': {  # TNT, Skull, Bone appear
            'total_items': (11, 13),
            'entities': {
                'MiniGold': 1.0,
                'NormalGoldPlus': 1.0,
                'BigGold': 2.0,
                'QuestionBag': 0.5,
                'Mole': 3.0,
                'TNT': 1.5,
                'Skull': 1.0,
                'Bone': 1.0,
            }
        },
        'L8': {
            'total_items': (10, 16),
            'variant': 'random',  # Two variants
            'variant_a': {  # Diamond + TNT heavy (clustered in middle)
                'entities': {
                    'Diamond': 4.0,
                    'TNT': 3.0,
                    'QuestionBag': 1.0,
                    'MoleWithDiamond': 1.5,
                },
                'spawn_preference': 'middle_clustered'  # Spawn gần nhau ở middle
            },
            'variant_b': {  # BigGold + lots of Rocks
                'entities': {
                    'BigGold': 2.5,
                    'MiniRock': 4.0,
                    'NormalRock': 5.0,
                    'Diamond': 2.5,
                    'TNT': 1.5,
                },
                'spawn_preference': 'rocks_between_gold'  # Rocks giữa các BigGold
            }
        },
        'L9': {
            'total_items': (8, 13),
            'variant': 'random',  # Three variants
            'variant_a': {  # 2 BigGold split screen + MoleWithDiamond + TNT
                'entities': {
                    'BigGold': 2.0,
                    'MoleWithDiamond': 2.5,
                    'TNT': 3.5,
                    'Bone': 0.5,
                    'QuestionBag': 0.5,
                    'MiniGold': 0.5,
                    'Diamond': 1.0,
                },
                'spawn_preference': 'split_screen_gold'  # 2 BigGold on left/right halves
            },
            'variant_b': {  # TNT + Moles scattered
                'entities': {
                    'TNT': 1.5,
                    'Mole': 2.5,
                    'MoleWithDiamond': 2.5,
                    'MiniGold': 1.0,
                },
                'spawn_preference': 'random_tnt_moles'  # TNT random, moles scattered
            },
            'variant_c': {  # Original L9 pattern (fallback)
                'entities': {
                    'MiniGold': 0.5,
                    'BigGold': 1.0,
                    'QuestionBag': 1.0,
                    'MoleWithDiamond': 2.5,
                    'TNT': 4.0,
                    'Skull': 0.5,
                    'Bone': 0.5,
                },
                'spawn_preference': None
            }
        },
        'L10': {  # Hardest levels
            'total_items': (12, 16),
            'variant': 'random',  # Two variants
            'variant_a': {  # Lots of gold at bottom, lots of rocks at top
                'entities': {
                    'MiniGold': 3.5,
                    'NormalGold': 3.0,
                    'NormalGoldPlus': 3.0,
                    'BigGold': 3.0,
                    'MiniRock': 2.0,
                    'NormalRock': 3.0,
                },
                'spawn_preference': 'gold_bottom_rocks_top'
            },
            'variant_b': {  # Diamond + TNT + MoleWithDiamond heavy
                'entities': {
                    'BigGold': 1.5,
                    'MoleWithDiamond': 2.5,
                    'Diamond': 3.5,
                    'TNT': 3.5,
                    'QuestionBag': 1.0,
                    'Skull': 0.5,
                    'Bone': 0.5,
                },
                'spawn_preference': 'diamond_tnt_interspersed'
            }
        }
    }
    
    # Spawn regions (y-axis) from analysis
    SPAWN_REGIONS = {
        'top': {
            'y_range': (200, 350),
            'preferred_entities': ['MiniGold', 'MiniRock', 'QuestionBag', 'Diamond'],
        },
        'middle': {
            'y_range': (350, 650),
            'preferred_entities': ['NormalGold', 'NormalRock', 'BigGold', 'Mole', 'MoleWithDiamond'],
        },
        'bottom': {
            'y_range': (650, 750),
            'preferred_entities': ['BigGold', 'BigRock', 'Diamond', 'TNT', 'Skull', 'Bone'],
        }
    }
    
    def __init__(self, screen_width: int = 1280, screen_height: int = 720):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.min_distance = 120  # Minimum spacing between entities (increased from 80 to avoid overlap)
    
    def generate_level(self, difficulty_level: int = 1) -> List[Dict]:
        """
        Generate level based on difficulty (1-10)
        Uses statistical templates from levels.json
        """
        # Map difficulty to template
        template_key = f'L{difficulty_level}'
        if template_key not in self.LEVEL_TEMPLATES:
            template_key = 'L5'  # Default to medium difficulty
        
        template = self.LEVEL_TEMPLATES[template_key]
        
        # Handle L8, L9, L10 variants
        if 'variant' in template:
            if template_key == 'L8':
                variant = random.choice(['variant_a', 'variant_b'])
            elif template_key == 'L9':
                variant = random.choice(['variant_a', 'variant_b', 'variant_c'])
            elif template_key == 'L10':
                variant = random.choice(['variant_a', 'variant_b'])
            else:
                variant = 'variant_a'  # Default
            
            template = template[variant]
            spawn_preference = template.get('spawn_preference', None)
        else:
            spawn_preference = None
        
        # Generate entities
        entities = []
        positions = []  # Track occupied positions
        
        for entity_type, avg_count in template['entities'].items():
            # Determine actual count (add variance)
            if avg_count < 1:  # Low probability entities
                count = 1 if random.random() < avg_count else 0
            else:
                # Add ±20% variance
                variance = int(avg_count * 0.2) + 1
                count = max(0, int(avg_count) + random.randint(-variance, variance))
            
            # Generate entities
            for _ in range(count):
                pos = self._generate_position(entity_type, positions, spawn_preference)
                if pos:
                    entity = {
                        'type': entity_type,
                        'pos': pos
                    }
                    
                    # Add direction for mobile entities
                    if 'Mole' in entity_type:
                        entity['dir'] = random.choice(['Left', 'Right'])
                    
                    entities.append(entity)
                    positions.append(pos)
        
        return entities
    
    def _generate_position(self, entity_type: str, occupied_positions: List[Dict], spawn_preference: str = None) -> Dict:
        """Generate non-overlapping position for entity"""
        # Handle special spawn preferences
        if spawn_preference == 'middle_clustered':
            # L8 variant A: Diamond + TNT scattered across wide area (not clustered)
            if entity_type in ['Diamond', 'TNT']:
                # Scatter across wide y-range (not just middle)
                y_min, y_max = 250, 650  # Wide vertical range
            else:
                # Other entities use default regions
                region = self._get_preferred_region(entity_type)
                y_min, y_max = region['y_range']
        
        elif spawn_preference == 'rocks_between_gold':
            # L8 variant B: Rocks spawn between BigGold positions
            if entity_type in ['MiniRock', 'NormalRock']:
                # Find BigGold positions
                gold_positions = [p for p in occupied_positions if any(
                    e.get('type') == 'BigGold' for e in [] if e.get('pos') == p
                )]
                # Note: This is simplified - just use wider y-range for rocks
                y_min, y_max = 300, 650
            else:
                region = self._get_preferred_region(entity_type)
                y_min, y_max = region['y_range']
        
        elif spawn_preference == 'split_screen_gold':
            # L9 variant A: 2 BigGold on left/right halves, MoleWithDiamond near them, TNT scattered
            if entity_type == 'BigGold':
                # First BigGold on left half, second on right half
                big_gold_count = sum(1 for p in occupied_positions if p.get('entity_type') == 'BigGold')
                if big_gold_count == 0:
                    # First BigGold: left half
                    x = random.randint(200, self.screen_width // 2 - 100)
                else:
                    # Second BigGold: right half
                    x = random.randint(self.screen_width // 2 + 100, self.screen_width - 200)
                y = random.randint(400, 600)  # Middle depth
                
                # Check distance and return early if valid
                if all(self._distance(x, y, p['x'], p['y']) >= self.min_distance 
                       for p in occupied_positions):
                    result = {'x': x, 'y': y}
                    result['entity_type'] = entity_type  # Track entity type
                    return result
                # Fall through to standard generation if failed
                region = self._get_preferred_region(entity_type)
                y_min, y_max = region['y_range']
            
            elif entity_type == 'MoleWithDiamond':
                # Spawn near BigGold positions (2 per half)
                y_min, y_max = 350, 650
            
            elif entity_type == 'TNT':
                # TNT scattered, prefer top/bottom
                if random.random() < 0.6:  # 60% chance top/bottom
                    y_min, y_max = random.choice([(250, 400), (600, 700)])
                else:
                    y_min, y_max = 250, 700  # Anywhere
            
            else:
                region = self._get_preferred_region(entity_type)
                y_min, y_max = region['y_range']
        
        elif spawn_preference == 'random_tnt_moles':
            # L9 variant B: 1 TNT anywhere, ~8 Moles scattered (ratio 2:1)
            if entity_type == 'TNT':
                # TNT at random position (not centered)
                y_min, y_max = 300, 650
            
            elif entity_type in ['Mole', 'MoleWithDiamond']:
                # Moles scattered across wide area
                y_min, y_max = 300, 650
            
            else:
                region = self._get_preferred_region(entity_type)
                y_min, y_max = region['y_range']
        
        elif spawn_preference == 'gold_bottom_rocks_top':
            # L10 variant A: Lots of gold at bottom, lots of rocks at top
            if entity_type in ['MiniGold', 'NormalGold', 'NormalGoldPlus', 'BigGold']:
                # Gold at bottom (deep)
                y_min, y_max = 550, 700
            
            elif entity_type in ['MiniRock', 'NormalRock']:
                # Rocks at top (shallow)
                y_min, y_max = 250, 450
            
            else:
                region = self._get_preferred_region(entity_type)
                y_min, y_max = region['y_range']
        
        elif spawn_preference == 'diamond_tnt_interspersed':
            # L10 variant B: Diamond + TNT interspersed, BigGold at bottom, MoleWithDiamond scattered
            if entity_type == 'BigGold':
                # BigGold at bottom
                y_min, y_max = 550, 700
            
            elif entity_type in ['Diamond', 'TNT']:
                # Diamond and TNT interspersed across middle region
                y_min, y_max = 350, 600
            
            elif entity_type == 'MoleWithDiamond':
                # MoleWithDiamond scattered
                y_min, y_max = 300, 650
            
            else:
                region = self._get_preferred_region(entity_type)
                y_min, y_max = region['y_range']
        
        else:
            # Default: use preferred region
            region = self._get_preferred_region(entity_type)
            y_min, y_max = region['y_range']
        
        # Standard position generation
        max_attempts = 50
        for _ in range(max_attempts):
            x = random.randint(100, self.screen_width - 100)
            y = random.randint(y_min, y_max)
            
            # Check if position is far enough from others
            if all(self._distance(x, y, p['x'], p['y']) >= self.min_distance 
                   for p in occupied_positions):
                return {'x': x, 'y': y}
        
        return None  # Failed to find position
    
    def _get_preferred_region(self, entity_type: str):
        """Get preferred spawn region for entity type"""
        for region in self.SPAWN_REGIONS.values():
            if entity_type in region['preferred_entities']:
                return region
        
        # Default to middle region
        return self.SPAWN_REGIONS['middle']
    
    def _distance(self, x1, y1, x2, y2):
        """Calculate Euclidean distance"""
        return ((x1 - x2)**2 + (y1 - y2)**2)**0.5


class ProceduralLevelManager:
    """
    Manages procedural level generation with caching
    """
    def __init__(self):
        self.generator = DataDrivenLevelGenerator()
        self.level_cache = {}
        self.current_difficulty = 1
    
    def get_level(self, level_id: str = None, difficulty: int = None) -> Dict:
        """
        Get level by ID or generate based on difficulty
        
        Args:
            level_id: Optional level ID for caching
            difficulty: Difficulty level (1-10) corresponding to L1-L10 templates
        """
        if level_id and level_id in self.level_cache:
            return self.level_cache[level_id]
        
        # Use provided difficulty or current difficulty
        if difficulty is None:
            difficulty = self.current_difficulty
        
        # Clamp difficulty to valid range (1-10)
        difficulty = min(max(1, difficulty), 10)
        
        # Generate new level
        entities = self.generator.generate_level(difficulty)
        
        level_data = {
            'type': random.choice(['LevelA', 'LevelB', 'LevelC', 'LevelD', 'LevelE']),
            'entities': entities
        }
        
        # Cache if level_id provided
        if level_id:
            self.level_cache[level_id] = level_data
        
        return level_data
    
    def generate_infinite_levels(self):
        """
        Generator for infinite level progression
        Difficulty increases every 3 levels
        """
        level_count = 0
        while True:
            # Increase difficulty every 3 levels (L1-L3, L4-L6, L7-L10)
            self.current_difficulty = min(10, (level_count // 3) + 1)
            
            level_id = f"GENERATED_{level_count}"
            yield self.get_level(level_id)
            level_count += 1


# Backward compatibility alias
LevelGenerator = DataDrivenLevelGenerator
