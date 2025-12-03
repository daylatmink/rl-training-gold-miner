"""
Module để lấy và export game state
"""
import json
from datetime import datetime
from define import get_score, get_goal, get_level, get_pause, get_time, get_dynamite_count


def get_game_state(game_scene):
    """
    Lấy toàn bộ state của game tại thời điểm hiện tại
    
    Args:
        game_scene: Instance của GameScene
    
    Returns:
        dict: Dictionary chứa toàn bộ state của game
    """
    
    # 1. State Game Chính (Global State)
    global_state = {
        'score': get_score(),
        'goal': get_goal(),
        'level': get_level(),
        'time_left': game_scene.timer if hasattr(game_scene, 'timer') else 0,
        'pause': get_pause(),
        'start_time': get_time(),
        'dynamite_count': get_dynamite_count()  # Thêm số dynamite vào global state
    }
    
    # 2. State của GameScene
    scene_state = {
        'level_number': game_scene.level,
        'current_level_id': game_scene.current_level_id if hasattr(game_scene, 'current_level_id') else None,
        'use_generated': game_scene.use_generated if hasattr(game_scene, 'use_generated') else False,
        'play_explosive': game_scene.play_Explosive,
        'pause': game_scene.pause if hasattr(game_scene, 'pause') else False,
        'pause_time': game_scene.pause_time if hasattr(game_scene, 'pause_time') else 0,
        'items_count': len(game_scene.items)
    }
    
    # 3. State của Miner
    miner_state = {
        'position': {
            'x': game_scene.miner.pos_x,
            'y': game_scene.miner.pos_y
        },
        'speed': game_scene.miner.speed,
        'current_frame': game_scene.miner.current_frame,
        'state': game_scene.miner.state,
        'state_name': _get_miner_state_name(game_scene.miner.state),
        'is_play_done': game_scene.miner.is_play_done
    }
    
    # 4. State của Rope
    rope_state = {
        'start_position': {
            'x': game_scene.rope.x1,
            'y': game_scene.rope.y1
        },
        'end_position': {
            'x': game_scene.rope.x2,
            'y': game_scene.rope.y2
        },
        'length': game_scene.rope.length,
        'speed': game_scene.rope.speed,
        'buff_speed': game_scene.rope.buff_speed,
        'weight': game_scene.rope.weight,
        'direction': game_scene.rope.direction,
        'state': game_scene.rope.state,
        'has_item': game_scene.rope.item is not None,
        'item_type': type(game_scene.rope.item).__name__ if game_scene.rope.item else None,
        'tnt_count': game_scene.rope.have_TNT,
        'is_use_tnt': game_scene.rope.is_use_TNT,
        'timer': game_scene.rope.timer
    }
    
    # 5. State của các Items/Entities
    items_state = []
    for idx, item in enumerate(game_scene.items):
        # Xác định loại item chi tiết
        item_type = type(item).__name__
        item_subtype = None
        
        # Phân biệt các loại "Other" dựa vào điểm
        if item_type == "Other":
            if item.point == 600 or item.point == 900:  # Diamond (600 hoặc 900 khi có gem polish)
                item_subtype = "Diamond"
            elif item.point == 20:
                item_subtype = "Skull"
            elif item.point == 7:
                item_subtype = "Bone"
            else:
                item_subtype = "Unknown"
        
        # Phân biệt Mole thường và MoleWithDiamond
        if item_type == "Mole":
            if item.point == 2:
                item_subtype = "Mole"
            elif item.point >= 600:  # MoleWithDiamond (602 hoặc 902 khi có gem polish)
                item_subtype = "MoleWithDiamond"
            else:
                item_subtype = "Mole"
        
        item_data = {
            'index': idx,
            'type': item_type,
            'position': {
                'x': item.x,
                'y': item.y
            },
            'size': item.size,
            'point': item.point,
            'is_move': item.is_move,
            'is_explosive': item.is_explosive
        }
        
        # Thêm subtype nếu có
        if item_subtype:
            item_data['subtype'] = item_subtype
        
        # Thêm thông tin đặc biệt cho từng loại item
        if hasattr(item, 'direction'):  # Mole
            item_data['direction'] = item.direction
            if hasattr(item, 'ranges'):
                item_data['ranges'] = item.ranges
        
        if hasattr(item, 'lucky'):  # QuestionBag
            item_data['lucky'] = item.lucky
        
        items_state.append(item_data)
    
    # 6. State của Explosive (nếu đang phát nổ)
    explosive_state = None
    if game_scene.play_Explosive and game_scene.explosive is not None:
        explosive_state = {
            'active': True,
            'is_exit': hasattr(game_scene.explosive, 'is_exit') and game_scene.explosive.is_exit
        }
    
    # Tổng hợp tất cả state
    complete_state = {
        'timestamp': datetime.now().isoformat(),
        'global_state': global_state,
        'scene_state': scene_state,
        'miner_state': miner_state,
        'rope_state': rope_state,
        'items': items_state,
        'explosive_state': explosive_state
    }
    
    return complete_state


def _get_miner_state_name(state_code):
    """Chuyển đổi state code thành tên dễ đọc"""
    state_names = {
        0: 'swinging',
        1: 'expanding',
        2: 'retracting',
        3: 'yeah',
        4: 'TNT'
    }
    return state_names.get(state_code, 'unknown')


def save_state_to_json(state, filename='state.json'):
    """
    Lưu state dict vào file JSON
    
    Args:
        state: Dictionary chứa game state (từ hàm get_game_state)
        filename: Tên file để lưu (mặc định: 'state.json')
    
    Returns:
        bool: True nếu lưu thành công, False nếu có lỗi
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"❌ Lỗi khi lưu state vào {filename}: {e}")
        return False
    
def json_to_state(filename='state.json'):
    """
    Đọc state từ file JSON và trả về dict
    
    Args:
        filename: Tên file JSON chứa state (mặc định: 'state.json')
    
    Returns:
        dict: Dictionary chứa game state, hoặc None nếu có lỗi
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            state = json.load(f)
        return state
    except Exception as e:
        print(f"❌ Lỗi khi đọc state từ {filename}: {e}")
        return None
