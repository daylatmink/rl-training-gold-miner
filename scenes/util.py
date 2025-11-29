from define import *
from entities.gold import Gold
from entities.tnt import TNT
from entities.other import Other
from entities.rock import Rock
from entities.mole import Mole
from entities.question import QuestionBag
import datetime

# THÊM IMPORT
from .level_generator import LevelGenerator, ProceduralLevelManager
# KHỞI TẠO LEVEL MANAGER (thêm vào sau imports)
level_generator = LevelGenerator()
level_manager = ProceduralLevelManager(level_generator)

# Kiểm tra va chạm giữa dây và item
def is_collision(rope, item):
    """Kiểm tra va chạm giữa rope và item"""
    try:
        # Đảm bảo cả rope.hoo và item đều có rect
        if hasattr(rope, 'hoo') and hasattr(rope.hoo, 'rect') and hasattr(item, 'rect'):
            if rope.hoo.rect.colliderect(item.rect) and rope.state == 'expanding':
                return True
        return False
    except Exception as e:
        print(f"Collision error: {e}")
        return False
def explosive_item(tnt, items):
    items_to_remove = []
    for item in items:
        if item == tnt:
            continue
        if math.sqrt(pow(abs(item.x-tnt.x),2) + pow(abs(item.y-tnt.y),2)) < 200:
            items_to_remove.append(item)
    for item in items_to_remove:
        items.remove(item)
        
def load_item(item_data,is_clover=False,is_gem=False,is_rock=False):
    item_name = item_data["type"]
    x = item_data["pos"]["x"]
    y = item_data["pos"]["y"]
    item = None
    match item_name:
        case "MiniGold":
            item = Gold(x,y,30,MiniGold_point)
        case "NormalGold":
            item = Gold(x,y,70,NormalGold_point)
        case "NormalGoldPlus":
            item = Gold(x,y,90,NormalGoldPlus_point)
        case "BigGold":
            item = Gold(x,y,150,BigGold_point)
        case "MiniRock":
            if is_rock:
                item = Rock(x,y,30,MiniRock_point*3)
            else: item = Rock(x,y,30,MiniRock_point)
        case "NormalRock":
            if is_rock:
                item = Rock(x,y,60,NormalRock_point*3)
            else:
                item = Rock(x,y,60,NormalRock_point)
        case "QuestionBag":
            if is_clover:
                item = QuestionBag(x,y,lucky=2)
            else: item = QuestionBag(x,y,lucky=1)
        case "Diamond":
            if is_gem:
                item = Other(x,y,diamond_image,int(Diamond_point*1.5))
            else: item = Other(x,y,diamond_image,Diamond_point)
        case "Mole":
            item = Mole(x,y,mole_image,Mole_point,direction=item_data["dir"])
        case "MoleWithDiamond":
            if is_gem:
                item = Mole(x,y,mole2_image,int(Diamond_point*1.5)+2,direction=item_data["dir"])
            else:
                item = Mole(x,y,mole2_image,MoleWithDiamond_point,direction=item_data["dir"])
        case "Skull":
            item = Other(x,y,skull_image,Skull_point)
        case "Bone":
            item = Other(x,y,bone_image,Bone_point)
        case "TNT":
            item = TNT(x,y)
        case _:
            print("None")
            item = None
    return item
def load_items(items_data,is_clover=False,is_gem=False,is_rock=False):
    items = []
    for item in items_data:
        # if(item != None):
        items.append(load_item(item,is_clover,is_gem,is_rock))
    return items
def load_level(level, is_clover=False, is_gem=False, is_rock=False):
    """Load level data - hỗ trợ cả original và generated levels"""
    
    # KIỂM TRA NẾU LÀ GENERATED LEVEL
    if level.startswith('GEN_') or level.startswith('TRAIN_'):
        try:
            # 🎯 ÁNH XẠ DIFFICULTY DỰA TRÊN LEVEL
            if isinstance(level, int) and level == 0:
                difficulty = "train"  # Level 0 = train difficulty
            elif level.startswith('TRAIN_'):
                difficulty = "train"  # TRAIN_ prefix = train difficulty
            else:
                difficulty = "medium"  # Mặc định
            
            level_data = level_manager.get_level(level, difficulty)
            
            if level_data:
                # Load background (sử dụng background mặc định)
                try:
                    from define import backgrounds
                    bg = backgrounds[0] if backgrounds else None
                except (ImportError, AttributeError):
                    bg = None
                    print("⚠️ Warning: backgrounds not found")
                
                # Tạo items từ level_data
                items = create_entities_from_data(level_data['entities'], is_clover, is_gem, is_rock)
                return bg, items
            else:
                # Fallback: sử dụng level mặc định nếu không tạo được
                print(f"❌ LOAD_LEVEL: Generated level {level} not found, using default")
                return load_level("L1-1", is_clover, is_gem, is_rock)
                
        except Exception as e:
            print(f"❌ LOAD_LEVEL: Error loading generated level {level}: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to default level
            return load_level("L1-1", is_clover, is_gem, is_rock)
    
    # XỬ LÝ ORIGINAL LEVELS (code cũ)
    try:
        from define import data, backgrounds
        
        if level not in data:
            print(f"❌ LOAD_LEVEL: Level {level} not found in data, using L1_1")
            level = "L1_1"
        
        entities_data = data[level]['entities']
        
        # Parse background từ 'type' (LevelA, LevelB, LevelC, LevelD)
        level_type = data[level].get('type', 'LevelA')
        bg_map = {'LevelA': 0, 'LevelB': 1, 'LevelC': 2, 'LevelD': 3}
        bg_index = bg_map.get(level_type, 0)
        bg = backgrounds[bg_index] if bg_index < len(backgrounds) else backgrounds[0]
        
        items = create_entities_from_data(entities_data, is_clover, is_gem, is_rock)
        return bg, items
        
    except Exception as e:
        print(f"❌ LOAD_LEVEL: Error loading original level {level}: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback cứng
        try:
            from define import backgrounds
            bg = backgrounds[0] if backgrounds else None
        except:
            bg = None
        
        return bg, []  # Trả về list rỗng nếu lỗi
def create_entities_from_data(entities_data, is_clover=False, is_gem=False, is_rock=False):
    """Tạo entities từ dữ liệu entities_data - hỗ trợ cả 2 định dạng"""
    items = []  # Sử dụng list thay vì pygame.sprite.Group
    
    for i, entity in enumerate(entities_data):
        try:
            # XỬ LÝ ĐỊNH DẠNG ORIGINAL LEVELS (có 'type', 'pos')
            if 'type' in entity and 'pos' in entity:
                entity_type = entity['type']
                x = entity['pos']['x']
                y = entity['pos']['y']
                
                # Sử dụng hàm load_item cũ cho original levels
                item = load_item(entity, is_clover, is_gem, is_rock)
                if item:
                    items.append(item)
                    
            # XỬ LÝ ĐỊNH DẠNG GENERATED LEVELS (có 'type', 'x', 'y')  
            elif 'x' in entity and 'y' in entity and 'type' in entity:
                entity_type = entity['type']
                x = entity['x']
                y = entity['y']
                value = entity.get('value', 0)
                
                # Xử lý các loại entity cho generated levels
                if entity_type == 'gold':
                    size = entity.get('size', 1)
                    point_value = value if value > 0 else MiniGold_point
                    if size == 1:
                        items.append(Gold(x, y, 30, point_value, is_rock))
                    elif size == 2:
                        items.append(Gold(x, y, 70, point_value, is_rock))
                    elif size == 3:
                        items.append(Gold(x, y, 150, point_value, is_rock))
                elif entity_type == 'stone':
                    size = entity.get('size', 1)
                    point_value = value if value > 0 else MiniRock_point
                    if size == 1:
                        if is_rock:
                            items.append(Rock(x, y, 30, point_value * 3))
                        else:
                            items.append(Rock(x, y, 30, point_value))
                    elif size == 2:
                        if is_rock:
                            items.append(Rock(x, y, 60, point_value * 3))
                        else:
                            items.append(Rock(x, y, 60, point_value))
                # ... (phần xử lý generated levels giữ nguyên)
                    
        except Exception as e:
            print(f"ERROR creating entity {entity}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return items
def random_level(level_number, use_generated=False):
    """Chọn level ngẫu nhiên - có thể dùng generated levels"""
    if use_generated:
        # Sử dụng level được generate ngẫu nhiên
        # 🎯 ÁNH XẠ: Level 0 = train, Level 1+ = easy/medium/hard/expert
        if level_number == 0:
            difficulty = "train"  # CHỈ 1 ITEM NGẪU NHIÊN
        else:
            difficulties = ["easy", "medium", "hard", "expert"]
            difficulty = difficulties[min(level_number - 1, len(difficulties) - 1)]
        
        level_id = f"RANDOM_{level_number}_{random.randint(1000, 9999)}"
        return level_id
    else:
        # Sử dụng level từ file JSON cũ
        ran_level = random.randint(1, 3)
        level_text = "L" + str(level_number) + "_" + str(ran_level)
        return level_text
def draw_point(rope,dt,miner):
    if rope.text == "dynamite" and rope.text_direction !="None":
        rope.time_text -= dt
        if rope.x_text > 500:
            rope.text_size += dt*rope.speed /(5)
        elif rope.text_size > 30 and rope.text_size < 46:
            rope.time_text = 0.4
            rope.text_size -= dt*rope.speed /(5)
        elif rope.text_size > 16 and rope.time_text < 0:
            rope.text_size -= dt*rope.speed /(25)
        if rope.time_text < 0:
            if rope.text_direction == "left":
                rope.x_text -= dt * rope.speed
                if rope.x_text <= 500:  # Reached left boundary, change direction
                    rope.text_direction = "right"
            elif rope.text_direction == "right":
                rope.x_text += dt * rope.speed
                if rope.x_text >= 700:  # Reached right boundary, change direction
                    rope.text_direction = "None"
        screen.blit(dynamite_image,(rope.x_text,10))
    elif rope.text == "strength" and rope.text_direction !="None":
        rope.time_text -= dt
        miner.state = 3
        if rope.x_text > 400:
            rope.text_size += dt*rope.speed /(8)
        elif rope.text_size > 30 and rope.text_size < 46:
            rope.time_text = 0.4
            rope.text_size -= dt*rope.speed /(5)
        elif rope.text_size > 16 and rope.time_text < 0:
            rope.text_size -= dt*rope.speed
        if rope.time_text < 0:
            if rope.text_direction == "left":
                rope.x_text -= dt * rope.speed
                if rope.x_text <= 400:  # Reached left boundary, change direction
                    rope.text_direction = "right"
            elif rope.text_direction == "right":
                rope.text_size -= dt*rope.speed /(5)
                if rope.text_size <= 0:  # Reached right boundary, change direction
                    miner.state = 3
                    rope.text_direction = "None"
        text_font = pygame.font.Font(os.path.join("assets", "fonts", 'Fernando.ttf'), int(rope.text_size))
        screen.blit(text_font.render("Sức mạnh", True, (0, 15, 0)), (rope.x_text, rope.y_text))
    elif rope.text != "" and rope.x_text > 120 and rope.text_direction !="None": # show tiền
        rope.time_text -= dt
        if rope.x_text > 500:
            rope.text_size += dt*rope.speed /(5)
        elif rope.text_size > 30 and rope.text_size < 46:
            rope.time_text = 0.2
            rope.text_size -= dt*rope.speed /(5)
        elif rope.text_size > 16 and rope.time_text < 0:
            rope.text_size -= dt*rope.speed /(25)
        if rope.time_text < 0:
            rope.x_text -= dt*rope.speed
        text_font = pygame.font.Font(os.path.join("assets", "fonts", 'Fernando.ttf'), int(rope.text_size))
        screen.blit(text_font.render("+$"+rope.text, True, (0, 15, 0)), (rope.x_text, rope.y_text))

# tminh
def generate_random_level(difficulty="medium", level_id=None):
    """Tạo level ngẫu nhiên cho training hoặc custom game"""
    if level_id is None:
        level_id = f"CUSTOM_{random.randint(1000, 9999)}"
    
    level_data = level_manager.get_level(level_id, difficulty)
    return level_id, level_data

def get_training_level(episode=None):
    """Lấy level cho RL training"""
    if episode is None:
        level_data = level_manager.get_level(f"TRAIN_{random.randint(1, 10000)}")
    else:
        # Curriculum learning: tăng difficulty theo episode
        if episode < 1000:
            difficulty = "easy"
        elif episode < 5000:
            difficulty = "medium"
        elif episode < 10000:
            difficulty = "hard"
        else:
            difficulty = "expert"
        level_data = level_manager.get_level(f"TRAIN_{episode}", difficulty)
    
    return level_data
def blit_text(surface, text, pos, font, color=pygame.Color('black')):
    words = [word.split(' ') for word in text.splitlines()]  # 2D array where each row is a list of words.
    space = font.size(' ')[0]  # The width of a space.
    max_width, max_height = surface.get_size()
    x, y = pos
    for line in words:
        for word in line:
            word_surface = font.render(word, 0, color)
            word_width, word_height = word_surface.get_size()
            if x + word_width >= max_width:
                x = pos[0]  # Reset the x.
                y += word_height  # Start on new row.
            surface.blit(word_surface, (x, y))
            x += word_width + space
        x = pos[0]  # Reset the x.
        y += word_height  # Start on new row.
def blit_nor_text(surface, text_in, pos, font, color=pygame.Color('black')):
    text = font.render(text_in, True, color)
    surface.blit(text,text.get_rect(center = pos))
def is_enough_money(item_price):
    if item_price > get_score():
        return False
    return True
def buy_item(item_id,price):
    match item_id:
        case 1: #rock_collectors_book
            if is_enough_money(price):
                set_score(get_score()-price)
                return True
            else: return False
        case 2: #strength_drink
            if is_enough_money(price):
                set_score(get_score()-price)
                return True
            else: return False
        case 3: #gem_polish
            if is_enough_money(price):
                set_score(get_score()-price)
                return True
            else: return False
        case 4: #clover
            if is_enough_money(price):
                set_score(get_score()-price)
                return True
            else: return False
        case 5: #dynamite
            if is_enough_money(price):
                set_score(get_score()-price)
                return True
            else: return False

def get_high_score_from_file():
    high_scores = []
    with open(high_score_file, "r") as file:
        lines = file.readlines()
        for line in lines:
            time_score = line.strip().split(": ")
            time = time_score[0]
            score = int(time_score[1])
            high_scores.append({"time": time, "score": score})
    return high_scores
def get_high_score_as_text():
    high_scores = get_high_score_from_file()
    text = ""
    for score in high_scores:
        text += str(score["time"])+"          "+str(score["score"]) + "\n"
    if text == "":
        text = "Chưa có danh sách điểm cao"
    return text
def write_high_score(score):
    # Đọc danh sách điểm cao từ file
    high_scores = get_high_score_from_file()
    # Lấy thời gian hiện tại
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Thêm điểm cao mới vào danh sách
    high_scores.append({"time": current_time, "score": score})

    # Sắp xếp danh sách theo điểm số giảm dần
    high_scores = sorted(high_scores, key=lambda x: x["score"], reverse=True)

    # Giới hạn chỉ lưu 3 điểm cao
    high_scores = high_scores[:3]

    # Ghi danh sách điểm cao vào file
    with open(high_score_file, "w") as file:
        for score in high_scores:
            file.write(f"{score['time']}: {score['score']}\n")

pygame.mixer.set_num_channels(8)
voice1 = pygame.mixer.Channel(1)
voice2 = pygame.mixer.Channel(2)
voice3 = pygame.mixer.Channel(3)
voice4 = pygame.mixer.Channel(4)
voice5 = pygame.mixer.Channel(5)
voice6 = pygame.mixer.Channel(6)
def load_sound(sound_name):
    match sound_name:
        case "explosive_sound":
            pygame.mixer.stop()
            voice1.play(explosive_sound)
        case "goal_sound":
            pygame.mixer.stop()
            voice2.play(goal_sound)
        case "grab_back_sound":
            voice4.stop()
            if not voice3.get_busy():
                voice3.play(grab_back_sound)
        case "grab_start_sound":
            if not voice4.get_busy():
                voice4.play(grab_start_sound)
        case "hook_reset_sound":
            voice3.stop()
            if not voice5.get_busy() or not voice1.get_busy():
                voice5.play(hook_reset_sound)
        case "high_value_sound":
            high_value_sound.play()
        case "normal_value_sound":
            normal_value_sound.play()
        case "money_sound":
            money_sound.play()
        case "made_goal_sound":
            pygame.mixer.stop()
            made_goal_sound.play()

#utility
def setup_training_levels(num_levels=1000):
    """Khởi tạo pool level cho training"""
    return level_manager.generate_infinite_levels("TRAIN", num_levels)

def get_level_difficulty(level_data):
    """Xác định difficulty của level dựa trên entity distribution"""
    entities = level_data['entities']
    
    # Đếm số lượng entity types
    gold_count = sum(1 for e in entities if "Gold" in e["type"])
    rock_count = sum(1 for e in entities if "Rock" in e["type"])
    enemy_count = sum(1 for e in entities if "Mole" in e["type"] or "TNT" in e["type"])
    diamond_count = sum(1 for e in entities if "Diamond" in e["type"])
    
    total_entities = len(entities)
    
    # Phân loại difficulty
    gold_ratio = gold_count / total_entities if total_entities > 0 else 0
    obstacle_ratio = (rock_count + enemy_count) / total_entities if total_entities > 0 else 0
    
    if gold_ratio > 0.6 and obstacle_ratio < 0.2:
        return "easy"
    elif gold_ratio > 0.4 and obstacle_ratio < 0.4:
        return "medium"
    elif gold_ratio > 0.2 and obstacle_ratio < 0.6:
        return "hard"
    else:
        return "expert"